from __future__ import division
import numpy
import theano.tensor as T
import theano
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import six.moves.cPickle as pickle
import xlsxwriter
import timeit
import os
import sys
from convolutional_mlp import LeNetConvPoolLayer, LogisticRegression, HiddenLayer
from fashion_mnist import load_mnist
import scipy.io


def theano_translation_updating(image_tensor_input, horizon_disp, verti_disp):
    tx = image_tensor_input

    def vertical_shift(image_input, displacement):

        txout1 = T.zeros_like(image_input)

        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, 0:27 - displacement, :], image_input[:, :, displacement:27, :])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, -displacement:27, :], image_input[:, :,  0:27 + displacement, :])
        else:
            txout1 = image_input
        return txout1

    def horizontal_shift(image_input, displacement):

        txout1 = T.zeros_like(image_input)
        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, :, displacement:27], image_input[:, :, :, 0:27 -displacement])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, :, 0:27 +displacement], image_input[:, :, :, -displacement:27])
        else:
            txout1 = image_input
        return txout1

    if verti_disp != 0 and horizon_disp == 0:
        txout = vertical_shift(tx, verti_disp)

    if horizon_disp != 0 and verti_disp == 0:
        txout = horizontal_shift(tx, horizon_disp)

    if horizon_disp != 0 and verti_disp != 0:
        txout = vertical_shift(tx, verti_disp)
        txout = horizontal_shift(txout, horizon_disp)

    if verti_disp == 0 and horizon_disp == 0:
        txout = tx

    #txout = T.Rebroadcast((1, True))(txout)
    txout = txout.eval()
    return txout


def loaddata_mnist():

    def shared_dataset(data_x, data_y, borrow=True):
        # 0-9 Label Representation
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    path = os.getcwd()

    train_set_x, train_set_y = load_mnist(path=path, kind='train')

    valid_set_x, valid_set_y = shared_dataset(train_set_x[50000:60000], train_set_y[50000:60000])
    train_set_x, train_set_y = shared_dataset(train_set_x[0:50000], train_set_y[0:50000])

    test_set_x, test_set_y = load_mnist(path=path, kind='t10k')
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def random_epoch_train_pt(learning_rate=0.05, weight_decay=0.001, n_epochs=200, batch_size=500, name='Fashion'):

    pre_trained_name = 'FashionMnist_0.05_0.001_[20, 30]no_decay_tanh'

    datasets = loaddata_mnist()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    # print(str(n_train), str(n_valid),str(n_test))

    test_set_x = test_set_x.reshape((n_test, 1, 28, 28))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    train_set_x = train_set_x.reshape((n_train, 1, 28, 28))

    temp_train_set_x = theano.shared(numpy.zeros(train_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_train_set_xx = T.Rebroadcast((1, True))(temp_train_set_x)

    temp_valid_set_x = theano.shared(numpy.zeros(valid_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_valid_set_xx = T.Rebroadcast((1, True))(temp_valid_set_x)

    temp_test_set_x = theano.shared(numpy.zeros(test_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_test_set_xx = T.Rebroadcast((1, True))(temp_test_set_x)

    n_train_batches = n_train // batch_size
    n_valid_batches = n_valid // batch_size
    n_test_batches = n_test // batch_size

    y = T.ivector('y')
    index = T.lscalar()

    dummy = T.ftensor4('dummy')

    update_train = (temp_train_set_x, dummy)
    update_valid = (temp_valid_set_x, dummy)
    update_test = (temp_test_set_x, dummy)

    replace_train = theano.function([dummy],temp_train_set_x, updates=[update_train])
    replace_valid = theano.function([dummy],temp_valid_set_x, updates=[update_valid])
    replace_test = theano.function([dummy],temp_test_set_x, updates=[update_test])

    print('... loading the model')
    with open(pre_trained_name + '.pkl', 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
        for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.995

    rand_trans_x = numpy.random.random_integers(-10, 10, 200)
    rand_trans_y = numpy.random.random_integers(-10, 10, 200)
    numpy.save('rand_trans_x.npy', rand_trans_x)
    numpy.save('rand_trans_y.npy', rand_trans_y)
    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: temp_test_set_xx[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: temp_valid_set_xx[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            layer0.input: temp_train_set_xx[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    start_time = timeit.default_timer()

    print('... training')

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 20000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        horizontal = rand_trans_x[epoch]
        vertical = rand_trans_y[epoch]

        tran_test_set_x = theano_translation_updating(test_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))
        tran_valid_set_x = theano_translation_updating(valid_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))
        tran_train_set_x = theano_translation_updating(train_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))

        replace_test(tran_test_set_x)
        replace_valid(tran_valid_set_x)
        replace_train(tran_train_set_x)

        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('Horizontal Shift:', horizontal, 'Vertical Shift:', vertical)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                error_line[epoch - 1] = this_validation_loss

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    [t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3] = \
                        [layer0, layer1, layer2_input, layer2, layer3]

            if patience <= iter:
                done_looping = True
                break

    with open(name + '.pkl', 'wb') as f:
        pickle.dump([t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3], f)

    error_line = error_line[0:epoch-1]*100
    scipy.io.savemat(name+'.mat', mdict={'Error_Spectrum': error_line})

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def random_epoch_train_pt_full(learning_rate=0.05, weight_decay=0.001, n_epochs=200, batch_size=500, name='Fashion'):

    pre_trained_name = 'FashionMnist_0.05_0.001_[20, 30]no_decay_tanh'

    datasets = loaddata_mnist()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    # print(str(n_train), str(n_valid),str(n_test))

    test_set_x = test_set_x.reshape((n_test, 1, 28, 28))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    train_set_x = train_set_x.reshape((n_train, 1, 28, 28))

    temp_train_set_x = theano.shared(numpy.zeros(train_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_train_set_xx = T.Rebroadcast((1, True))(temp_train_set_x)

    temp_valid_set_x = theano.shared(numpy.zeros(valid_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_valid_set_xx = T.Rebroadcast((1, True))(temp_valid_set_x)

    temp_test_set_x = theano.shared(numpy.zeros(test_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_test_set_xx = T.Rebroadcast((1, True))(temp_test_set_x)

    n_train_batches = n_train // batch_size
    n_valid_batches = n_valid // batch_size
    n_test_batches = n_test // batch_size

    y = T.ivector('y')
    index = T.lscalar()

    dummy = T.ftensor4('dummy')

    update_train = (temp_train_set_x, dummy)
    update_valid = (temp_valid_set_x, dummy)
    update_test = (temp_test_set_x, dummy)

    replace_train = theano.function([dummy], temp_train_set_x, updates=[update_train])
    replace_valid = theano.function([dummy], temp_valid_set_x, updates=[update_valid])
    replace_test = theano.function([dummy], temp_test_set_x, updates=[update_test])

    print('... loading the model')
    with open(pre_trained_name + '.pkl', 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
        for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.995

    rand_trans_x = numpy.random.random_integers(-10, 10, 200)
    rand_trans_y = numpy.random.random_integers(-10, 10, 200)
    numpy.save('rand_trans_x.npy', rand_trans_x)
    numpy.save('rand_trans_y.npy', rand_trans_y)
    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: temp_test_set_xx[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: temp_valid_set_xx[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            layer0.input: temp_train_set_xx[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    start_time = timeit.default_timer()

    print('... training')

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 20000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        horizontal = rand_trans_x[epoch]
        vertical = rand_trans_y[epoch]

        tran_test_set_x = theano_translation_updating(test_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))
        tran_valid_set_x = theano_translation_updating(valid_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))
        tran_train_set_x = theano_translation_updating(train_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))

        replace_test(tran_test_set_x)
        replace_valid(tran_valid_set_x)
        replace_train(tran_train_set_x)

        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('Horizontal Shift:', horizontal, 'Vertical Shift:', vertical)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                error_line[epoch - 1] = this_validation_loss

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    [t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3] = \
        [layer0, layer1, layer2_input, layer2, layer3]

    with open(name + '.pkl', 'wb') as f:
        pickle.dump([t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3], f)

    error_line = error_line[0:epoch - 1] * 100
    scipy.io.savemat(name + '.mat', mdict={'Error_Spectrum': error_line})

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def random_epoch_train_begining(learning_rate=0.05, weight_decay=0.001, nkerns=[20, 30], n_epochs=200, batch_size=500, name_given ='test'):

    #name = 'FashionMnist_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns) + 'Rand_Trans_Relu2_Begin'
    name = name_given
    rng = numpy.random.RandomState(23455)
    datasets = loaddata_mnist()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    # print(str(n_train), str(n_valid),str(n_test))

    test_set_x = test_set_x.reshape((n_test, 1, 28, 28))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    train_set_x = train_set_x.reshape((n_train, 1, 28, 28))

    temp_train_set_x = theano.shared(numpy.zeros(train_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_train_set_xx = T.Rebroadcast((1, True))(temp_train_set_x)

    temp_valid_set_x = theano.shared(numpy.zeros(valid_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_valid_set_xx = T.Rebroadcast((1, True))(temp_valid_set_x)

    temp_test_set_x = theano.shared(numpy.zeros(test_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_test_set_xx = T.Rebroadcast((1, True))(temp_test_set_x)

    n_train_batches = n_train // batch_size
    n_valid_batches = n_valid // batch_size
    n_test_batches = n_test // batch_size

    x = T.fmatrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    dummy = T.ftensor4('dummy')

    update_train = (temp_train_set_x, dummy)
    update_valid = (temp_valid_set_x, dummy)
    update_test = (temp_test_set_x, dummy)

    replace_train = theano.function([dummy], temp_train_set_x, updates=[update_train])
    replace_valid = theano.function([dummy], temp_valid_set_x, updates=[update_valid])
    replace_test = theano.function([dummy], temp_test_set_x, updates=[update_test])

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 28, 28))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2),
        activation=T.nnet.relu
    )

    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
        for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.995

    start_time = timeit.default_timer()

    rand_trans_x = numpy.random.random_integers(-10, 10, 200)
    rand_trans_y = numpy.random.random_integers(-10, 10, 200)
    numpy.save('rand_trans_x.npy', rand_trans_x)
    numpy.save('rand_trans_y.npy', rand_trans_y)
    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: temp_test_set_xx[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: temp_valid_set_xx[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            layer0.input: temp_train_set_xx[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})


    print('... training')

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 20000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):

        horizontal = rand_trans_x[epoch]
        vertical = rand_trans_y[epoch]

        tran_test_set_x = theano_translation_updating(test_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))
        tran_valid_set_x = theano_translation_updating(valid_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))
        tran_train_set_x = theano_translation_updating(train_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))

        replace_test(tran_test_set_x)
        replace_valid(tran_valid_set_x)
        replace_train(tran_train_set_x)

        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('Horizontal Shift:', horizontal, 'Vertical Shift:', vertical)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                error_line[epoch - 1] = this_validation_loss

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    [t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3] = \
        [layer0, layer1, layer2_input, layer2, layer3]

    with open(name + '.pkl', 'wb') as f:
        pickle.dump([t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3], f)

    error_line = error_line[0:epoch-1]*100
    scipy.io.savemat(name+'.mat', mdict={'Error_Spectrum': error_line})

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == "__main__":
    #random_epoch_train_begining()
    #ordered_training()
    #for x in range(1, 11, 1):
    #    name = 'FashionMnist_0.05_0.001_[20, 30]Rand_Trans_tanh_Begin_'+str(x)
    #    random_epoch_train_begining(name_given=name)
    #loop_all()
    #random_rotation_epoch_train()
    for x in range(1, 11, 1):
        name = 'FashionMnist_0.05_0.001_[20, 30]Rand_Trans_Relu1_Begin_Full_' +str(x)
        random_epoch_train_begining(name_given=name)
