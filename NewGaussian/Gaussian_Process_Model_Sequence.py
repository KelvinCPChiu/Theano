from __future__ import division
import numpy
import theano.tensor as T
import theano
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import six.moves.cPickle as pickle
import timeit
import scipy.io
import matplotlib.pyplot as plt
from Gaussian_Process_Model import LogisticRegression, HiddenLayer, LeNetConvPoolLayer
from Gaussian_Process_Model import Generate_Set_ez, Generate_Test_Set_ez
from Weight_Check import Weight_Open


def weight_read(model):

    [layer0, layer1, layer2, layer3] = model

    layer_0_weight_matrix = numpy.array(layer0.W.eval())
    layer_0_b_value = numpy.array(layer0.b.eval())

    layer_1_weight_matrix = numpy.array(layer1.W.eval())
    layer_1_b_value = numpy.array(layer1.b.eval())

    layer_2_weight_matrix = numpy.array(layer2.W.eval())
    layer_2_b_value = numpy.array(layer2.b.eval())

    layer_3_weight_matrix = numpy.array(layer3.W.eval())
    layer_3_b_value = numpy.array(layer3.b.eval())

    rval = [(layer_3_weight_matrix, layer_3_b_value),
            (layer_2_weight_matrix, layer_2_b_value),
            (layer_1_weight_matrix, layer_1_b_value),
            (layer_0_weight_matrix, layer_0_b_value)]

    return rval


def cross_corre_value(weight_1, weight_2):

    denominator = float(1 / weight_1.shape[3] / weight_1.shape[2])

    weight_mean = numpy.sum(numpy.sum(weight_1, axis=3), axis=2) * denominator

    weight_2_mean = numpy.sum(numpy.sum(weight_2, axis=3), axis=2) * denominator

    sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(weight_1 ** 2, axis=3), axis=2) *
                                    denominator - weight_mean ** 2))
    sd2 = numpy.sqrt(
        numpy.absolute(numpy.sum(numpy.sum(weight_2 ** 2, axis=3), axis=2) *
                       denominator - weight_2_mean ** 2))
    sd = sd1 * sd2

    r_value = (numpy.sum(numpy.sum(weight_1 * weight_2, axis=3), axis=2)
               * denominator - weight_mean * weight_2_mean) / sd

    return r_value


def normalized_cross_correlation(Model_1, Model_2, layer0_r, layer1_r):

    # R.M.S of the filter normalized cross correlation

    r_value_two_layers = numpy.zeros(4)

    for layer_number in range(0, 4, 1):

        Model_1_Layer = Model_1[3 - layer_number][0]
        Model_2_Layer = Model_2[3 - layer_number][0]

        if layer_number <= 1:

            Model_1_filter_number = Model_1_Layer.shape[0]
            filter_z = Model_1_Layer.shape[1]
            filter_y = Model_1_Layer.shape[2]
            filter_x = Model_1_Layer.shape[3]

            denominator = float(1 / filter_y / filter_x)

            Model_1_Weight_mean = numpy.sum(numpy.sum(Model_1_Layer, axis=3), axis=2) * denominator

            Model_2_Weight_mean = numpy.sum(numpy.sum(Model_2_Layer, axis=3), axis=2) * denominator
            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(Model_1_Layer ** 2, axis=3), axis=2) *
                                            denominator - Model_1_Weight_mean ** 2))
            sd2 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(Model_2_Layer ** 2, axis=3), axis=2) *
                                            denominator - Model_2_Weight_mean ** 2))

            sd = sd1 * sd2

            temp_r_value = (numpy.sum(numpy.sum(Model_1_Layer * Model_2_Layer, axis=3), axis=2)
                            * denominator - Model_1_Weight_mean * Model_2_Weight_mean) / sd

            if layer_number == 0:
                temp_r_value = temp_r_value*layer0_r
                size_value = numpy.sum(layer0_r)
            else:
                temp_r_value = temp_r_value*layer1_r
                size_value = numpy.sum(layer1_r)

            r_value = numpy.sqrt(numpy.sum(numpy.sum(temp_r_value ** 2, axis=1), axis=0) / size_value)
            #   (filter_z * Model_1_filter_number - size_value))

            r_value_two_layers[layer_number] = r_value

        if layer_number >= 2:

            filter_y = Model_1_Layer.shape[1]
            filter_x = Model_1_Layer.shape[0]

            denominator = float(1 / filter_y / filter_x)
            Model_1_Weight_mean = numpy.sum(numpy.sum(Model_1_Layer, axis=1), axis=0) * denominator
            Model_2_Weight_mean = numpy.sum(numpy.sum(Model_2_Layer, axis=1), axis=0) * denominator
            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(Model_1_Layer ** 2, axis=1), axis=0) *
                                            denominator - Model_1_Weight_mean ** 2))
            sd2 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(Model_2_Layer ** 2, axis=1), axis=0) *
                                            denominator - Model_2_Weight_mean ** 2))
            sd = sd1 * sd2
            temp_r_value = (numpy.sum(numpy.sum(Model_1_Layer * Model_2_Layer, axis=1), axis=0)
                            * denominator - Model_1_Weight_mean * Model_2_Weight_mean) / sd

            r_value_two_layers[layer_number] = temp_r_value

    #print(r_value_two_layers)
    return r_value_two_layers


def main_ver1_sqeu(learning_rate=0.05, weight_decay=0.001, n_epochs=2000, nkerns=[20, 30], data_set='Gaussian_Data_Set.npy', batch_size=500):

    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns)

    if data_set == 'Gaussian_White_Noise.npy':
        name += '_WN'

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng2 = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3
    datasets = numpy.load(data_set)

    train_set_x, train_set_y = Generate_Set_ez(datasets, 50000)

    valid_set_x, valid_set_y = Generate_Set_ez(datasets, 10000)

    test_set_x, test_set_y = Generate_Test_Set_ez(datasets, 10000)

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((n_test, 1, 28, 28))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    train_set_x = train_set_x.reshape((n_train, 1, 28, 28))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    y = T.fmatrix('y')
    index = T.lscalar()

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
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=numpy.round(nkerns[1] * 4 * 4/2).astype(int),
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=numpy.round(nkerns[1] * 4 * 4/2).astype(int), n_out=10)

    cost = layer3.sigmoid_cost_function(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 10
    improvement_threshold = 0.001

    start_time = timeit.default_timer()

    final_model = Weight_Open('Gaussian_Model_0.05_0.001_[20, 30]_WN.pkl')
    temp_model = weight_read([layer0, layer1, layer2, layer3])
    # Discard the unchanged kernels
    unlearnt_kernels_layer_0 = cross_corre_value(temp_model[3][0], final_model[3][0]) < 0.8
    unlearnt_kernels_layer_1 = cross_corre_value(temp_model[2][0], final_model[2][0]) < 0.8
    normalized_cross_value = normalized_cross_correlation(final_model, temp_model, unlearnt_kernels_layer_0, unlearnt_kernels_layer_1)

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 200000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False
    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            layer0.input: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))
                error_line[epoch-1] = this_validation_loss

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))

                temp_model = weight_read([layer0, layer1, layer2, layer3])
                temp_normalized_cross_value = normalized_cross_correlation(final_model, temp_model, unlearnt_kernels_layer_0, unlearnt_kernels_layer_1)
                normalized_cross_value = numpy.vstack((normalized_cross_value, temp_normalized_cross_value))

            if patience <= iter:
                done_looping = True
                break

    scipy.io.savemat('Normalized_Cross_Coe_epochs_WN.mat',  mdict={'CrossValueFilters': normalized_cross_value})
    #error_line = error_line[0:epoch-1]/100

    #scipy.io.savemat('Gaussian_Model_WN_0.05_weight.mat', mdict={'Error_Spectrum': error_line})

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == "__main__":
    main_ver1_sqeu(data_set='Gaussian_White_Noise.npy')
