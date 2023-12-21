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
from Gaussian_Process_Model import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, ConvPoolLayer_NoMaxPool


def printimage(test_set_x):
    # Print Image from tensor to numpy and plot it
    #mm = numpy.squeeze(test_set_x.eval(), axis=(0,))
    # print(mm)
    mm = test_set_x
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(mm)  # , cmap='gray')
    plt.axis('off')
    fig.savefig('figure1.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    return


def Generate_Set(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    def interpolation(input_image_1, input_image_2):

        morphing_coeff = numpy.random.random(input_image_1.shape[0])
        resultant_set = morphing_coeff[:, None, None] * input_image_1+(1-morphing_coeff)[:, None, None]*input_image_2

        return resultant_set, morphing_coeff

    def Cropping(input_image, set_size):
        # x_dim = input_image.shape[2]
        # y_dim = input_image.shape[1]
        x_dim_max = input_image.shape[2] - 28
        y_dim_max = input_image.shape[1] - 28

        cropping_x_dim = numpy.random.random_integers(0, x_dim_max, set_size)
        cropping_y_dim = numpy.random.random_integers(0, y_dim_max, set_size)

        image_label = numpy.random.random_integers(0, 9, set_size)
        output_image = numpy.zeros((set_size, 28, 28))
        for i in range(0, set_size, 1):
            output_image[i, :, :] = input_image[image_label[i],
                                                cropping_x_dim[i]:cropping_x_dim[i]+28,
                                                cropping_y_dim[i]:cropping_y_dim[i]+28]

        return output_image, image_label

    temp_image_1, temp_label_1 = Cropping(raw_image_set, size_desired)
    temp_image_2, temp_label_2 = Cropping(raw_image_set, size_desired)

    generated_image_set, morphing_constant = interpolation(temp_image_1, temp_image_2)
    number_of_classes = 10
    set_order1 = one_hot(temp_label_1, number_of_classes)
    set_order2 = one_hot(temp_label_2, number_of_classes)

    generated_label_set = set_order1*morphing_constant[:, None] + set_order2*((1-morphing_constant)[:, None])

    return shared_dataset(generated_image_set, generated_label_set)


def Generate_Set_ez(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    def interpolation(input_image_1, input_image_2):

        morphing_coeff = numpy.random.random(input_image_1.shape[0])
        resultant_set = morphing_coeff[:, None, None] * input_image_1 + (1-morphing_coeff)[:, None, None]*input_image_2

        return resultant_set, morphing_coeff

    def Cropping(input_image, set_size):
        # x_dim = input_image.shape[2]
        # y_dim = input_image.shape[1]
        x_dim_max = input_image.shape[2] - 28
        y_dim_max = input_image.shape[1] - 28

        cropping_x_dim = numpy.random.random_integers(0, x_dim_max, set_size)
        cropping_y_dim = numpy.random.random_integers(0, y_dim_max, set_size)

        image_label = numpy.random.random_integers(0, 9, set_size)
        output_image = numpy.zeros((set_size, 28, 28))
        for i in range(0, set_size, 1):
            output_image[i, :, :] = input_image[image_label[i],
                                                cropping_x_dim[i]:cropping_x_dim[i]+28,
                                                cropping_y_dim[i]:cropping_y_dim[i]+28]

        return output_image, image_label

    temp_image_1, temp_label_1 = Cropping(raw_image_set, size_desired)
    temp_image_2, temp_label_2 = Cropping(raw_image_set, size_desired)

    generated_image_set, morphing_constant = interpolation(temp_image_1, temp_image_2)
    number_of_classes = 10
    set_order1 = one_hot(temp_label_1, number_of_classes)
    set_order2 = one_hot(temp_label_2, number_of_classes)
    generated_label_set = set_order1 + set_order2
    generated_label_set = generated_label_set - (generated_label_set == 2)*generated_label_set/2
    #generated_label_set = set_order1*morphing_constant[:, None] + set_order2*((1-morphing_constant)[:, None])

    return shared_dataset(generated_image_set, generated_label_set)


def Generate_Set_ez_fixed_seq(raw_image_set, size_desired, seq1, seq2):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    def interpolation(input_image_1, input_image_2):

        morphing_coeff = numpy.random.random(input_image_1.shape[0])
        resultant_set = morphing_coeff[:, None, None] * input_image_1+(1-morphing_coeff)[:, None, None]*input_image_2

        return resultant_set, morphing_coeff

    def Cropping(input_image, set_size, random_sequence, name):
        x_dim_max = input_image.shape[2] - 28
        y_dim_max = input_image.shape[1] - 28

        if random_sequence is None:
            cropping_x_dim = numpy.random.random_integers(0, x_dim_max, set_size)
            cropping_y_dim = numpy.random.random_integers(0, y_dim_max, set_size)
            numpy.save('Order_'+name+'.npy', [cropping_x_dim, cropping_y_dim])
        else:
            cropping_x_dim = random_sequence[0]
            cropping_y_dim = random_sequence[1]

        image_label = numpy.random.random_integers(0, 9, set_size)
        output_image = numpy.zeros((set_size, 28, 28))
        for i in range(0, set_size, 1):
            output_image[i, :, :] = input_image[image_label[i], cropping_x_dim[i]:cropping_x_dim[i]+28, cropping_y_dim[i]:cropping_y_dim[i]+28]

        return output_image, image_label

    temp_image_1, temp_label_1 = Cropping(raw_image_set, size_desired, seq1, 'seq1')
    temp_image_2, temp_label_2 = Cropping(raw_image_set, size_desired, seq2, 'seq2')

    generated_image_set, morphing_constant = interpolation(temp_image_1, temp_image_2)
    number_of_classes = 10
    set_order1 = one_hot(temp_label_1, number_of_classes)
    set_order2 = one_hot(temp_label_2, number_of_classes)
    generated_label_set = set_order1 + set_order2
    generated_label_set = generated_label_set - (generated_label_set == 2)*generated_label_set/2

    return shared_dataset(generated_image_set, generated_label_set)


def Generate_Test_Set(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    def interpolation(input_image_1, input_image_2, input_image_3):

        morphing_coeff = numpy.random.random(input_image_1.shape[0])
        morphing_coeff2 = numpy.random.random(input_image_1.shape[0])
        morphing_coeff3 = numpy.random.random(input_image_1.shape[0])

        resultant_set = morphing_coeff[:, None, None] * input_image_1 + \
            morphing_coeff2[:, None, None] * input_image_2 + \
            morphing_coeff3[:, None, None] * input_image_3

        resultant_set = resultant_set / ((morphing_coeff3 + morphing_coeff2 + morphing_coeff)[:, None, None])

        return resultant_set, [morphing_coeff, morphing_coeff2, morphing_coeff3]

    def Cropping(input_image, set_size):
        x_dim_max = input_image.shape[2] - 28
        y_dim_max = input_image.shape[1] - 28

        cropping_x_dim = numpy.random.random_integers(0, x_dim_max, set_size)
        cropping_y_dim = numpy.random.random_integers(0, y_dim_max, set_size)

        image_label = numpy.random.random_integers(0, 9, set_size)
        output_image = numpy.zeros((set_size, 28, 28))
        for i in range(0, set_size, 1):
            output_image[i, :, :] = input_image[image_label[i],
                                                cropping_x_dim[i]:cropping_x_dim[i]+28,
                                                cropping_y_dim[i]:cropping_y_dim[i]+28]

        return output_image, image_label

    temp_image_1, temp_label_1 = Cropping(raw_image_set, size_desired)
    temp_image_2, temp_label_2 = Cropping(raw_image_set, size_desired)
    temp_image_3, temp_label_3 = Cropping(raw_image_set, size_desired)

    generated_image_set, morphing_constant = interpolation(temp_image_1, temp_image_2, temp_image_3)
    number_of_classes = 10
    set_order1 = one_hot(temp_label_1, number_of_classes)
    set_order2 = one_hot(temp_label_2, number_of_classes)
    set_order3 = one_hot(temp_label_3, number_of_classes)

    constant_sum = morphing_constant[0] + morphing_constant[1] + morphing_constant[2]
    generated_label_set = (set_order1 * morphing_constant[0][:, None] +
                           set_order2 * morphing_constant[1][:, None] +
                           set_order3 * morphing_constant[2][:, None])/constant_sum[:, None]

    return shared_dataset(generated_image_set, generated_label_set)


def Generate_Test_Set_ez(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    def interpolation(input_image_1, input_image_2, input_image_3):

        morphing_coeff = numpy.random.random(input_image_1.shape[0])
        morphing_coeff2 = numpy.random.random(input_image_1.shape[0])
        morphing_coeff3 = numpy.random.random(input_image_1.shape[0])

        resultant_set = morphing_coeff[:, None, None] * input_image_1 + \
            morphing_coeff2[:, None, None] * input_image_2 + \
            morphing_coeff3[:, None, None] * input_image_3

        resultant_set = resultant_set / ((morphing_coeff3 + morphing_coeff2 + morphing_coeff)[:, None, None])

        return resultant_set, [morphing_coeff, morphing_coeff2, morphing_coeff3]

    def Cropping(input_image, set_size):
        x_dim_max = input_image.shape[2] - 28
        y_dim_max = input_image.shape[1] - 28

        cropping_x_dim = numpy.random.random_integers(0, x_dim_max, set_size)
        cropping_y_dim = numpy.random.random_integers(0, y_dim_max, set_size)

        image_label = numpy.random.random_integers(0, 9, set_size)
        output_image = numpy.zeros((set_size, 28, 28))
        for i in range(0, set_size, 1):
            output_image[i, :, :] = input_image[image_label[i],
                                                cropping_x_dim[i]:cropping_x_dim[i]+28,
                                                cropping_y_dim[i]:cropping_y_dim[i]+28]

        return output_image, image_label

    temp_image_1, temp_label_1 = Cropping(raw_image_set, size_desired)
    temp_image_2, temp_label_2 = Cropping(raw_image_set, size_desired)
    temp_image_3, temp_label_3 = Cropping(raw_image_set, size_desired)

    generated_image_set, morphing_constant = interpolation(temp_image_1, temp_image_2, temp_image_3)
    number_of_classes = 10
    set_order1 = one_hot(temp_label_1, number_of_classes)
    set_order2 = one_hot(temp_label_2, number_of_classes)
    set_order3 = one_hot(temp_label_3, number_of_classes)
    generated_label_set = set_order1 + set_order2 + set_order3

    generated_label_set_temp = generated_label_set

    generated_label_set_temp[numpy.nonzero(generated_label_set == 3)] = 1
    generated_label_set_temp[numpy.nonzero(generated_label_set == 2)] = 1

    generated_label_set = generated_label_set_temp
    return shared_dataset(generated_image_set, generated_label_set)


def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(numpy.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                               dtype=theano.config.floatX))
        v_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                               dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta))
    updates.append((t, t + 1.))
    return updates


def adam_weight_decay(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8, weight_decay=0.00001):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(numpy.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                               dtype=theano.config.floatX))
        v_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                               dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta))
    updates.append((t, t + 1.))
    return updates


def main_ver1(learning_rate=0.001, weight_decay=0.001, n_epochs=2000, nkerns=[20, 30],
          data_set='Gaussian_Data_Set.npy', batch_size=500):

    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns) + '_adam_weight'

    if data_set == 'Gaussian_White_Noise.npy':
        name += '_WN'

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng2 = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3
    datasets = numpy.load(data_set)
    # default Setting is ez mode
    # weight = weight label set up, with ez marking
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

    with open(name + '_Initial.pkl', 'wb') as f:
        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

    cost = layer3.sigmoid_cost_function(y)
    #cost = layer3.mse_cost_function(y)
    params = layer3.params + layer2.params + layer1.params + layer0.params

    #grads = T.grad(cost, params)

    #updates = [
    #        (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
    #        for param_i, grad_i in zip(params, grads)]

    updates = adam(cost, params, learning_rate=learning_rate)

    patience_increase = 2
    improvement_threshold = 0.001

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 50000
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

                    [t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3] = \
                        [layer0, layer1, layer2_input, layer2, layer3]

            if patience <= iter:
                done_looping = True
                break

    error_line = error_line[0:epoch-1]

    scipy.io.savemat(name+'.mat', mdict={'Error_Spectrum': error_line})

    with open(name + '.pkl', 'wb') as f:
        pickle.dump([t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3], f)

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == "__main__":
    #single_layer_precepton(dataset='Gaussian_Data_Set.npy')
    main_ver1(nkerns=[20, 30])

    #initial_weight(nkerns=[12, 30])
    #main_ver1_3layers()
