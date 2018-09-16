from __future__ import division
import numpy
import theano.tensor as T
import theano
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import six.moves.cPickle as pickle
import timeit
import scipy.io


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.output = T.nnet.relu(T.dot(input, self.W) + self.b)
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

         # self.y_pred = T.round(self.output)
        # T.dot(input, self.W) + self.b
        # T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):

        return T.mean(T.square(y - self.output))
               #- 0.01 * T.mean(y * T.log(self.output))
               #-T.mean(y * T.log(y / self.output))
        # end-snippet-2

    def errors(self, y):

        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):

            return T.mean(T.square(y - self.output))
            #1 - T.mean(T.all(T.isclose(y, self.output, rtol=0, atol=0.02), axis=1))
            #T.mean(T.sqr(y - self.output))
            #1 - T.mean(T.all(T.isclose(y, self.output, rtol=0, atol=0.5), axis=1))
                #1 - T.mean(T.all(T.isclose(y, self.output, rtol=0, atol=0.2), axis=1))
                # T.abs_(T.mean(T.invert(T.all(T.isclose(self.output, y, rtol=0.005, atol=0.3), axis=1))))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):

        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):


        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class ConvPoolLayer_NoMaxPool(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # Filter_shape[1] is the input kernel number
        # Filter_shape[0] is the output kernel number
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def Gaussian_Data():

    def kernel(x_1, x_2):
        sqdist = numpy.sum(x_1 ** 2, 1).reshape(-1, 1) + numpy.sum(x_2 ** 2, 1) - 2 * numpy.dot(x_1, x_2.T)
        return numpy.exp(-.5 * sqdist)

    n = 28
    training_set_sample = 10
    xtest = numpy.linspace(-5, 5, n).reshape(-1, 1)
    gaussian_kernel = kernel(xtest, xtest)

    cholesky = numpy.linalg.cholesky(gaussian_kernel+ 10 ** (-6) * numpy.eye(n))

    norm = numpy.random.normal(size=(training_set_sample, n, n))

    f_prior = numpy.tensordot(cholesky, numpy.tensordot(cholesky, norm, axes=(1, 1)), axes=(1, 2))

    f_prior = f_prior / numpy.amax(numpy.amax(f_prior, axis=0), axis=0)

    f_temp = numpy.zeros([10, 28, 28])
    for i in range(0, 10, 1):
        f_temp[i, :, :] = f_prior[:, :, i]

    numpy.save('Gaussian_Data_Set.npy', f_temp)

    return f_temp


def Generate_Set(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def class_image(input_image, set_size):

        #input_image = numpy.zeros((set_size, input_image.shape[1], input_image.shape[2]))

        morphing1 = numpy.random.random_integers(0, 9, set_size)
        morphing2 = numpy.random.random_integers(0, 9, set_size)
        morphing_coeff = numpy.random.random(set_size)
        resultant_set = morphing_coeff[:, None, None] * input_image[morphing1]+(1-morphing_coeff)[:, None, None]*input_image[morphing2]
        return resultant_set, morphing1, morphing2, morphing_coeff

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    generated_image_set, set_order1, set_order2, morphing_constant = class_image(raw_image_set, size_desired)
    number_of_classes = 10
    set_order1 = one_hot(set_order1, number_of_classes)
    set_order2 = one_hot(set_order2, number_of_classes)
    generated_label_set = set_order1*morphing_constant[:, None] + set_order2*((1-morphing_constant)[:, None])

    return shared_dataset(generated_image_set, generated_label_set)


def Generate_Test_Set(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def class_image(input_image, set_size):

        #input_image = numpy.zeros((set_size, input_image.shape[1], input_image.shape[2]))

        morphing1 = numpy.random.random_integers(0, 9, set_size)
        morphing2 = numpy.random.random_integers(0, 9, set_size)
        morphing3 = numpy.random.random_integers(0, 9, set_size)

        morphing_coeff = numpy.random.random(set_size)
        morphing_coeff2 = numpy.random.random(set_size)
        morphing_coeff3 = numpy.random.random(set_size)

        resultant_set = morphing_coeff[:, None, None] * input_image[morphing1] + morphing_coeff2[:, None, None] * input_image[morphing2] + morphing_coeff3[:, None, None] * input_image[morphing3]

        resultant_set = resultant_set/((morphing_coeff3+morphing_coeff2 + morphing_coeff)[:, None, None])
        return [resultant_set,
                [morphing1, morphing2, morphing3],
                [morphing_coeff, morphing_coeff2, morphing_coeff3]]

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

        return shared_x, shared_y

    generated_image_set, set_order, constant = class_image(raw_image_set, size_desired)
    number_of_classes = 10
    set_order1 = one_hot(set_order[0], number_of_classes)
    set_order2 = one_hot(set_order[1], number_of_classes)
    set_order3 = one_hot(set_order[2], number_of_classes)
    constant_sum = constant[0] + constant[1] + constant[2]
    generated_label_set = (set_order1*constant[0][:, None] + set_order2 * constant[1][:, None] + set_order3 * constant[2][:, None])/constant_sum[:, None]

    return shared_dataset(generated_image_set, generated_label_set)


def Generate_Set_modified(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def class_image(input_image, set_size):

        # input_image = numpy.zeros((set_size, input_image.shape[1], input_image.shape[2]))

        morphing1 = numpy.random.random_integers(0, 9, set_size)
        morphing2 = numpy.random.random_integers(0, 9, set_size)
        digit = numpy.random.random_integers(8, 9, set_size)
        morphing_coeff = (numpy.random.random(set_size) + digit)/10
        resultant_set = morphing_coeff[:, None, None] * input_image[morphing1]+(1-morphing_coeff)[:, None, None]*input_image[morphing2]
        return resultant_set, morphing1, morphing_coeff

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    generated_image_set, set_order1, morphing_constant = class_image(raw_image_set, size_desired)
    #number_of_classes = 10
    #set_order1 = one_hot(set_order1, number_of_classes)

    return shared_dataset(generated_image_set, set_order1)


def Generate_Test_Set_modified(raw_image_set, size_desired):

    #  The range is restricted in this case

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def class_image(input_image, set_size):

        #input_image = numpy.zeros((set_size, input_image.shape[1], input_image.shape[2]))

        morphing1 = numpy.random.random_integers(0, 9, set_size)
        morphing2 = numpy.random.random_integers(0, 9, set_size)
        digit = numpy.random.random_integers(5, 7, set_size)
        morphing_coeff = (numpy.random.random(set_size) + digit)/10
        resultant_set = morphing_coeff[:, None, None] * input_image[morphing1]+(1-morphing_coeff)[:, None, None]*input_image[morphing2]
        return resultant_set, morphing1, morphing_coeff

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    generated_image_set, set_order1, morphing_constant = class_image(raw_image_set, size_desired)
    #number_of_classes = 10
    #set_order1 = one_hot(set_order1, number_of_classes)

    return shared_dataset(generated_image_set, set_order1)


def main_ver1(learning_rate=0.01, weight_decay=0.001, n_epochs=2000, nkerns=[20, 30],
          dataset='Gaussian_Data_Set.npy', batch_size=500):

    name = 'Gauss'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns)

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3
    datasets = numpy.load(dataset)

    train_set_x, train_set_y = Generate_Set(datasets, 50000)

    valid_set_x, valid_set_y = Generate_Set(datasets, 10000)
    #random_num = numpy.random.random_integers(0, 49999, 10000)
    #valid_set_x = theano.shared(numpy.asarray(train_set_x[random_num].eval(), dtype=theano.config.floatX), borrow=True)
    #valid_set_y = theano.shared(numpy.asarray(train_set_y[random_num].eval(), dtype=theano.config.floatX), borrow=True)

    test_set_x, test_set_y = Generate_Test_Set(datasets, 20000)

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

    layer0 = ConvPoolLayer_NoMaxPool(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 9, 9)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 20, 20),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # construct a fully-connected sigmoidal layer
    #layer2_input = T.concatenate([layer1.output.flatten(2), layer1a.output.flatten(2)], axis=1)

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 9 * 9,
        n_out=1000,
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=1000, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.01

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 100000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False
    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: test_set_x[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            layer0.input: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

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


def main_ver2(learning_rate=0.01, weight_decay=0.001, n_epochs=2000, nkerns=[5],
         dataset='Gaussian_Data_Set.npy', batch_size=500):

    rng = numpy.random.RandomState(23455)

    datasets = numpy.load(dataset)

    train_set_x, train_set_y = Generate_Set(datasets, 50000)

    valid_set_x, valid_set_y = Generate_Set(datasets, 10000)
    #random_num = numpy.random.random_integers(0, 49999, 10000)
    #valid_set_x = theano.shared(numpy.asarray(train_set_x[random_num].eval(), dtype=theano.config.floatX), borrow=True)
    #valid_set_y = theano.shared(numpy.asarray(train_set_y[random_num].eval(), dtype=theano.config.floatX), borrow=True)

    test_set_x, test_set_y = Generate_Test_Set(datasets, 20000)

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    #print(str(n_train), str(n_valid),str(n_test))
    test_set_x = test_set_x.reshape((n_test, 1, 28, 28))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    train_set_x = train_set_x.reshape((n_train, 1, 28, 28))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    # Need to check how to update the x such that no need to input in such a way
    y = T.fmatrix('y')
    index = T.lscalar()

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 28, 28))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(4, 4)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    #layer2_input = T.concatenate([layer1.output.flatten(2), layer1a.output.flatten(2)], axis=1)
    layer1_input = layer0.output.flatten(2)

    # classify the values of the fully-connected sigmoidal layer
    layer1 = LogisticRegression(input=layer1_input, n_in=180, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer1.negative_log_likelihood(y)
    # + 0.3*layer3.L2_lost(y)

    params = layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.995

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 1000000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False
    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer1.errors(y),
        givens={
            layer0.input: test_set_x[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer1.errors(y),
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            layer0.input: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    while (epoch < n_epochs) and (not done_looping):
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

                print('epoch %i, minibatch %i/%i, validation error %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))
                error_line[epoch-1] = this_validation_loss
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
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))

                    #with open('Upated_Gaussian_Mode.pkl', 'wb') as f:
                    #    pickle.dump([layer0, layer1], f)

            if patience <= iter:
                done_looping = True
                break

    error_line = error_line[0:epoch-1]/100

    #scipy.io.savemat('Gaussian_Model_min_max_pool_2.mat', mdict={'Error_Spectrum': error_line})

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def main_5_layers(learning_rate=0.01, weight_decay=0.001, n_epochs=200, nkerns=[20, 30, 40],
         dataset='Gaussian_Data_Set.npy', batch_size=500):

    rng = numpy.random.RandomState(23455)

    datasets = numpy.load(dataset)

    train_set_x, train_set_y = Generate_Set(datasets, 50000)

    valid_set_x, valid_set_y = Generate_Set(datasets, 10000)
    #random_num = numpy.random.random_integers(0, 49999, 10000)
    #valid_set_x = theano.shared(numpy.asarray(train_set_x[random_num].eval(), dtype=theano.config.floatX), borrow=True)
    #valid_set_y = theano.shared(numpy.asarray(train_set_y[random_num].eval(), dtype=theano.config.floatX), borrow=True)

    test_set_x, test_set_y = Generate_Test_Set(datasets, 20000)

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    #print(str(n_train), str(n_valid),str(n_test))
    test_set_x = test_set_x.reshape((n_test, 1, 28, 28))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    train_set_x = train_set_x.reshape((n_train, 1, 28, 28))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    # Need to check how to update the x such that no need to input in such a way
    y = T.fmatrix('y')
    index = T.lscalar()

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 28, 28))

    #print(layer0_input.type)
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = ConvPoolLayer_NoMaxPool(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 3, 3),
    )
    #print(layer0.input.type)
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 26, 26),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 12, 12),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2)
    )
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[2] * 5 * 5,
        n_out=500,
        activation=T.nnet.relu
    )
    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.995

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 100000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False

    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            layer0.input: test_set_x[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            layer0.input: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            layer0.input: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    while (epoch < n_epochs) and (not done_looping):
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

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

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

                    #with open('Con_MLP_Train_Trans_random_After.pkl', 'wb') as f:
                    #    pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

            if patience <= iter:
                done_looping = True
                break

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def precepton_learning(learning_rate=0.01, weight_decay=0.001, n_epochs=500,
         dataset='Gaussian_Data_Set.npy', batch_size=500):

    rng = numpy.random.RandomState(23455)

    datasets = numpy.load(dataset)

    train_set_x, train_set_y = Generate_Set(datasets, 50000)

    valid_set_x, valid_set_y = Generate_Set(datasets, 10000)
    #random_num = numpy.random.random_integers(0, 49999, 10000)
    #valid_set_x = theano.shared(numpy.asarray(train_set_x[random_num].eval(), dtype=theano.config.floatX), borrow=True)
    #valid_set_y = theano.shared(numpy.asarray(train_set_y[random_num].eval(), dtype=theano.config.floatX), borrow=True)

    test_set_x, test_set_y = Generate_Test_Set(datasets, 20000)

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    #print(str(n_train), str(n_valid),str(n_test))
    test_set_x = test_set_x.reshape((n_test, 784))
    valid_set_x = valid_set_x.reshape((n_valid, 784))
    train_set_x = train_set_x.reshape((n_train, 784))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    # Need to check how to update the x such that no need to input in such a way
    y = T.fmatrix('y')
    index = T.lscalar()

    print('... loading the model')
    layer0_input = x

    layer0 = HiddenLayer(
        rng,
        input=layer0_input,
        n_in=28*28,
        n_out=500,
        activation=T.nnet.relu
    )

    layer2 = HiddenLayer(
        rng,
        input=layer0.output,
        n_in=500,
        n_out=200,
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=200, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.01

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 1000000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False

    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    while (epoch < n_epochs) and (not done_looping):
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

                print('epoch %i, minibatch %i/%i, validation error %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))
                error_line[epoch-1] = this_validation_loss
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
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))

                    #with open('Gaussian_Model_perceptron_white_noise.pkl', 'wb') as f:
                    #    pickle.dump([layer0, layer2, layer3], f)

            if patience <= iter:
                done_looping = True
                break

    error_line = error_line[0:epoch-1]/100

    #scipy.io.savemat('Gaussian_Model_perceptron_white_noise.mat', mdict={'Error_Spectrum': error_line})

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def single_layer_precepton(learning_rate=0.01, weight_decay=0.001, n_epochs=500,
         dataset='Gaussian_Data_Set.npy', batch_size=500):

    rng = numpy.random.RandomState(23455)

    datasets = numpy.load(dataset)

    train_set_x, train_set_y = Generate_Set(datasets, 50000)

    valid_set_x, valid_set_y = Generate_Set(datasets, 10000)
    #random_num = numpy.random.random_integers(0, 49999, 10000)
    #valid_set_x = theano.shared(numpy.asarray(train_set_x[random_num].eval(), dtype=theano.config.floatX), borrow=True)
    #valid_set_y = theano.shared(numpy.asarray(train_set_y[random_num].eval(), dtype=theano.config.floatX), borrow=True)

    test_set_x, test_set_y = Generate_Test_Set(datasets, 20000)

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    #print(str(n_train), str(n_valid),str(n_test))
    test_set_x = test_set_x.reshape((n_test, 784))
    valid_set_x = valid_set_x.reshape((n_valid, 784))
    train_set_x = train_set_x.reshape((n_train, 784))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    # Need to check how to update the x such that no need to input in such a way
    y = T.fmatrix('y')
    index = T.lscalar()

    print('... loading the model')
    layer0_input = x

    layer3 = LogisticRegression(input=x, n_in=784, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.01

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 100000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False

    error_line = numpy.zeros(n_epochs)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * 500: (index + 1) * 500],
            y: train_set_y[index * 500: (index + 1) * 500]})

    while (epoch < n_epochs) and (not done_looping):
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

                print('epoch %i, minibatch %i/%i, validation error %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))
                error_line[epoch-1] = this_validation_loss
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
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))

                    #with open('Gaussian_Model_perceptron_white_noise.pkl', 'wb') as f:
                    #    pickle.dump([layer0, layer2, layer3], f)

            if patience <= iter:
                done_looping = True
                break

    error_line = error_line[0:epoch-1]/100

    #scipy.io.savemat('Gaussian_Model_perceptron_white_noise.mat', mdict={'Error_Spectrum': error_line})

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def initial_weight(nkerns=[20, 30], batch_size=500):

        rng = numpy.random.RandomState(23455)
        # seed 1
        #rng = numpy.random.RandomState(10000)
        #seed 2
        #rng = numpy.random.RandomState(100)
        # seed 3

        x = T.matrix('x')

        print('... loading the model')

        layer0_input = x.reshape((batch_size, 1, 28, 28))

        layer0 = ConvPoolLayer_NoMaxPool(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 9, 9)
        )

        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 20, 20),
            filter_shape=(nkerns[1], nkerns[0], 3, 3),
            poolsize=(2, 2)
        )

        layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 9 * 9,
            n_out=1000,
            activation=T.nnet.relu
        )

        layer3 = LogisticRegression(input=layer2.output, n_in=1000, n_out=10)

        with open('Updated_Gaussian_Model_Test_3_initial_stage_seed2.pkl', 'wb') as f:
            pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)


if __name__ == "__main__":
    main_ver2()
    #single_layer_precepton()
    #main_ver1()
    #precepton_learning()
    #initial_weight()
