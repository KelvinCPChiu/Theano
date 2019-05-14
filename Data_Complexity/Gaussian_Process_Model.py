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
from Adam import adam


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2
    # Negative log likelihood should be replaced by sigmoid for training, need to be checked again. For the correlation lenght cases.
    # For the New Gaussian Data, the cost should be investigated again.

    def sigmoid_cost_function(self, y):
        # This is only for fvector
        return T.mean(T.switch(T.eq(y, 1), -T.log(self.p_y_given_x), -T.log(1-self.p_y_given_x)))

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):

            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LogisticRegression_2(object):

    def __init__(self, input, n_in, n_out, rng):

        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.asarray(
                    rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )

    #    self.W = theano.shared(
    #        value=numpy.zeros(
    #            (n_in, n_out),
    #            dtype=theano.config.floatX
    #        ),
    #        name='W',
    #        borrow=True
    #    )

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
        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

         # self.y_pred = T.round(self.output)
        # T.dot(input, self.W) + self.b
        self.prep_y = T.argmax(self.output, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        # This is not really good as the relu may resulting output 0, and returning nan
        return -T.mean(y*T.log(self.output) + (1-y)*T.log(1-self.output))

    def sigmoid_cost_function(self, y):

        return T.mean(T.switch(T.eq(y, 1), -T.log(self.output), -T.log(1-self.output)))

    def mse_cost_function(self, y):
        return T.mean(T.square(y - self.output))

    def errors(self, y):

        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):

            return T.mean(T.square(y - self.output))
            #T.mean(T.neq(self.y_pred, y))
                #T.mean(T.switch(T.eq(y, 1), -T.log(self.output), -T.log(1-self.output)))
                #T.mean(T.square(y - self.output))
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
                    low=-numpy.sqrt(6 / (n_in + n_out)),
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
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

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
            ws=poolsize,
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
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

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


def shared_dataset(data_x, data_y, borrow=True):
    # 0-9 Label Representation
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def shared_dataset_2(data_x, data_y, borrow=True):
    # One Hot Representation of Label
    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    data_y = one_hot(data_y.astype(int), 10)

    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

    return shared_x, shared_y



def main_ver1_8080_softmax(learning_rate=0.05, weight_decay=0.001, n_epochs=300,
              nkerns=[20, 30], batch_size=500):
    # Need to reproduce softmax as Wrong Regression Cost
    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns)

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3

    train_set_x = numpy.load('Gaussian_Data_Set.npy')
    train_set_y = numpy.load('Gaussian_Label_Set.npy')

    valid_set_x = numpy.load('Gaussian_Valid_Data_Set.npy')
    valid_set_y = numpy.load('Gaussian_Valid_Label_Set.npy')

    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = valid_set_x, valid_set_y

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((n_test, 1, 80, 80))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 80, 80))
    train_set_x = train_set_x.reshape((n_train, 1, 80, 80))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 80, 80))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 80, 80),
        filter_shape=(nkerns[0], 1, 21, 21),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 30, 30),
        filter_shape=(nkerns[1], nkerns[0], 11, 11),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 10 * 10,
        n_out=numpy.round(nkerns[1] * 10 * 10/2).astype(int),
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=numpy.round(nkerns[1] * 10 * 10/2).astype(int), n_out=10)

    with open(name + '_Initial.pkl', 'wb') as f:
        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 10
    improvement_threshold = 0.001

    start_time = timeit.default_timer()

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

                    [t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3] = \
                        [layer0, layer1, layer2_input, layer2, layer3]

            if patience <= iter:
                done_looping = True
                break

    error_line = error_line[0:epoch-1]

    #if data_set == 'Gaussian_White_Noise.npy':
    #    name += '_WN'

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


def main_ver1_8080(learning_rate=0.05, weight_decay=0.001, n_epochs=300,
              nkerns=[20, 30], batch_size=500):

    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns)

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3

    train_set_x = numpy.load('Gaussian_Data_Set.npy')
    train_set_y = numpy.load('Gaussian_Label_Set.npy')

    valid_set_x = numpy.load('Gaussian_Valid_Data_Set.npy')
    valid_set_y = numpy.load('Gaussian_Valid_Label_Set.npy')

    train_set_x, train_set_y = shared_dataset_2(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset_2(valid_set_x, valid_set_y)
    test_set_x, test_set_y = valid_set_x, valid_set_y

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((n_test, 1, 80, 80))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 80, 80))
    train_set_x = train_set_x.reshape((n_train, 1, 80, 80))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    y = T.fmatrix('y')
    index = T.lscalar()

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 80, 80))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 80, 80),
        filter_shape=(nkerns[0], 1, 11, 11),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 35, 35),
        filter_shape=(nkerns[1], nkerns[0], 11, 11),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 12 * 12,
        n_out=numpy.round(nkerns[1] * 12 * 12/2).astype(int),
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression_2(input=layer2.output, n_in=numpy.round(nkerns[1] * 12 * 12/2).astype(int), n_out=10)

    with open(name + '_Initial.pkl', 'wb') as f:
        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 10
    improvement_threshold = 0.001

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 10000
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

    #if data_set == 'Gaussian_White_Noise.npy':
    #    name += '_WN'

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


def main_ver_4040_softmax(learning_rate=0.05, weight_decay=0.001, n_epochs=1000,
              nkerns=[20, 30], batch_size=500):
    # Need to reproduce softmax as Wrong Regression Cost ?
    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns) +'_Softmax'

    #if data_set == 'Gaussian_White_Noise.npy':
    #    name += '_WN'

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3

    train_set_x = numpy.load('Gaussian_Data_Set.npy')
    train_set_y = numpy.load('Gaussian_Label_Set.npy')

    valid_set_x = numpy.load('Gaussian_Valid_Data_Set.npy')
    valid_set_y = numpy.load('Gaussian_Valid_Label_Set.npy')

    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = valid_set_x, valid_set_y

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((n_test, 1, 40, 40))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 40, 40))
    train_set_x = train_set_x.reshape((n_train, 1, 40, 40))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 40, 40))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 40, 40),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 18, 18),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 7 * 7,
        n_out=numpy.round(nkerns[1] * 7 * 7/2).astype(int),
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=numpy.round(nkerns[1] * 7 * 7/2).astype(int), n_out=10)

    with open(name + '_Initial.pkl', 'wb') as f:
        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

    #cost = layer3.negative_log_likelihood(y)
    cost = layer3.sigmoid_cost_function(y)
    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 10
    improvement_threshold = 0.001

    start_time = timeit.default_timer()

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

                    [t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3] = \
                        [layer0, layer1, layer2_input, layer2, layer3]

            if patience <= iter:
                done_looping = True
                break

    #error_line = error_line[0:epoch-1]

    #scipy.io.savemat(name+'.mat', mdict={'Error_Spectrum': error_line})

    #with open(name + '.pkl', 'wb') as f:
    #   pickle.dump([t_layer0, t_layer1, t_layer2_input, t_layer2, t_layer3], f)

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def main_ver_4040(learning_rate=0.05, weight_decay=0.001, n_epochs=1000,
              nkerns=[20, 30], batch_size=500):

    #if data_set == 'Gaussian_White_Noise.npy':
    #    name += '_WN'

    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns) +'_2_20'

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3

    train_set_x = numpy.load('Gaussian_Data_Set_2_20.npy')
    train_set_y = numpy.load('Gaussian_Label_Set_2_20.npy')

    valid_set_x = numpy.load('Gaussian_Valid_Data_Set_2_20.npy')
    valid_set_y = numpy.load('Gaussian_Valid_Label_Set_2_20.npy')

    train_set_x, train_set_y = shared_dataset_2(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset_2(valid_set_x, valid_set_y)
    test_set_x, test_set_y = valid_set_x, valid_set_y

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((n_test, 1, 40, 40))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 40, 40))
    train_set_x = train_set_x.reshape((n_train, 1, 40, 40))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    y = T.fmatrix('y')
    index = T.lscalar()

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 40, 40))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 40, 40),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 18, 18),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 7 * 7,
        n_out=numpy.round(nkerns[1] * 7 * 7/2).astype(int),
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression_2(input=layer2.output, n_in=numpy.round(nkerns[1] * 7 * 7/2).astype(int), n_out=10, rng=rng)

    with open(name + '_Initial.pkl', 'wb') as f:
        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

    cost = layer3.sigmoid_cost_function(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]
    #updates = adam(cost, params)

    patience_increase = 4
    improvement_threshold = 0.00001

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 5000000
    validation_frequency = min(n_train_batches, patience // 2)
    epoch = 0
    done_looping = False
    error_line = numpy.zeros(n_epochs+1)

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

    validation_losses = [validate_model(i) for i
                         in range(n_valid_batches)]
    this_validation_loss = numpy.mean(validation_losses)
    print('Initial validation error %f' % this_validation_loss)
    error_line[0] = this_validation_loss

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
                error_line[epoch] = this_validation_loss

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

    error_line = error_line[0:epoch]

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


def main_ver_4040_converge_check_pending(learning_rate=0.05, weight_decay=0.001, n_epochs=1000,
              nkerns=[20, 30], batch_size=500):

    #if data_set == 'Gaussian_White_Noise.npy':
    #    name += '_WN'

    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns) +'_Softmax'

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3

    train_set_x = numpy.load('Gaussian_Data_Set.npy')
    train_set_y = numpy.load('Gaussian_Label_Set.npy')

    valid_set_x = numpy.load('Gaussian_Valid_Data_Set.npy')
    valid_set_y = numpy.load('Gaussian_Valid_Label_Set.npy')

    train_set_x, train_set_y = shared_dataset_2(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset_2(valid_set_x, valid_set_y)
    test_set_x, test_set_y = valid_set_x, valid_set_y

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((n_test, 1, 40, 40))
    valid_set_x = valid_set_x.reshape((n_valid, 1, 40, 40))
    train_set_x = train_set_x.reshape((n_train, 1, 40, 40))

    n_train_batches = n_train//batch_size
    n_valid_batches = n_valid//batch_size
    n_test_batches = n_test//batch_size

    x = T.matrix('x')
    y = T.fmatrix('y')
    index = T.lscalar()

    print('... loading the model')

    layer0_input = x.reshape((batch_size, 1, 40, 40))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 40, 40),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 18, 18),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 7 * 7,
        n_out=numpy.round(nkerns[1] * 7 * 7/2).astype(int),
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression_2(input=layer2.output, n_in=numpy.round(nkerns[1] * 7 * 7/2).astype(int), n_out=10)

    with open(name + '_Initial.pkl', 'wb') as f:
        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

    cost = layer3.sigmoid_cost_function(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 10
    improvement_threshold = 0.00001

    start_time = timeit.default_timer()

    print('... training')

    temp_time_1 = timeit.default_timer()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 500000
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


def initial_weight(learning_rate=0.05, weight_decay=0.001, nkerns=[20, 30], batch_size=500):

        rng = numpy.random.RandomState(23455)
        # seed 1
        #rng = numpy.random.RandomState(10000)
        #seed 2
        #rng = numpy.random.RandomState(100)
        # seed 3

        x = T.matrix('x')

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
            n_out=numpy.round(nkerns[1] * 4 * 4 / 2).astype(int),
            activation=T.nnet.relu
        )

        layer3 = LogisticRegression(input=layer2.output, n_in=numpy.round(nkerns[1] * 4 * 4 / 2).astype(int), n_out=10)

        name = 'Gaussian_Model_' + str(learning_rate) + '_' + str(weight_decay) + '_' + str(nkerns) + '_Initial.pkl'

        with open(name, 'wb') as f:
            pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)


if __name__ == "__main__":
    #single_layer_precepton()
    #main_ver1()
    #initial_weight(nkerns=[10, 20])
    #main_ver1_3layers()
    main_ver_4040(nkerns=[20, 30])
