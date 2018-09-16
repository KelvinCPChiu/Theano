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


class LogisticRegression(object):


    def __init__(self, input, n_in, n_out):

        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
     #   self.W = theano.shared(
     #       value=numpy.asarray(
     #           rng.uniform(
     #               low=-numpy.sqrt(6. / (n_in + n_out)),
     #               high=numpy.sqrt(6. / (n_in + n_out)),
     #               size=(n_in, n_out)), dtype=theano.config.floatX),
     #       name='W',
     #       borrow=True
     #   )

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

        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):

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

        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
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

    # For binary label Generation of GPD

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

    #For Weight Label Generation of GPD

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


def main_ver1(learning_rate=0.05, weight_decay=0.001, n_epochs=2000, nkerns=[20, 30],
          data_set='Gaussian_Data_Set.npy', batch_size=500):

    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay) + '_' + str(nkerns)

    if data_set == 'Gaussian_White_Noise.npy':
        name += '_WN'

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
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

    with open(name + '_Initial.pkl', 'wb') as f:
        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)

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


def main_ver1_3layers(learning_rate=0.01, weight_decay=0.001, n_epochs=1000, nkerns=[6],
          data_set='Gaussian_Data_Set.npy', batch_size=500):
    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
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

    layer1_input = layer0.output.flatten(2)

    layer1 = HiddenLayer(
        rng,
        input=layer1_input,
        n_in=nkerns[0] * 12 * 12,
        n_out=numpy.round(nkerns[0] * 12 * 12/2).astype(int),
        activation=T.nnet.relu
    )

    layer2 = LogisticRegression(input=layer1.output, n_in=numpy.round(nkerns[0] * 12 * 12/2).astype(int), n_out=10)

    cost = layer2.negative_log_likelihood(y)

    params = layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
            (param_i, param_i - learning_rate * (grad_i + weight_decay * param_i))
            for param_i, grad_i in zip(params, grads)]

    patience_increase = 2
    improvement_threshold = 0.0001

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
        layer2.errors(y),
        givens={
            layer0.input: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(
        [index],
        layer2.errors(y),
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

                    [t_layer0, t_layer1_input, t_layer1, t_layer2] = \
                        [layer0, layer1_input, layer1, layer2]

            if patience <= iter:
                done_looping = True
                break

    error_line = error_line[0:epoch-1]
    name = 'Gaussian_Model_'+str(learning_rate)+'_'+str(weight_decay)

    if data_set == 'Gaussian_White_Noise.npy':
        name += '_WN'

    #scipy.io.savemat(name+'.mat', mdict={'Error_Spectrum': error_line})

    #with open(name + '.pkl', 'wb') as f:
    #    pickle.dump([t_layer0, t_layer1_input, t_layer1, t_layer2], f)

    temp_time_2 = timeit.default_timer()
    print('%.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def main_ver1_fixed_seq(learning_rate=0.05, weight_decay=0.001, n_epochs=500, nkerns=[20, 30],
                        data_set='Gaussian_White_Noise.npy', batch_size=500):

    rng = numpy.random.RandomState(23455)
    # seed 1
    #rng = numpy.random.RandomState(10000)
    # seed 2
    #rng = numpy.random.RandomState(100)
    # seed 3
    datasets = numpy.load(data_set)
    if data_set == 'Gaussian_Data_Set.npy':
        train_set_x, train_set_y = Generate_Set_ez_fixed_seq(datasets, 50000, None, None)
    if data_set == 'Gaussian_White_Noise.npy':
        seq1 = numpy.load('Order_seq1.npy')
        seq2 = numpy.load('Order_seq2.npy')
        train_set_x, train_set_y = Generate_Set_ez_fixed_seq(datasets, 50000, seq1, seq2)

    valid_set_x, valid_set_y = Generate_Set_ez(datasets, 10000)

    test_set_x, test_set_y = Generate_Set_ez(datasets, 10000)

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

    # construct a fully-connected sigmoidal layer
    #layer2_input = T.concatenate([layer1.output.flatten(2), layer1a.output.flatten(2)], axis=1)

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=numpy.rint(nkerns[1] * 4 * 4/2),
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=numpy.rint(nkerns[1] * 4 * 4/2), n_out=10)

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
    patience = 1000000
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

                    with open('Gaussian_Model_WN_0.05_fix_seq.pkl', 'wb') as f:
                        pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)
                        #pickle.dump([layer0, layer1, layer1_input, layer2, layer3], f)

            if patience <= iter:
                done_looping = True
                break

    error_line = error_line[0:epoch-1]/100

    scipy.io.savemat('Gaussian_Model_WN_0.05_fix_seq.mat', mdict={'Error_Spectrum': error_line})

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

        layer3 = LogisticRegression(rng, input=layer2.output, n_in=numpy.round(nkerns[1] * 4 * 4 / 2).astype(int), n_out=10)

        name = 'Gaussian_Model_' + str(learning_rate) + '_' + str(weight_decay) + '_' + str(nkerns) + '_Initial.pkl'

        with open(name, 'wb') as f:
            pickle.dump([layer0, layer1, layer2_input, layer2, layer3], f)


def single_layer_precepton(learning_rate=0.05, weight_decay=0.001, n_epochs=2000,
            dataset='Gaussian_Data_Set.npy', batch_size=500):

    rng = numpy.random.RandomState(23455)

    datasets = numpy.load(dataset)

    train_set_x, train_set_y = Generate_Set_ez(datasets, 50000)

    valid_set_x, valid_set_y = Generate_Set_ez(datasets, 10000)
    #random_num = numpy.random.random_integers(0, 49999, 10000)
    #valid_set_x = theano.shared(numpy.asarray(train_set_x[random_num].eval(), dtype=theano.config.floatX), borrow=True)
    #valid_set_y = theano.shared(numpy.asarray(train_set_y[random_num].eval(), dtype=theano.config.floatX), borrow=True)

    test_set_x, test_set_y = Generate_Test_Set_ez(datasets, 20000)

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
    improvement_threshold = 0.0001

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

    error_line = error_line[0:epoch-1]

    scipy.io.savemat('Gaussian_Model_perceptron.mat', mdict={'Error_Spectrum': error_line})

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
    #main_ver1(nkerns=[20, 30], data_set='Gaussian_Data_Set_Range.npy')
    #initial_weight(nkerns=[12, 30])
    #main_ver1_3layers()
