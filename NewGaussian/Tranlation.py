import os
import sys
import struct
import matplotlib.pyplot as plt
import numpy
import gzip
import theano.tensor as T
import theano
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import six.moves.cPickle as pickle
import xlsxwriter
import timeit
import scipy.io


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
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
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
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


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


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
                    ds=poolsize,
                    ignore_border=True
                )

                # add the bias term. Since the bias is a vector (1D array), we first
                # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
                # thus be broadcasted across mini-batches and feature map
                # width & height
                self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

                # store parameters of this layer
                self.params = [self.W, self.b]

                # keep track of model input
                self.input = input


def shared_dataset(data_x, data_y, borrow=True):
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def theano_translation_old(image_tensor_input, displacement, horizontal, vertical, borrow=True):
    tx = image_tensor_input
    temp1 = numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX)
    txout = theano.shared(temp1, borrow=borrow)
    txout = T.Rebroadcast((1, True))(txout)
    if vertical == 0:
        if displacement >= 0:
            txout = T.set_subtensor(txout[:, :, :, displacement:27], tx[:, :, :, 0:27-displacement])
        else:
            txout = T.set_subtensor(txout[:, :, :, 0:27+displacement], tx[:, :, :, -displacement:27])

    if horizontal == 0:
        if displacement >= 0:
            txout = T.set_subtensor(txout[:, :, displacement:27, :], tx[:, :, 0:27-displacement, :])
        else:
            txout = T.set_subtensor(txout[:, :, 0:27 - displacement, :], tx[:, :, displacement:27, :])

    return txout


def theano_translation(image_tensor_input, horizon_disp, verti_disp, borrow=True):
    tx = image_tensor_input

    def vertical_shift(image_input, displacement, borrow=True):
        temp1 = numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX)
        txout1 = theano.shared(temp1, borrow=True)
        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, 0:27 - displacement, :], image_input[:, :, displacement:27, :])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, -displacement:27, :], image_input[:, :,  0:27 + displacement, :])
        else:
            txout1 = image_input
        return txout1

    def horizontal_shift(image_input, displacement, borrow=True):
        temp1 = numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX)
        txout1 = theano.shared(temp1, borrow=True)
        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, :, displacement:27], image_input[:, :, :, 0:27-displacement])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, :, 0:27+displacement], image_input[:, :, :, -displacement:27])
        else:
            txout1 = image_input
        return txout1

    if verti_disp != 0 and horizon_disp == 0:
        txout = vertical_shift(tx, verti_disp, borrow=True)

    if horizon_disp != 0 and verti_disp == 0:
        txout = horizontal_shift(tx, horizon_disp, borrow=True)

    if horizon_disp != 0 and verti_disp != 0:
        txout = vertical_shift(tx, verti_disp, borrow=True)
        txout = horizontal_shift(txout, horizon_disp,  borrow=True)

    if verti_disp == 0 and horizon_disp == 0:
        txout = tx

    txout = T.Rebroadcast((1, True))(txout)

    return txout

    #if TransNum >= 0:
    #    tempim[:, TransNum:27] = im[:, 0:27 - TransNum]  # Left
    #else:
    #   tempim[:, 0:27 - TransNum] = im[:, TransNum:27]  # Right


def theano_rotation(image_tensor_input, angle, borrow=True):
    tx = image_tensor_input
    txout = theano.shared(numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX), borrow=borrow)

    if (angle % 90) == 0:

        if angle == 180 or angle == -180:
            txout = tx[:, :, ::-1, ::-1]

        if angle == 90 or angle == -270:
            txout = tx.dimshuffle(0, 1, 3, 2)
            txout = txout[:, :, :, ::-1]

        if angle == 270 or angle == -90:
            txout = tx.dimshuffle(0, 1, 3, 2)
            txout = txout[:, :, ::-1, :]

        if angle == 0:
            txout = tx

    if (angle % 90) != 0:
        if angle > 90 or angle < -90:
            tx = tx[:, :, ::-1, ::-1]
            angle = angle - numpy.sign(angle)*180
            #txout = image_tensor_input
        angle = numpy.radians(angle)
        temp_position_ori = numpy.zeros((2, 28*28))
        for x in range(-14, 14):
            for y in range(-14, 14):
                temp_position_ori[0, 28*(x+14)+y+14], temp_position_ori[1, 28*(x+14)+y+14] = x, y
        # print(temp_position_ori)
        #print(temp_position_ori)
        #rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
        #temp_position = numpy.floor(numpy.dot(rotation, temp_position_ori))
        rotation1 = numpy.array([[1, -numpy.tan(angle/2)], [0, 1]])
        rotation2 = numpy.array([[1, 0], [numpy.sin(angle), 1]])
        rotation3 = numpy.array([[1, -numpy.tan(angle/2)], [0, 1]])
        temp_position = numpy.floor(numpy.dot(rotation1, temp_position_ori))
        temp_position = numpy.floor(numpy.dot(rotation2, temp_position))
        temp_position = numpy.floor(numpy.dot(rotation3, temp_position))

        logic_temp_pos_p = temp_position < 14
        logic_temp_pos_n = temp_position >= -14
        #print(logic_temp_pos_p[0]*logic_temp_pos_p[1])
        temp_position = (logic_temp_pos_p[0]*logic_temp_pos_p[1]) * \
                        (logic_temp_pos_n[0]*logic_temp_pos_n[1]) * temp_position

        temp_position = temp_position.astype(int)
        temp_position_ori = temp_position_ori.astype(int)

        txout = T.set_subtensor(txout[:, :, temp_position[1, :]+14, temp_position[0, :]+14],
                                tx[:, :, temp_position_ori[1, :]+14, temp_position_ori[0, :]+14])

    txout = T.Rebroadcast((1, True))(txout)
    return txout


def theano_rotation_X(image_tensor_input, angle, borrow=True):
    tx = image_tensor_input
    txout = theano.shared(numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX), borrow=borrow)

    if (angle % 90) == 0:

        if angle == 180 or angle == -180:
            txout = tx[:, :, ::-1, ::-1]

        if angle == 90 or angle == -270:
            txout = tx.dimshuffle(0, 1, 3, 2)
            txout = txout[:, :, :, ::-1]

        if angle == 270 or angle == -90:
            txout = tx.dimshuffle(0, 1, 3, 2)
            txout = txout[:, :, ::-1, :]

        if angle == 0:
            txout = tx

    if (angle % 90) != 0:
        if angle > 90 or angle < -90:
            tx = tx[:, :, ::-1, ::-1]
            angle = angle - numpy.sign(angle)*180
        txout_c = theano.shared(numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX), borrow=borrow)
        angle = numpy.radians(angle)
        temp_position_ori = numpy.zeros((2, 28*28))
        for x in range(-14, 14):
            for y in range(-14, 14):
                temp_position_ori[0, 28*(x+14)+y+14], temp_position_ori[1, 28*(x+14)+y+14] = x, y
        # print(temp_position_ori)
        #print(temp_position_ori)
        #rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
        #temp_position = numpy.floor(numpy.dot(rotation, temp_position_ori))
        rotation1 = numpy.array([[1, -numpy.tan(angle/2)], [0, 1]])
        rotation2 = numpy.array([[1, 0], [numpy.sin(angle), 1]])
        rotation3 = numpy.array([[1, -numpy.tan(angle/2)], [0, 1]])
        temp_position = numpy.floor(numpy.dot(rotation1, temp_position_ori))
        temp_position = numpy.floor(numpy.dot(rotation2, temp_position))
        temp_position = numpy.floor(numpy.dot(rotation3, temp_position))

        temp_position_c = numpy.round(numpy.dot(rotation1, temp_position_ori))
        temp_position_c = numpy.round(numpy.dot(rotation2, temp_position_c))
        temp_position_c = numpy.round(numpy.dot(rotation3, temp_position_c))

        logic_temp_pos_p = temp_position < 14
        logic_temp_pos_n = temp_position >= -14
        logic_temp_pos_pc = temp_position_c < 14
        logic_temp_pos_nc = temp_position_c >= -14
        #print(logic_temp_pos_p[0]*logic_temp_pos_p[1])
        temp_position = (logic_temp_pos_p[0]*logic_temp_pos_p[1]) * \
                        (logic_temp_pos_n[0]*logic_temp_pos_n[1]) * temp_position

        temp_position_c = (logic_temp_pos_pc[0]*logic_temp_pos_pc[1]) * \
                        (logic_temp_pos_nc[0]*logic_temp_pos_nc[1]) * temp_position_c

        temp_position_c = temp_position_c.astype(int)
        temp_position = temp_position.astype(int)
        temp_position_ori = temp_position_ori.astype(int)

        txout = T.set_subtensor(txout[:, :, temp_position[1, :]+14, temp_position[0, :]+14],
                                tx[:, :, temp_position_ori[1, :]+14, temp_position_ori[0, :]+14])

        txout_c = T.set_subtensor(txout_c[:, :, temp_position_c[1, :]+14, temp_position_c[0, :]+14],
                                tx[:, :, temp_position_ori[1, :]+14, temp_position_ori[0, :]+14])

        txout = ((txout <= txout_c)*txout_c + (txout_c <= txout)*txout - (txout_c > txout)*txout)

    txout = T.Rebroadcast((1, True))(txout)

    return txout


def loaddata_mnist(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def loaddata_origin(index, num_image, image_set, label_set, borrow=True):
    fin = image_set
    finL = label_set
    #fin = 'train-images.idx3-ubyte'
    #finL = 'train-labels.idx1-ubyte'
    buf = open(fin, 'rb').read()
    buf2 = open(finL, 'rb').read()
    #magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    #magic2, num2 = struct.unpack_from('>II', buf2, lindex)
    #index += struct.calcsize('>IIII')
    #lindex += struct.calcsize('>B')
    #image = struct.unpack_from('>47040000IIII', buf, index)
    #label = struct.unpack_from('>60000B', buf2, lindex)
    image = struct.unpack_from('>'+str(num_image)+'IIII', buf, struct.calcsize('>IIII')+index*struct.calcsize('>IIII'))
    label = struct.unpack_from('>'+str(num_image)+'B', buf2, struct.calcsize('>II')+index*struct.calcsize('>B'))
    image_out, label_out = shared_dataset(image,label)
    return image_out, label_out


def printimage(test_set_x):
        # Print Image from tensor to numpy and plot it
        mm = numpy.squeeze(test_set_x.eval(), axis=(0,))
        #print(mm)
        fig = plt.figure()
        plotwindow = fig.add_subplot(111)
        plt.imshow(mm)#, cmap='gray')
        plt.axis('off')
        fig.savefig('figure1.png', bbox_inches='tight', pad_inches=0)
        plt.show()
        return
        # print(test_set_x.shape[0])
        # print(len(test_set_y.eval()))
        # print(test_set_x.shape[1])
        # for x_trans in range (-7,8):
        #     for y_trans in range (-7,8):
        #         trans_test_set_x = theano_translation(test_set_x, ,x_move)
        # layer0_input = x.reshape((batch_size, 1, 28, 28))


def theano_translation_update(image_tensor_input, image_tensor_output, horizon_disp, verti_disp, set_size, borrows):
    tx = image_tensor_input

    def vertical_shift(image_input, txout1, displacement):

        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, 0:27 - displacement, :], image_input[:, :, displacement:27, :])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, -displacement:27, :], image_input[:, :,  0:27 + displacement, :])
        else:
            txout1 = image_input
        return txout1

    def horizontal_shift(image_input, txout1, displacement):

        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, :, displacement:27], image_input[:, :, :, 0:27-displacement])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, :, 0:27+displacement], image_input[:, :, :, -displacement:27])
        else:
            txout1 = image_input
        return txout1

    if verti_disp != 0 and horizon_disp == 0:
        image_tensor_output = vertical_shift(tx, image_tensor_output, verti_disp)

    if horizon_disp != 0 and verti_disp == 0:
        image_tensor_output = horizontal_shift(tx, image_tensor_output, horizon_disp)

    if horizon_disp != 0 and verti_disp != 0:
        image_tensor_output_temp = vertical_shift(tx, image_tensor_output, verti_disp)
        image_tensor_output = horizontal_shift(image_tensor_output_temp, image_tensor_output, horizon_disp)

    if verti_disp == 0 and horizon_disp == 0:
        image_tensor_output = tx

    image_tensor_output = T.Rebroadcast((1, True))(image_tensor_output)

    return image_tensor_output


def rotation_error_spectrum(model_file):

    y = T.ivector('y')
    index = T.lscalar()
    dataset = 'mnist.pkl.gz'
    datasets = loaddata_mnist(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.reshape((10000, 1, 28, 28))

    with open(model_file, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    error_spectrum = numpy.zeros(37)

    print('Start Predicting...')
    start_time = timeit.default_timer()
    for rotation_angle in range(-180, 190, 10):
        #rotation_angle = rotation_angle/180 * numpy.pi
        print('Data Set Rotation Angle:'+str(rotation_angle))
        temp_time_1 = timeit.default_timer()
        predicted_values = 0
        t_test_set_x = theano_rotation(test_set_x, rotation_angle)
        #printimage(t_test_set_x[0])
        predict_model = theano.function(inputs=[index],
                                        outputs=layer3.errors(y),
                                        givens={layer0.input: t_test_set_x[index * 500: (index + 1) * 500],
                                                y: test_set_y[index * 500: (index + 1) * 500]})

        for batch_value in range(0, 20, 1):
            temp_predicted_values = predict_model(batch_value)
            predicted_values = temp_predicted_values + predicted_values
        predicted_values = predicted_values / 20

        error_spectrum[rotation_angle/10+18] = predicted_values
        print('Error :'+str(predicted_values))

        temp_time_2 = timeit.default_timer()
        print('This loop ran for %.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    scipy.io.savemat(model_file + 'rotation_error_spectrum.mat', mdict={'Error_Spectrum': error_spectrum})

    return error_spectrum


def translation_prediction(model_file):

    y = T.ivector('y')
    index = T.lscalar()
    dataset = 'mnist.pkl.gz'
    datasets = loaddata_mnist(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.reshape((10000, 1, 28, 28))

    printimage(test_set_x[15])
    #numpy.save('test_set_label.npy', test_set_y.eval())
    #print('end')
    #return
    #test_set_x = T.Rebroadcast((1, True))(test_set_x)
    #test_set_x = T.Rebroadcast((1, True))(test_set_x)

#    with open('Con_MLP_Train_Trans_0_4.pkl', 'rb') as f:

    with open(model_file, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    error_spectrum = numpy.zeros((21, 21))

    #testy = test_set_y.eval()
    #numpy.save('Test_Set_Label.npy', testy)
    #t_test_set_x = theano_translation(test_set_x, -20, -20, borrow=True)
    #printimage(t_test_set_x[0])

    print('Start Predicting...')
    start_time = timeit.default_timer()
    for horizontal in range(-20, 21, 2):
        temp_time_1 = timeit.default_timer()
        for vertical in range(-20, 21, 2):
            predicted_values = 0
            t_test_set_x = theano_translation(test_set_x, horizontal, vertical, borrow=True)
            predict_model = theano.function(inputs=[index],
                                            outputs=layer3.errors(y),
                                            givens={layer0.input: t_test_set_x[index * 500: (index + 1) * 500],
                                                    y: test_set_y[index * 500: (index + 1) * 500]})
            #print('Horizontal Shift:' + str(horizontal) + '; Vertical Shift:' + str(vertical))
            for batch_value in range(0, 20, 1):
                temp_predicted_values = predict_model(batch_value)
                predicted_values = temp_predicted_values + predicted_values
            predicted_values = predicted_values/20
            #print('Error Rate:' + str(predicted_values))
            error_spectrum[vertical/2 + 10, horizontal/2 + 10] = predicted_values

            #numpy.save(model_file+'_Error_Spectrum.npy', error_spectrum)

            #printimage(t_test_set_x[1])
        temp_time_2 = timeit.default_timer()
        print('This loop ran for %.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    scipy.io.savemat(model_file+'error_spectrum.mat', mdict={'Error_Spectrum': error_spectrum})

    return error_spectrum

    #workbook = xlsxwriter.Workbook('CMLP_Trans_Full_Data.xlsx')

    #worksheet = workbook.add_worksheet(model_file)

    #for row in range(0, 21):
    #    for column in range(0, 21):
    #       worksheet.write(row, column, error_spectrum[row, column])
    #workbook.close()


def run_all_model_excel():

    workbook = xlsxwriter.Workbook('CMLP_Trans_Full_Data.xlsx')

    range_x = numpy.array([-4, -2, 2, 4])

    for x in range_x:
        print(x)
        filename = 'Con_MLP_Train_Trans_'+str(x)+'_0.pkl'
        worksheet = workbook.add_worksheet(filename)
        error = translation_prediction(filename)

        for row in range(0, 21):
            for column in range(0, 21):
                if row == 0:
                    worksheet.write(0, column+1, 2*(column-10))
                if column == 0:
                    worksheet.write(row+1, 0, 2*(row-10))
                worksheet.write(row+1, column+1, error[row, column])

        filename = 'Con_MLP_Train_Trans_0_' + str(x) + '.pkl'
        worksheet = workbook.add_worksheet(filename)
        error = translation_prediction(filename)

        for row in range(0, 21):
            for column in range(0, 21):
                if row == 0:
                    worksheet.write(0, column+1, 2*(column-10))
                if column == 0:
                    worksheet.write(row+1, 0, 2*(row-10))
                worksheet.write(row+1, column+1, error[row, column])

    workbook.close()


def run_all_model_matlab():

    horizontal = numpy.array([2, 0, -2, 0, 4, 0, -4, 0, 0])
    vertical = numpy.array([0, -2, 0, 2, 0, -4, 0, 4, 0])

    for shifting_order in range(0, 9, 1):
        print(shifting_order)
        filename = 'Con_MLP_Train_Trans_New' + str(horizontal[shifting_order]) + '_' + str(
                                vertical[shifting_order]) + '.pkl'
        error = translation_prediction(filename)


if __name__ == "__main__":

    #translation_prediction('Con_MLP_Train_Trans_random_After.pkl')
    #translation_prediction('Con_MLP_Ordered_2_0.pkl')
    #run_all_model_matlab()
    #rotation_error_spectrum('Con_MLP_Train_Rand_Rotate.pkl')
    translation_prediction('Con_MLP_Train_Rand_Rotate.pkl')