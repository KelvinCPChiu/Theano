
from __future__ import division
import theano.tensor as T
import theano
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import numpy
import xlsxwriter
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
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

            return T.mean(T.sqr(y - self.output))
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

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input


class ConvPoolLayer_NoMaxPool(object):

    def __init__(self, rng, input, filter_shape, image_shape):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), alpha=0.2)

        self.params = [self.W, self.b]

        self.input = input


def weight_translation(image_tensor_input, horizon_disp, verti_disp):
    tx = image_tensor_input

    def vertical_shift(image_input, displacement):
        tx_out = numpy.zeros(image_input.shape, dtype=float)
        if displacement < 0:
            tx_out[:, :, 0:image_input.shape[2] + displacement, :] = image_input[:, :, -displacement:image_input.shape[2], :]     # Matrix to the bottom
        elif displacement > 0:
            tx_out[:, :, displacement:image_input.shape[2] , :] = image_input[:, :,  0:image_input.shape[2] - displacement, :]     # Matrix to the top
        else:
            tx_out = image_input
        return tx_out

    def horizontal_shift(image_input, displacement):
        tx_out = numpy.zeros(image_input.shape, dtype=float)
        if displacement < 0:
            tx_out[:, :, :, 0:image_input.shape[3] + displacement] = image_input[:, :, :, -displacement:image_input.shape[3]]    # Matrix to the left
        elif displacement > 0:
            tx_out[:, :, :, displacement:image_input.shape[3]] = image_input[:, :, :, 0:image_input.shape[3] - displacement]   # Matrix to the right
        else:
            tx_out = image_input
        return tx_out

    if verti_disp != 0 and horizon_disp == 0:
        txout = vertical_shift(tx, verti_disp)

    if horizon_disp != 0 and verti_disp == 0:
        txout = horizontal_shift(tx, horizon_disp)

    if horizon_disp != 0 and verti_disp != 0:
        txout = vertical_shift(tx, verti_disp)
        txout = horizontal_shift(txout, horizon_disp)

    if verti_disp == 0 and horizon_disp == 0:
        txout = tx

    return txout


def Weight_Open(model_file):

    with open(model_file, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

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


def weight_write_m_file(model_name):

    #   filename = 'Con_MLP_Train_Trans_random_After.pkl'
    #   filename = 'Con_MLP_3.pkl'
    #   filename = 'Con_MLP_Train_Rand_Trans.pkl'
    filename = model_name
    Weight_3, Weight_2, Weight_1, Weight_0 = Weight_Open(filename)

    for page_number in range(2, 4, 1):
        page_name = 'Weight_' + str(page_number)
        layer = eval(page_name)
        layer = layer[0]
        scipy.io.savemat(filename + page_name + '.mat', mdict={page_name: layer})


def Weight_Write(model_name):
    #   Writting Weight Space into excel file
    #   filename = 'Con_MLP_Train_Trans_random_After.pkl'
    #   filename = 'Con_MLP_3.pkl'
    #   filename = 'Con_MLP_Train_Rand_Trans.pkl'
    filename = model_name
    workbook = xlsxwriter.Workbook(filename + '.xlsx')
    Weight_3, Weight_2, Weight_1, Weight_0 = Weight_Open(filename)

    for page_number in range(2, 4, 1):
        page_name = 'Weight_' + str(page_number)
        worksheet = workbook.add_worksheet(page_name)
        page_name = eval(page_name)

        if page_number <= 1:

            filter_number = page_name[0].shape[0]
            filter_z = page_name[0].shape[1]
            filter_x = page_name[0].shape[2]
            filter_y = page_name[0].shape[3]
            b_row_value = page_name[1].shape[0]
            # print(filter_number, filter_x, filter_y, b_row_value)
            for number_of_filter in range(0, filter_number):

                plt.subplot(5, filter_number / 5, number_of_filter + 1)
                plt.pcolor(page_name[0][number_of_filter, 0, :, :])
                # plt.colorbar()
                plt.axis('off')
                plt.savefig(filename + str(page_number) + '.jpg')

                for row in range(0, filter_y):
                    for column in range(0, filter_x):
                        worksheet.write((filter_y + 1) * number_of_filter + row, column,
                                        page_name[0][number_of_filter, 0, row, column])

            for b_row in range(0, b_row_value):
                worksheet.write(b_row, filter_x + 1, page_name[1][b_row])
        total = 0
        if page_number >= 2:
            row_value = page_name[0].shape[0]
            column_value = page_name[0].shape[1]
            b_row_value = page_name[1].shape[0]
            # print(row_value, column_value, b_row_value)
            for row in range(0, row_value):
                for column in range(0, column_value):
                    worksheet.write(row, column, page_name[0][row, column])
                    total = total + page_name[0][row, column]
            mean = total / row_value / column_value
            r_m_s = numpy.sqrt(numpy.sum(page_name[0]**2)/row_value/column_value)
            sd = numpy.sqrt(numpy.sum(numpy.sum((page_name[0] - mean) ** 2))) / row_value / column_value
            worksheet.write(0, column_value, 'Root Mean Square')
            worksheet.write(1, column_value, r_m_s)
            worksheet.write(2, column_value, 'S.D')
            worksheet.write(3, column_value, sd)
            for b_row in range(0, b_row_value):
                worksheet.write(b_row, column_value + 1, page_name[1][b_row])

    workbook.close()


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
                [morphing_coeff, morphing_coeff2, morphing3]]

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


def Generate_Continuous_Set(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def class_image(input_image, set_size):

        morphing_coeff = numpy.repeat([numpy.linspace(0, 1, set_size)], 100, axis=0).flatten()

        morphing1 = numpy.zeros(10 * 10 * set_size)
        morphing2 = numpy.zeros(10 * 10 * set_size)

        for set1 in range(0, 10, 1):
            for set2 in range(0, 10, 1):

                    morphing1[set_size * 10 * set1: set_size * 10 * (set1 + 1)] = set1
                    morphing2[set_size * (set1*10 + set2):  set_size * (set2 + 1 + set1*10)] = set2
                    #   print morphing1[set_size * 10 * set1: set_size * 10 * (set1 + 1)]

        morphing1 = morphing1.afstype(int)
        morphing2 = morphing2.astype(int)

        resultant_set = morphing_coeff[:, None, None] * input_image[morphing1] + (1 - morphing_coeff)[:, None, None] * input_image[morphing2]

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


def Generate_Triangular_Set(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def class_image(input_image, set_size):

        #input_image = numpy.zeros((set_size, input_image.shape[1], input_image.shape[2]))

        morphing1 = numpy.random.random_integers(0, 2, set_size)
        morphing2 = numpy.random.random_integers(0, 2, set_size)
        morphing3 = numpy.random.random_integers(0, 2, set_size)

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


def Generate_Linear_Triangular_Set(raw_image_set, size_desired):

    def one_hot(imput_class, number_of_class):
        imput_class = numpy.array(imput_class)
        assert imput_class.ndim == 1
        return numpy.eye(number_of_class)[imput_class]

    def class_image(input_image, set_size):
        # input_image = numpy.zeros((set_size, input_image.shape[1], input_image.shape[2]))

        morphing1 = numpy.random.random_integers(0, 2, set_size)
        morphing2 = numpy.random.random_integers(0, 2, set_size)
        morphing3 = numpy.random.random_integers(0, 2, set_size)

        morphing_coeff = numpy.random.random(set_size)
        morphing_coeff2 = numpy.random.random(set_size)
        morphing_coeff3 = numpy.random.random(set_size)

        resultant_set = morphing_coeff[:, None, None] * input_image[morphing1] + morphing_coeff2[:, None, None] * input_image[
                                                                                     morphing2] + morphing_coeff3[:,
                                                                                                  None, None] * \
                                                                                                  input_image[morphing3]

        resultant_set = resultant_set / ((morphing_coeff3 + morphing_coeff2 + morphing_coeff)[:, None, None])
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
    generated_label_set = (
                          set_order1 * constant[0][:, None] + set_order2 * constant[1][:, None] + set_order3 * constant[2][:,None]) / constant_sum[:,None]
    return shared_dataset(generated_image_set, generated_label_set)


def triangular_data(model_name):

    batch_size = 500
    datasets = numpy.load('Gaussian_Data_Set.npy')
    data_points = 500000
    valid_set_x, valid_set_y = Generate_Triangular_Set(datasets, data_points)

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    n_valid_batches = n_valid//batch_size

    y = T.fmatrix('y')
    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.output,
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    validate_label = [validate_model(i) for i in range(n_valid_batches)]
    validate_label = numpy.asarray(validate_label)
    validate_label = validate_label.reshape(data_points, 10)
    valid_set_y = valid_set_y.eval()

    scipy.io.savemat(model_name+'_Ground_Truth_Data_3_labels_first3random_500k.mat', mdict={'Ground_Truth': valid_set_y})
    scipy.io.savemat(model_name+'_Predicted_Data_3_labels_first3random_500k.mat', mdict={'Predicted': validate_label})


if __name__ == "__main__":
    name = 'Updated_Gaussian_Model_Test_3.pkl'
    triangular_data(name)

