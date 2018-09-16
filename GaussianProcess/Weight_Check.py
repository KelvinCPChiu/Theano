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
from matplotlib.patches import Rectangle


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


def weight_write_filter_graph(model_name):
    # filename = 'Con_MLP_Train_Trans_random_After.pkl'
    # filename = 'Con_MLP_3.pkl'
    #filename = 'Con_MLP_Train_Rand_Trans.pkl'
    filename = model_name
    Weight_3, Weight_2, Weight_1, Weight_0 = Weight_Open(filename)

    for page_number in range(0, 2, 1):
        page_name = 'Weight_' + str(page_number)
        page_name = eval(page_name)

        filter_number = page_name[0].shape[0]
        filter_z = page_name[0].shape[1]
        filter_x = page_name[0].shape[2]
        filter_y = page_name[0].shape[3]

        if page_number == 0:
            #   Only write the first slice of 20.

            for number_of_filter in range(0, filter_number):
                print('Layer:' + str(page_number) + ' ; weight_write_filter_graph:' + str(number_of_filter))
                plt.subplot(5, filter_number / 5, number_of_filter + 1)
                plt.pcolor(page_name[0][number_of_filter, 0, :, :])
                #, vmin=-0.7, vmax=0.7)
                plt.axis('off')
            #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            #cax = plt.axes([0.825, 0.1, 0.075, 0.8])
            #plt.colorbar(cax=cax)
            plt.savefig(filename + str(page_number) + 'filters.jpg')

        if page_number == 1:

            #   Writting all 50*20 filters for layer 1
            fig = plt.gcf()
            DPI = fig.get_dpi()
            fig.set_size_inches(2000 / float(DPI), 5000 / float(DPI))

            for number_of_filter in range(0, filter_number):
                print('Layer:' + str(page_number) + ' ; weight_write_filter_graph:' + str(number_of_filter))
                for r_z in range(0, filter_z, 1):
                    r_m_s = round(
                        numpy.sqrt(numpy.sum(page_name[0][number_of_filter, r_z, :, :] ** 2) / filter_x / filter_y)
                        * 100, 1)
                    ax = plt.subplot2grid((filter_number, filter_z), (number_of_filter, r_z))
                    ax.pcolor(page_name[0][number_of_filter, r_z, :, :])
                    #, vmin=-0.05, vmax=0.05)
                    # ax.set_title(str(number_of_filter + 1) + ';' + str(r_z + 1) + ';' + str(r_m_s), fontsize=10)
                    ax.set_title(str(number_of_filter + 1) + ';' + str(r_z + 1), fontsize=10)
                    plt.axis('off')

            plt.savefig(filename + str(page_number) + 'filters.jpg')


def weight_write_filter_graph_update(model_name):

    def boundary(sub1):
        autoAxis = sub1.axis()
        rec = Rectangle((autoAxis[0] - 0.7, autoAxis[2] - 0.2), (autoAxis[1] - autoAxis[0]) + 1,
                        (autoAxis[3] - autoAxis[2]) + 0.4, fill=False, lw=2, edgecolor='r')
        rec = sub1.add_patch(rec)
        rec.set_clip_on(False)

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

    filename = model_name + '.pkl'
    Weight_3, Weight_2, Weight_1, Weight_0 = Weight_Open(filename)
    initial_filename = model_name + '_Initial.pkl'
    Weight_3_i, Weight_2_i, Weight_1_i, Weight_0_i = Weight_Open(initial_filename)

    for page_number in range(0, 2, 1):
        page_name = 'Weight_' + str(page_number)
        page_name = eval(page_name)
        initial_page_name = 'Weight_' + str(page_number) + '_i'
        initial_page_name = eval(initial_page_name)

        cross_value = cross_corre_value(page_name[0], initial_page_name[0]) >= 0.8

        filter_number = page_name[0].shape[0]
        filter_z = page_name[0].shape[1]
        filter_x = page_name[0].shape[2]
        filter_y = page_name[0].shape[3]
        fig1 = plt.figure()

        if page_number == 0:
            #   Only write the first slice of 20.

            for number_of_filter in range(0, filter_number):
                print('Layer:' + str(page_number) + ' ; weight_write_filter_graph:' + str(number_of_filter))
                temp = plt.subplot(5, 5, number_of_filter + 1)
                temp.pcolor(page_name[0][number_of_filter, 0, :, :])
                if cross_value[number_of_filter] == 1:
                    boundary(temp)
                #, vmin=-0.7, vmax=0.7)
                temp.axis('off')
                #plt.colorbar()
            #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            #cax = plt.axes([0.825, 0.1, 0.075, 0.8])
            #plt.colorbar(cax=cax)
            plt.savefig(filename + str(page_number) + 'filters.jpg')

        if page_number == 1:

            #   Writting all 50*20 filters for layer 1
            fig = plt.gcf()
            DPI = fig.get_dpi()
            fig.set_size_inches(2000 / float(DPI), 5000 / float(DPI))

            for number_of_filter in range(0, filter_number):
                print('Layer:' + str(page_number) + ' ; weight_write_filter_graph:' + str(number_of_filter))
                for r_z in range(0, filter_z, 1):
                    r_m_s = round(
                        numpy.sqrt(numpy.sum(page_name[0][number_of_filter, r_z, :, :] ** 2) / filter_x / filter_y)
                        * 100, 1)
                    ax = plt.subplot2grid((filter_number, filter_z), (number_of_filter, r_z))
                    ax.pcolor(page_name[0][number_of_filter, r_z, :, :])
                    if cross_value[number_of_filter, r_z] == 1:
                        boundary(ax)
                    #, vmin=-0.05, vmax=0.05)
                    ax.set_title(str(number_of_filter + 1) + ';' + str(r_z + 1), fontsize=10)
                    plt.axis('off')

            plt.savefig(filename + str(page_number) + 'filters.jpg')


def weight_4d_correlation_filter_z_dim(filename, page_name, page_number):

    # Averaging over the w.r.t individual filter
    # Some problems on layer 0 filters
    filter_number = page_name.shape[0]
    filter_z = page_name.shape[1]
    filter_y = page_name.shape[2]
    filter_x = page_name.shape[3]

    r_value = numpy.zeros((filter_number, filter_z, filter_y * 2 - 1, filter_x * 2 - 1))
    for r_y in range(-filter_y + 1, filter_y, 1):
        print('Layer:' + str(page_number) + 'rx:' + str(r_y))
        for r_x in range(-filter_x + 1, filter_x, 1):
            temp_weight = weight_translation(page_name, r_x, r_y)

            denominator = float(1/filter_y/filter_x)

            #   denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))
            #   denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)) / filter_z)
            #   denominator = float(1 / filter_y / filter_x / filter_z)
            #   weight_mean = numpy.sum(numpy.sum(page_name, axis=3), axis=2) * denominator
            weight_mean = numpy.sum(numpy.sum(page_name, axis=3), axis=2) * denominator
            #   print('weight_mean'+numpy.isnan(weight_mean))

            temp_weight_mean = numpy.sum(numpy.sum(temp_weight, axis=3), axis=2) * denominator

            #   temp_weight_mean = numpy.sum(numpy.sum(temp_weight, axis=3), axis=2) * denominator

            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(page_name ** 2, axis=3), axis=2) *
                                            denominator - weight_mean ** 2))

            # print('weight_mean:'+str(weight_mean))
            # print('input:'+str(numpy.sum(numpy.sum(numpy.sum(page_name ** 2, axis=3), axis=2), axis=1)))
            # print('sd1:'+str(sd1))
            sd2 = numpy.sqrt(
                numpy.absolute(numpy.sum(numpy.sum(temp_weight ** 2, axis=3), axis=2) *
                               denominator - temp_weight_mean ** 2))
            # print(sd2)
            # print('sd2:'+str(sd2))
            sd = sd1 * sd2
            # print('sd:'+str(sd))
            #   sd = numpy.sqrt((numpy.sum(numpy.sum(page_name ** 2, axis=3), axis=2) *
            #                 denominator - weight_mean ** 2) * (numpy.sum(numpy.sum(
            #    temp_weight ** 2, axis=3), axis=2) * denominator - temp_weight_mean ** 2))

            temp_r_value = (numpy.sum(numpy.sum(temp_weight * page_name, axis=3), axis=2)
                            * denominator - weight_mean * temp_weight_mean) / sd
            #   temp_r_value = (numpy.sum(numpy.sum(temp_weight * page_name, axis=3), axis=2)
            #                * denominator - weight_mean * temp_weight_mean) / sd
            # print('temp_r_value:'+str(temp_r_value))
            r_value[:, :, r_y + filter_y - 1, r_x + filter_x - 1] = temp_r_value[:, :]

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2000 / float(DPI), 5000 / float(DPI))

    for number_of_filter in range(0, filter_number):
        for r_z in range(0, filter_z, 1):
            ax = plt.subplot2grid((filter_number, filter_z), (number_of_filter, r_z))
            #   plt.subplot(5, filter_number / 5, number_of_filter + 1)
            ax.pcolor(r_value[number_of_filter, r_z, :, :], vmin=-1, vmax=1)
            ax.set_title(str(number_of_filter+1)+';'+str(r_z+1))
            plt.axis('off')
    #   plt.colorbar()
    #plt.savefig(filename + str(page_number) + '_all_filters_Weight_Correlation.jpg')


def weight_4d_correlation_filter(filename, page_name, page_number):

    #   Including averaging over the .shape[1] dimensions of that set of filter
    #   Working good for layer 0
    filter_number = page_name.shape[0]
    filter_z = page_name.shape[1]
    filter_y = page_name.shape[2]
    filter_x = page_name.shape[3]

    r_value = numpy.zeros((filter_number, filter_y*2-1, filter_x*2-1))
    for r_y in range(-filter_y + 1, filter_y, 1):
        print('Layer:' + str(page_number) + '; r_y:' + str(r_y))
        for r_x in range(-filter_x + 1, filter_x, 1):
            temp_weight = weight_translation(page_name, r_x, r_y)

            #   denominator = float(1/filter_y/filter_x)
            #   denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))
            #   denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x))/filter_z)
            denominator = float(1 / filter_y / filter_x / filter_z)
            #   weight_mean = numpy.sum(numpy.sum(page_name, axis=3), axis=2) * denominator
            weight_mean = numpy.sum(numpy.sum(numpy.sum(page_name, axis=3), axis=2), axis=1) * denominator
            #   print('weight_mean'+numpy.isnan(weight_mean))

            temp_weight_mean = numpy.sum(numpy.sum(numpy.sum(temp_weight, axis=3), axis=2), axis=1) * denominator

            #   temp_weight_mean = numpy.sum(numpy.sum(temp_weight, axis=3), axis=2) * denominator

            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(page_name ** 2, axis=3), axis=2), axis=1) *
                                            denominator - weight_mean ** 2))

            #print('weight_mean:'+str(weight_mean))
            #print('input:'+str(numpy.sum(numpy.sum(numpy.sum(page_name ** 2, axis=3), axis=2), axis=1)))
            #print('sd1:'+str(sd1))
            sd2 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(temp_weight ** 2, axis=3), axis=2), axis=1) *
                                            denominator - temp_weight_mean ** 2))
            #print(sd2)
            #print('sd2:'+str(sd2))
            sd = sd1*sd2
            #print('sd:'+str(sd))
            #   sd = numpy.sqrt((numpy.sum(numpy.sum(page_name ** 2, axis=3), axis=2) *
            #                 denominator - weight_mean ** 2) * (numpy.sum(numpy.sum(
            #    temp_weight ** 2, axis=3), axis=2) * denominator - temp_weight_mean ** 2))

            temp_r_value = (numpy.sum(numpy.sum(numpy.sum(temp_weight * page_name, axis=3), axis=2), axis=1)
                            * denominator - weight_mean * temp_weight_mean) / sd
            #   temp_r_value = (numpy.sum(numpy.sum(temp_weight * page_name, axis=3), axis=2)
            #                * denominator - weight_mean * temp_weight_mean) / sd
            #print('temp_r_value:'+str(temp_r_value))
            r_value[:, r_y + filter_y - 1, r_x + filter_x - 1] = temp_r_value[:]

    for number_of_filter in range(0, filter_number):
        fig = plt.subplot(5, filter_number / 5, number_of_filter + 1)
        plt.pcolor(r_value[number_of_filter, :, :], vmin=-1, vmax=1)
        plt.axis('off')
    #   plt.colorbar()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.825, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    #plt.show()
    plt.savefig(filename + str(page_number) + 'filters_Weight_Correlation.jpg')


def weight_4d_correlation_all(filename, page_name, page_number):

    #   Get a general correlation for all filters within one layer

    filter_number = page_name.shape[0]
    filter_z = page_name.shape[1]
    filter_y = page_name.shape[2]
    filter_x = page_name.shape[3]

    r_value = numpy.zeros((filter_y * 2 - 1, filter_x * 2 - 1))
    for r_y in range(-filter_y + 1, filter_y, 1):
        print('Layer:' + str(page_number) + ' ; r_y:' + str(r_y))
        for r_x in range(-filter_x + 1, filter_x, 1):
            temp_weight = weight_translation(page_name, r_x, r_y)
            denominator = float(1 / filter_y / filter_x / filter_z / filter_number)
            #   denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)) / filter_z / filter_number)
            #   filter_number/filter_z)
            weight_mean = numpy.sum(page_name) * denominator

            temp_weight_mean = numpy.sum(temp_weight) * denominator

            sd1 = float(numpy.sqrt(numpy.absolute(numpy.sum(page_name ** 2)*denominator - weight_mean ** 2)))
            #   print(numpy.sqrt(numpy.sum(page_name ** 2))*denominator)
            sd2 = float(numpy.sqrt(numpy.absolute(numpy.sum(temp_weight ** 2)*denominator - temp_weight_mean ** 2)))
            #   print('sd1:'+str(sd1) + ';sd2:'+str(sd2))

            sd = sd1*sd2
            r_value[r_y + filter_y - 1, r_x + filter_x - 1] = (numpy.sum(
                temp_weight * page_name) * denominator - weight_mean *
                                                               temp_weight_mean) / sd
            #   print('denominator:' + str(denominator))
            #   print('weight_mean:' + str(weight_mean))
            #   print('sd:' + str(sd))
            #   print('temp_weight:' + str(temp_weight_mean))
    #   print(r_value)
    plt.subplot(1, 2, page_number+1)
    plt.pcolor(r_value, vmin=-0.3, vmax=1)
    plt.title('Layer'+str(page_number))
    #    , vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig(filename + str(page_number) + 'Weight_Correlation.jpg')


def normalized_cross_correlation_2(filename_1, filename_2):
    Model_1 = Weight_Open(filename_1)
    Model_2 = Weight_Open(filename_2)

    for layer_number in range(0, 2, 1):

        Model_1_Layer = Model_1[3 - layer_number][0]
        Model_2_Layer = Model_2[3 - layer_number][0]

        Model_1_filter_number = Model_1_Layer.shape[0]
        filter_z = Model_1_Layer.shape[1]
        filter_y = Model_1_Layer.shape[2]
        filter_x = Model_1_Layer.shape[3]
        print('Layer Number :' + str(layer_number))

        r_value = numpy.zeros(Model_1_filter_number)

        if layer_number <= 1:
            denominator = float(1 / filter_y / filter_x / filter_z)
            Model_1_Weight_mean = numpy.sum(numpy.sum(numpy.sum(Model_1_Layer, axis=3), axis=2), axis=1) * denominator
            Model_2_Weight_mean = numpy.sum(numpy.sum(numpy.sum(Model_2_Layer, axis=3), axis=2), axis=1) * denominator
            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(Model_1_Layer ** 2, axis=3), axis=2), axis=1) *
                                            denominator - Model_1_Weight_mean ** 2))
            sd2 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(Model_2_Layer ** 2, axis=3), axis=2), axis=1) *
                                            denominator - Model_2_Weight_mean ** 2))

            sd = sd1 * sd2

            temp_r_value = (numpy.sum(numpy.sum(numpy.sum(Model_1_Layer * Model_2_Layer, axis=3), axis=2), axis=1)
                            * denominator - Model_1_Weight_mean * Model_2_Weight_mean) / sd
            #r_value[:] = temp_r_value[:]
            r_value = numpy.sum(temp_r_value)/Model_1_filter_number
            print(r_value)

        if layer_number >= 2:
            pass

        return r_value


def normalized_cross_correlation(filename_1, filename_2):

    # R.M.S of the filter normalized cross correlation

    Model_1 = Weight_Open(filename_1)
    Model_2 = Weight_Open(filename_2)

    r_value_two_layers = numpy.zeros(4)

    for layer_number in range(0, 4, 1):

        Model_1_Layer = Model_1[3 - layer_number][0]
        Model_2_Layer = Model_2[3 - layer_number][0]

        if layer_number <= 1:

            Model_1_filter_number = Model_1_Layer.shape[0]
            filter_z = Model_1_Layer.shape[1]
            filter_y = Model_1_Layer.shape[2]
            filter_x = Model_1_Layer.shape[3]
            # print('Layer Number :' + str(layer_number))

            # r_value = numpy.zeros(Model_1_filter_number)

            #   denominator = float(1 / filter_y / filter_x / filter_z)
            denominator = float(1 / filter_y / filter_x)
            #   Model_1_Weight_mean = numpy.sum(numpy.sum(numpy.sum(Model_1_Layer, axis=3), axis=2), axis=1) * denominator
            Model_1_Weight_mean = numpy.sum(numpy.sum(Model_1_Layer, axis=3), axis=2) * denominator
            #   Model_2_Weight_mean = numpy.sum(numpy.sum(numpy.sum(Model_2_Layer, axis=3), axis=2), axis=1) * denominator
            Model_2_Weight_mean = numpy.sum(numpy.sum(Model_2_Layer, axis=3), axis=2) * denominator
            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(Model_1_Layer ** 2, axis=3), axis=2) *
                                            denominator - Model_1_Weight_mean ** 2))
            sd2 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(Model_2_Layer ** 2, axis=3), axis=2) *
                                            denominator - Model_2_Weight_mean ** 2))
            #   sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(Model_1_Layer ** 2, axis=3), axis=2), axis=1) *
            #                                   denominator - Model_1_Weight_mean ** 2))
            #   sd2 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(Model_2_Layer ** 2, axis=3), axis=2), axis=1) *
            #                                   denominator - Model_2_Weight_mean ** 2))

            sd = sd1 * sd2

            temp_r_value = (numpy.sum(numpy.sum(Model_1_Layer * Model_2_Layer, axis=3), axis=2)
                            * denominator - Model_1_Weight_mean * Model_2_Weight_mean) / sd

            r_value = numpy.sqrt(
                numpy.sum(numpy.sum(temp_r_value ** 2, axis=1), axis=0) / filter_z / Model_1_filter_number)

            r_value_two_layers[layer_number] = r_value

            # scipy.io.savemat(filename_1+str(layer_number) + '.mat', mdict={r_value: temp_r_value})
            # temp_r_value = (numpy.sum(numpy.sum(numpy.sum(Model_1_Layer * Model_2_Layer, axis=3), axis=2), axis=1)
            #                * denominator - Model_1_Weight_mean * Model_2_Weight_mean) / sd
            #   temp_r_value = (numpy.sum(numpy.sum(temp_weight * page_name, axis=3), axis=2)
            #                * denominator - weight_mean * temp_weight_mean) / sd
            # print('temp_r_value:'+str(temp_r_value))
            # r_value[:] = temp_r_value[:]
            # print(r_value)

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

    print(r_value_two_layers)
    return r_value_two_layers


def run_normailzed_cross_corre():
    x = numpy.array([2, 0, -2, 0, 4, 0, -4, 0, 0])
    y = numpy.array([0, -2, 0, 2, 0, -4, 0, 4, 0])
    #x = numpy.array([-2, -4, 2, 4, 0, 0, 0, 0, 0])
    #y = numpy.array([0, 0, 0, 0, -2, -4, 2, 4, 0])
    r_value = numpy.zeros((30, 2))
    for iteration in range(-1, 29, 1):
        print('Iteration Number'+str(iteration))

        if iteration == -1:
            r_value[iteration+1, :] = normalized_cross_correlation(
                'Con_MLP_3.pkl',
                'Con_MLP_Train_Trans_'+str(x[0])+'_'+str(y[0])+'.pkl')

        if 0 <= iteration <= 7:
            r_value[iteration + 1, :] = normalized_cross_correlation(
                'Con_MLP_Train_Trans_' + str(x[iteration]) + '_' + str(y[iteration]) + '.pkl',
                'Con_MLP_Train_Trans_' + str(x[iteration+1]) + '_' + str(y[iteration+1]) + '.pkl')

        if iteration == 8:
            r_value[iteration + 1, :] = normalized_cross_correlation(
                'Con_MLP_Train_Trans_random_After.pkl',
                'Con_MLP_Train_Trans_' + str(x[8]) + '_' + str(y[8]) + '.pkl')

        if iteration == 9:
            r_value[iteration + 1, :] = normalized_cross_correlation(
                'Con_MLP_Train_Trans_random_After.pkl',
                'Con_MLP_Train_Rand_Trans.pkl')
        if iteration == 10:
            r_value[iteration+1, :] = normalized_cross_correlation(
                'Con_MLP_Train_Trans_random_After.pkl',
                'Con_MLP_3.pkl')

        if 11 <= iteration <= 19:
            r_value[iteration + 1, :] = normalized_cross_correlation(
                'Con_MLP_3.pkl',
                'Con_MLP_Train_Trans_' + str(x[iteration-11]) + '_' + str(y[iteration-11]) + '.pkl')

        if iteration >= 20:
            r_value[iteration + 1, :] = normalized_cross_correlation(
                'Con_MLP_Train_Rand_Trans.pkl',
                'Con_MLP_Train_Trans_' + str(x[iteration - 20]) + '_' + str(y[iteration - 20]) + '.pkl')

        print(r_value[iteration + 1, :])

    print(r_value)
    scipy.io.savemat('r_value_new.mat', mdict={'cross_correlation': r_value})


def weight_correlation(model_name):

    #   Used to call all different correlation function used
    filename = model_name
    #   filename = 'Con_MLP_3.pkl'
    #   filename = 'Con_MLP_Train_Rand_Trans.pkl'
    #   filename = 'Con_MLP_Train_Trans_random_After.pkl'
    Weight_3, Weight_2, Weight_1, Weight_0 = Weight_Open(filename)

    for page_number in range(0, 2, 1):
        page_name = 'Weight_' + str(page_number)
        page_name = eval(page_name)
        page_name = page_name[0]
        #   print(page_name.shape)
        print('Layer Number :' + str(page_number))

        if page_number <= 1:
            #weight_4d_correlation_filter(filename, page_name, page_number)
            #weight_4d_correlation_all(filename, page_name, page_number)
            weight_4d_correlation_filter_z_dim(filename, page_name, page_number)

        if page_number >= 2:
            pass


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


def test_and_compare(model_name):

    batch_size = 500
    datasets = numpy.load('Gaussian_Data_Set.npy')
    total_size = 10000
    valid_set_x, valid_set_y = Generate_Test_Set(datasets, total_size)

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    #valid_set_x = valid_set_x.reshape((n_valid, 784))
    n_valid_batches = n_valid//batch_size

    y = T.fmatrix('y')
    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.output,
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    validate_label = [validate_model(i) for i in range(n_valid_batches)]
    validate_label = numpy.asarray(validate_label)
    validate_label = validate_label.reshape(total_size, 10)

    valid_set_y = valid_set_y.eval()
    scipy.io.savemat(model_name+'_Ground_Truth_Data_3_labels_10k.mat', mdict={'Ground_Truth': valid_set_y})
    scipy.io.savemat(model_name+'_Predicted_Data_3_labels_10k.mat', mdict={'Predicted': validate_label})


def continuous_morphing(model_name):

    batch_size = 500

    datasets = numpy.load('Gaussian_Data_Set.npy')

    valid_set_x, valid_set_y = Generate_Continuous_Set(datasets, 100)

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))

    #valid_set_x = valid_set_x.reshape((n_valid, 784))
    n_valid_batches = n_valid//batch_size

    y = T.fmatrix('y')

    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer2, layer3 = pickle.load(f)
        #layer0, layer1, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.output,
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    predicted_label = [validate_model(i) for i in range(n_valid_batches)]
    predicted_label = numpy.asarray(predicted_label)
    predicted_label = predicted_label.reshape(n_valid, 10)

    valid_set_y = valid_set_y.eval()
    scipy.io.savemat(model_name + 'Ground_Truth_continouous_100.mat', mdict={'Ground_Truth': valid_set_y})
    scipy.io.savemat(model_name + 'Predicted_continuous_100.mat', mdict={'Predicted': predicted_label})


def test_transformation_learnt(model_name):

    def shared_dataset(data_x, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        return shared_x

    batch_size = 500

    datasets = numpy.load('Gaussian_Data_Set_Second.npy')

    valid_set_x = shared_dataset(numpy.repeat(datasets, 50, axis=0))

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))

    #valid_set_x = valid_set_x.reshape((n_valid, 784))
    n_valid_batches = n_valid//batch_size

    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer2, layer3 = pickle.load(f)
        #layer0, layer1, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.output,
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    predicted_label = [validate_model(i) for i in range(n_valid_batches)]
    predicted_label = numpy.asarray(predicted_label)
    predicted_label = predicted_label.reshape(n_valid, 10)

    #valid_set_y = valid_set_y.eval()
    #scipy.io.savemat(model_name + 'Ground_Truth_continouous_100.mat', mdict={'Ground_Truth': valid_set_y})
    scipy.io.savemat(model_name + 'Robust.mat', mdict={'Predicted': predicted_label})


def robustness_test(model_name):

    def shared_dataset(data_x, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        return shared_x

    batch_size = 500

    datasets = numpy.load('Gaussian_Data_Set_Second.npy')

    valid_set_x = shared_dataset(numpy.repeat(datasets, 50, axis=0))

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))

    #valid_set_x = valid_set_x.reshape((n_valid, 784))
    n_valid_batches = n_valid//batch_size

    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)


    validate_model = theano.function(
        [index],
        layer3.output,
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    predicted_label = [validate_model(i) for i in range(n_valid_batches)]
    predicted_label = numpy.asarray(predicted_label)
    predicted_label = predicted_label.reshape(n_valid, 10)

    #valid_set_y = valid_set_y.eval()
    #scipy.io.savemat(model_name + 'Ground_Truth_continouous_100.mat', mdict={'Ground_Truth': valid_set_y})
    scipy.io.savemat(model_name + 'Robust.mat', mdict={'Predicted': predicted_label})


def position333(model_name):

    batch_size = 500

    def Generate_333_set(raw_image_set, size_desired):
        def one_hot(imput_class, number_of_class):
            imput_class = numpy.array(imput_class)
            assert imput_class.ndim == 1
            return numpy.eye(number_of_class)[imput_class]

        def class_image(input_image, set_size):

            morphing1 = numpy.zeros(set_size).astype(int)
            morphing2 = numpy.ones(set_size).astype(int)
            morphing3 = (numpy.ones(set_size)*2).astype(int)

            morphing_coeff = numpy.ones(set_size)/3
            morphing_coeff2 = numpy.ones(set_size)/3
            morphing_coeff3 = numpy.ones(set_size)/3

            resultant_set = morphing_coeff[:, None, None] * input_image[morphing1] + morphing_coeff2[:, None, None] * \
                                                                                     input_image[
                                                                                         morphing2] + morphing_coeff3[:,
                                                                                                      None, None] * \
                                                                                                      input_image[
                                                                                                          morphing3]

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
        generated_label_set = (set_order1 * constant[0][:, None] + set_order2 * constant[1][:, None] + set_order3 *
                               constant[2][:, None]) / constant_sum[:, None]

        return shared_dataset(generated_image_set, generated_label_set)

    datasets = numpy.load('Gaussian_Data_Set.npy')

    test_set_x, test_set_y = Generate_333_set(datasets, 500)

    n_valid = test_set_x.get_value(borrow=True).shape[0]
    valid_set_x = test_set_x.reshape((n_valid, 1, 28, 28))

    n_valid_batches = n_valid//batch_size

    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.output,
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    predicted_label = [validate_model(i) for i in range(n_valid_batches)]
    predicted_label = numpy.asarray(predicted_label)
    predicted_label = predicted_label.reshape(n_valid, 10)

    scipy.io.savemat(model_name + '333Point.mat', mdict={'Predicted': predicted_label})


def all_ten_mix(model_name):

    batch_size = 500

    def Generate_all_10(raw_image_set, size_desired):

        def class_image(input_image):

            morphing_coeff = numpy.ones((10,1))*0.1
            resultant_set = numpy.sum(input_image*morphing_coeff[:, None, None], axis=0)
            print resultant_set.shape

            return [resultant_set, morphing_coeff]

        def shared_dataset(data_x, data_y, borrow=True):
            shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

            return shared_x, shared_y

        generated_image_set, generated_label_set = class_image(raw_image_set)
        print(generated_image_set.shape)
        print(generated_label_set.shape)
        generated_image_set = numpy.repeat(generated_image_set, size_desired, axis=0)
        generated_label_set = numpy.repeat(generated_label_set, size_desired, axis=1)
        print(generated_image_set.shape)
        print(generated_label_set.shape)
        return shared_dataset(generated_image_set, generated_label_set)

    datasets = numpy.load('Gaussian_Data_Set.npy')

    test_set_x, test_set_y = Generate_all_10(datasets, 500)
    n_valid = test_set_x.get_value(borrow=True).shape[0]
    valid_set_x = test_set_x.reshape((n_valid, 1, 28, 28))

    n_valid_batches = n_valid//batch_size

    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.output,
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    predicted_label = [validate_model(i) for i in range(n_valid_batches)]
    predicted_label = numpy.asarray(predicted_label)
    predicted_label = predicted_label.reshape(n_valid, 10)

    scipy.io.savemat(model_name + 'all_10.mat', mdict={'Predicted': predicted_label})


def turn_off_filter(model_name):

    batch_size = 500
    datasets = numpy.load('Gaussian_Data_Set.npy')
    total_size = 10000
    valid_set_x, valid_set_y = Generate_Set(datasets, total_size)

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    #valid_set_x = valid_set_x.reshape((n_valid, 784))
    n_valid_batches = n_valid//batch_size

    y = T.fmatrix('y')
    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer3 = pickle.load(f)

    #print layer0.W.eval()
    temp_W = theano.shared(numpy.zeros(layer0.W.shape.eval(), dtype=theano.config.floatX), borrow=True)
    #layer0.W = theano.tensor.set_subtensor(layer0.W[[2, 5, 8, 9, 13, 15], :, :, :], temp_W[[2, 5, 8, 9, 13, 15], :, :, :])
    temp_W = theano.tensor.set_subtensor(temp_W[[2, 5, 8, 9, 13, 15], :, :, :], layer0.W[[2, 5, 8, 9, 13, 15], :, :, :])
    layer0.W = temp_W

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    validation_losses = [validate_model(i) for i in range(n_valid_batches)]
    this_validation_loss = numpy.mean(validation_losses)
    print(this_validation_loss)
    #validate_label = numpy.asarray(validate_label)
    #validate_label = validate_label.reshape(total_size, 10)

    #valid_set_y = valid_set_y.eval()
    #scipy.io.savemat(model_name+'_Ground_Truth_Data_TurnOff_1.mat', mdict={'Ground_Truth': valid_set_y})
    #scipy.io.savemat(model_name+'_Predicted_Data_TurnOff_1.mat', mdict={'Predicted': validate_label})


if __name__ == "__main__":
    name = 'Updated_Gaussian_Model_Test_3.pkl'
    #turn_off_filter(name)
    #all_ten_mix(name)
    #weight_write_filter_graph_update(name)
    #for indecies in range(3, 11, 1):
    #    name = 'Updated_Gaussian_Model_Test_3_FirstLayer'+str(indecies)+'.pkl'
    #    weight_write_filter_graph_update(name)

    #run_normailzed_cross_corre()
    #normalized_cross_correlation('Updated_Gaussian_Model_Test_3_Seed3.pkl', 'Updated_Gaussian_Model_Test_3_Seed3_WhiteNoise.pkl')
    #normalized_cross_correlation('Updated_Gaussian_Model_Test_3_Seed3.pkl', 'Updated_Gaussian_Model_Test_3_initial_stage_seed3.pkl')
    #normalized_cross_correlation('Updated_Gaussian_Model_Test_3_initial_stage_seed3.pkl', 'Updated_Gaussian_Model_Test_3_Seed3_WhiteNoise.pkl')
    #model_loaded = 'Con_MLP_Train_Rand_Trans_tanh_best.pkl'
    #model_loaded = 'Con_MLP_Train_Rand_Trans_relu_best.pkl'
    #weight_correlation(name)
    # Weight_Write(model_loaded)
    #weight_write_filter_graph(model_loaded)
    #model_loaded = 'Con_MLP_3.pkl'
    #model_loaded = 'Con_MLP_Train_Rand_Trans.pkl'
    #model_loaded = 'Con_MLP_Train_Trans_random_After.pkl'
    #for number in range(-4, 5, 2):
    #model_loaded = 'Con_MLP_Train_Trans_0_'+str(number)+'.pkl'
    #model_loaded = 'Con_MLP_Train_Trans_0_0.pkl'
    #model_loaded = 'Con_MLP_relu_decay.pkl'
    #weight_write_filter_graph('Gaussian_Model_1_SoftMax.pkl')
    #  weight_write_filter_graph(model_loaded)
    #weight_write_m_file(model_loaded)
    #   Weight_Write(model_loaded)
    weight_correlation(name)
    #continuous_morphing(name)
    #print 'Printing Label...'
    #test_and_compare(name)
    #print 'Writting Filter Graph...'
    #robustness_test(name)
