from __future__ import division
import theano.tensor as T
import theano
import numpy
import xlsxwriter
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import scipy.io
from Tranlation import loaddata_mnist
from matplotlib.patches import Rectangle
import itertools
from scipy.special import comb
import timeit
from mlp import LogisticRegression,HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer


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


def weight_translation(image_tensor_input, horizon_disp, verti_disp):
    # This is only working for kernels of 5 pixel dimension
    tx = image_tensor_input

    def vertical_shift(image_input, displacement):
        tx_out = numpy.zeros(image_input.shape, dtype=float)
        if displacement < 0:
            tx_out[:, :, 0:5 + displacement, :] = image_input[:, :, -displacement:5, :]     # Matrix to the bottom
        elif displacement > 0:
            tx_out[:, :, displacement:5, :] = image_input[:, :,  0:5 - displacement, :]     # Matrix to the top
        else:
            tx_out = image_input
        return tx_out

    def horizontal_shift(image_input, displacement):
        tx_out = numpy.zeros(image_input.shape, dtype=float)
        if displacement < 0:
            tx_out[:, :, :, 0:5+displacement] = image_input[:, :, :, -displacement:5]    # Matrix to the left
        elif displacement > 0:
            tx_out[:, :, :, displacement:5] = image_input[:, :, :, 0:5-displacement]   # Matrix to the right
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


def weight_translation_2d(image_tensor_input, horizon_disp, verti_disp):
    # This will work for any kernel dimension
    tx = image_tensor_input

    def vertical_shift(image_input, displacement):
        tx_out = numpy.zeros(image_input.shape, dtype=float)
        if displacement > 0:
            tx_out[0: image_input.shape[0] - displacement, :] = image_input[displacement: image_input.shape[0], :]
            #   To the bottom
        elif displacement < 0:
            tx_out[-displacement:image_input.shape[0], :] = image_input[0:image_input.shape[0] + displacement, :]
            #   To the top
        else:
            tx_out = image_input
        return tx_out

    def horizontal_shift(image_input, displacement):
        tx_out = numpy.zeros(image_input.shape, dtype=float)
        if displacement > 0:
            tx_out[:, 0:image_input.shape[1]-displacement] = image_input[:, displacement:image_input.shape[1]]
            # To the left
        elif displacement < 0:
            tx_out[:, -displacement:image_input.shape[1]] = image_input[:, 0:image_input.shape[1]+displacement]
            # To the right
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


def weight_theano_translation_2d(image_tensor_input, horizon_disp, verti_disp, borrow=True):

    tx = image_tensor_input
    image_size = image_tensor_input.shape

    def vertical_shift(image_input, displacement, borrow=True):
        temp1 = numpy.zeros(image_size, dtype=theano.config.floatX)
        tx_out = theano.shared(temp1, borrow=True)
        if displacement > 0:
            tx_out = T.set_subtensor(tx_out[0: image_input.shape[0] - displacement, :],
                                     image_input[displacement: image_input.shape[0], :])
        elif displacement < 0:
            tx_out = T.set_subtensor(tx_out[-displacement:image_input.shape[0], :],
                                     image_input[0:image_input.shape[0] + displacement, :])
        else:
            tx_out = image_input
        return tx_out

    def horizontal_shift(image_input, displacement, borrow=True):
        temp1 = numpy.zeros(image_size, dtype=theano.config.floatX)
        tx_out = theano.shared(temp1, borrow=True)
        if displacement > 0:
            tx_out = T.set_subtensor(tx_out[:, 0:image_input.shape[1]-displacement],
                                     image_input[:, displacement:image_input.shape[1]])
        elif displacement < 0:
            tx_out = T.set_subtensor(tx_out[:, -displacement:image_input.shape[1]],
                                     image_input[:, 0:image_input.shape[1]+displacement])
        else:
            tx_out = image_input
        return tx_out

    if verti_disp != 0 and horizon_disp == 0:
        txout = vertical_shift(tx, verti_disp, borrow=True)

    if horizon_disp != 0 and verti_disp == 0:
        txout = horizontal_shift(tx, horizon_disp, borrow=True)

    if horizon_disp != 0 and verti_disp != 0:
        txout = vertical_shift(tx, verti_disp, borrow=True)
        txout = horizontal_shift(txout, horizon_disp,  borrow=True)

    if verti_disp == 0 and horizon_disp == 0:
        txout = tx

    return txout


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

        if page_number <= 1:
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
                    ax.set_title(str(number_of_filter + 1) + ';' + str(r_z + 1) , fontsize=10)
                    plt.axis('off')

            plt.savefig(filename + str(page_number) + 'filters.jpg')


def weight_write_filter_graph_update(model_name):

    def round_significant(x):
        return numpy.round(x, -(int(numpy.floor(numpy.log10(numpy.abs(x))))-2))

    def boundary(sub1):
        autoAxis = sub1.axis()
        rec = Rectangle((autoAxis[0] - 0.6, autoAxis[2] - 0.4), (autoAxis[1] - autoAxis[0]) + 1,
                        (autoAxis[3] - autoAxis[2]) + 1, fill=False, lw=2, edgecolor='r')
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
    initial_filename = 'Mnist_0.05_[20, 50]pt_Initial.pkl'#model_name + '_Initial.pkl'
    Weight_3_i, Weight_2_i, Weight_1_i, Weight_0_i = Weight_Open(initial_filename)

    for page_number in range(0, 2, 1):
        page_name = 'Weight_' + str(page_number)
        page_name = eval(page_name)
        initial_page_name = 'Weight_' + str(page_number) + '_i'
        initial_page_name = eval(initial_page_name)

        cross_value_temp = cross_corre_value(page_name[0], initial_page_name[0])
        cross_value = cross_value_temp >0.8

        print cross_value_temp

        filter_number = page_name[0].shape[0]
        filter_z = page_name[0].shape[1]
        filter_x = page_name[0].shape[2]
        filter_y = page_name[0].shape[3]
        fig1 = plt.figure()

        if page_number == 0:
            #   Only write the first slice of 20.

            for number_of_filter in range(0, filter_number):
                print('Layer:' + str(page_number) + ' ; weight_write_filter_graph:' + str(number_of_filter))
                temp = plt.subplot(5, 4, number_of_filter + 1)
                #fig, temp = plt.subplots(5, 4)
                temp.pcolor(page_name[0][number_of_filter, 0, :, :])
                temp_mean = round_significant(numpy.mean(page_name[0][number_of_filter, 0, :, :]))
                temp.text(0, 0.05, temp_mean,
                          horizontalalignment='left',
                          verticalalignment='baseline',
                          rotation='horizontal',
                          transform=temp.transAxes,
                          color='white',
                          fontsize=10)
                #temp.set_title(temp_mean,horizontalalignment='center', verticalalignment='baseline', fontsize=10)
                if cross_value[number_of_filter] == 1:
                    boundary(temp)
                #, vmin=-0.7, vmax=0.7)
                temp.axis('off')
                #plt.tight_layout()
            #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            #cax = plt.axes([0.825, 0.1, 0.075, 0.8])
            #plt.colorbar(cax=cax)

            plt.savefig(filename + str(page_number) + 'filters.jpg')

        if page_number == 1:

            #   Writting all 50*20 filters for layer 1
            fig = plt.gcf()
            DPI = fig.get_dpi()
            fig.set_size_inches(1500 / float(DPI), 2500 / float(DPI))

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
                    ax.text(0, 0.05, str(number_of_filter + 1) + ';' + str(r_z + 1),
                            horizontalalignment='left',
                            verticalalignment='baseline',
                            rotation='horizontal',
                            transform=ax.transAxes,
                            color='white',
                            fontsize=10)
                    #ax.set_title(str(number_of_filter + 1) + ';' + str(r_z + 1), fontsize=10)
                    plt.axis('off')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.03)
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
            # Preferred Pseudo Auto Correlation
            denominator = float(1/filter_y/filter_x)

            # For overlapping only
            # denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))

            # For overlapping and overall stacking
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
    plt.savefig(filename + str(page_number) + 'filters_Weight_Correlation.jpg')


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
            # Preferred ( Psuedo Auto Correlation )
            denominator = float(1/filter_y/filter_x)

            # Preferred (Could be applicable for Layer 1)
            # denominator = float(1 / filter_y / filter_x / filter_z)

            # Cross Correlation
            # denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))
            #   denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x))/filter_z)
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


def weight_4d_auto_correlation_filter(filename, page_name, page_number):

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
            page_name_temp = page_name*(temp_weight != 0)

            denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))

            weight_mean = numpy.sum(numpy.sum(numpy.sum(page_name_temp, axis=3), axis=2), axis=1) * denominator

            temp_weight_mean = numpy.sum(numpy.sum(numpy.sum(temp_weight, axis=3), axis=2), axis=1) * denominator

            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(page_name_temp ** 2, axis=3), axis=2), axis=1) *
                                            denominator - weight_mean ** 2))

            sd2 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(numpy.sum(temp_weight ** 2, axis=3), axis=2), axis=1) *
                                            denominator - temp_weight_mean ** 2))

            sd = sd1*sd2

            temp_r_value = (numpy.sum(numpy.sum(numpy.sum(temp_weight * page_name_temp, axis=3), axis=2), axis=1)
                            * denominator - weight_mean * temp_weight_mean) / sd

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
    plt.savefig(filename + str(page_number) + 'filters_Weight_auto_Correlation.jpg')


def weight_4d_auto_correlation_filter_z_dim(filename, page_name, page_number):

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
            page_name_temp = page_name*(temp_weight != 0)
            denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))

            weight_mean = numpy.sum(numpy.sum(page_name_temp, axis=3), axis=2) * denominator

            temp_weight_mean = numpy.sum(numpy.sum(temp_weight, axis=3), axis=2) * denominator

            sd1 = numpy.sqrt(numpy.absolute(numpy.sum(numpy.sum(page_name_temp ** 2, axis=3), axis=2) *
                                            denominator - weight_mean ** 2))

            sd2 = numpy.sqrt(
                numpy.absolute(numpy.sum(numpy.sum(temp_weight ** 2, axis=3), axis=2) *
                               denominator - temp_weight_mean ** 2))

            sd = sd1 * sd2

            temp_r_value = (numpy.sum(numpy.sum(temp_weight * page_name_temp, axis=3), axis=2)
                            * denominator - weight_mean * temp_weight_mean) / sd

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
    plt.savefig(filename + str(page_number) + 'Weight_filters_auto_Correlation.jpg')


def weight_4d_overlap_filter_z_dim(filename, page_name, page_number):

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
            denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))
            temp_r_value = numpy.sum(numpy.sum(temp_weight * page_name, axis=3), axis=2) * denominator

            r_value[:, :, r_y + filter_y - 1, r_x + filter_x - 1] = temp_r_value[:, :]

    fig = plt.gcf()
    #DPI = fig.get_dpi()
    DPI = 96
    fig.set_size_inches(2000 / float(DPI), 5000 / float(DPI))
    tx1 = numpy.max(r_value)
    tx2 = numpy.min(r_value)

    for number_of_filter in range(0, filter_number):
        for r_z in range(0, filter_z, 1):
            #ax = plt.subplot2grid((filter_number, filter_z), (number_of_filter, r_z))
            #ax.pcolor(r_value[number_of_filter, r_z, :, :], vmin=tx2, vmax=tx1)
            fig = plt.subplot(filter_number, filter_z, filter_z * number_of_filter + r_z + 1)
            plt.pcolor(r_value[number_of_filter, r_z, :, :], vmin=tx2, vmax=tx1)
            plt.title(str(number_of_filter+1)+';'+str(r_z+1))
            plt.axis('off')

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.825, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax, ax=fig)
    #   plt.colorbar()
    plt.savefig(filename + str(page_number) + 'filters_Weight_Correlation.jpg')


def weight_4d_overlap_filter(filename, page_name, page_number):

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
            denominator = float(1 / (filter_y - numpy.absolute(r_y)) / (filter_x - numpy.absolute(r_x)))
            temp_r_value = numpy.sum(numpy.sum(numpy.sum(temp_weight * page_name, axis=3), axis=2), axis=1) * denominator

            r_value[:, r_y + filter_y - 1, r_x + filter_x - 1] = temp_r_value[:]

    tx1 = numpy.max(r_value)
    tx2 = numpy.min(r_value)

    for number_of_filter in range(0, filter_number):
        fig = plt.subplot(5, filter_number / 5, number_of_filter + 1)
        plt.pcolor(r_value[number_of_filter, :, :], vmin=tx2, vmax=tx1)
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
        print('Layer:' + str(page_number) + 'rx:' + str(r_y))
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


def normalized_cross_correlation_2(filename_1,filename_2):
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
        if page_number == 0:
            weight_4d_auto_correlation_filter(filename, page_name, page_number)
        if page_number == 1:
            #weight_4d_correlation_filter(filename, page_name, page_number)
            #weight_4d_correlation_all(filename, page_name, page_number)
            weight_4d_auto_correlation_filter_z_dim(filename, page_name, page_number)

        if page_number >= 2:
            pass


def turn_off_filter(model_name, kernels_switch, kernum):

    batch_size = 500

    dataset = 'mnist.pkl.gz'
    datasets = loaddata_mnist(dataset)
    test_set_x, test_set_y = datasets[2]
    n_valid = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((-1, 1, 28, 28))

    n_valid_batches = n_valid//batch_size

    y = T.ivector('y')
    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer3 = pickle.load(f)



    if kernels_switch:
        temp_w = layer0.W.get_value()
        temp_w[[kernum], :, :, :] = 0
        layer0.W.set_value(temp_w, borrow=True)

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: test_set_x[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    validation_losses = [validate_model(i) for i in range(n_valid_batches)]
    this_validation_loss = numpy.mean(validation_losses)*100
    print(this_validation_loss)


def turn_off_combinatorial_filter(model_name, nnumber):
    batch_size = 500

    dataset = 'mnist.pkl.gz'
    datasets = loaddata_mnist(dataset)
    test_set_x, test_set_y = datasets[2]
    n_valid = test_set_x.get_value(borrow=True).shape[0]

    test_set_x = test_set_x.reshape((-1, 1, 28, 28))

    n_valid_batches = n_valid // batch_size

    y = T.ivector('y')
    index = T.lscalar()

    range_list = itertools.combinations(numpy.arange(0,20,1), nnumber)
    value = numpy.zeros(comb(20,nnumber).astype(int))
    iter_index = 0

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer3 = pickle.load(f)

    with open(model_name, 'rb') as f:
        temp_layer0, temp_layer1, temp_layer2_input, temp_layer2, temp_layer3 = pickle.load(f)
        #   layer0, layer1, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: test_set_x[index * 500: (index + 1) * 500],
            y: test_set_y[index * 500: (index + 1) * 500]})

    start_time = timeit.default_timer()

    for x in range_list:
        temp_w = temp_layer0.W.get_value()
        temp_w[[x], :, :, :] = 0
        layer0.W.set_value(temp_w, borrow=True)
        validation_losses = [validate_model(i) for i in range(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses) * 100
        value[iter_index] = this_validation_loss
        iter_index += 1
        if iter_index%1000==0:
            print iter_index
            print(this_validation_loss)
    end_time = timeit.default_timer()
    scipy.io.savemat('PT_error20_'+str(nnumber), mdict={'error': value})
    print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == "__main__":
    #run_normailzed_cross_corre()
    #normalized_cross_correlation('Con_MLP_Train_Rand_Trans_relu_best.pkl', 'Con_MLP_Train_Rand_Rotate.pkl')
    #model_loaded = 'Con_MLP_Train_Rand_Trans_tanh_best.pkl'
    #model_loaded = 'Con_MLP_Train_Rand_Trans_relu_best.pkl'
    #   weight_correlation(model_loaded)
    #  Weight_Write(model_loaded)
    #weight_write_filter_graph(model_loaded)
    #model_loaded = 'Con_MLP_3.pkl'
    #model_loaded = 'Con_MLP_Train_Rand_Trans.pkl'
    #model_loaded = 'Con_MLP_Train_Trans_random_After.pkl'
    #for number in range(-4, 5, 2):
    #model_loaded = 'Con_MLP_Train_Trans_0_'+str(number)+'.pkl'
    #model_loaded = 'Con_MLP_Train_Trans_0_0.pkl'
    #model_loaded = 'Con_MLP_relu_decay.pkl'
    #weight_write_filter_graph( 'Con_MLP_Train_Rand_Rotate.pkl')
    #  weight_write_filter_graph(model_loaded)
    #weight_write_m_file(model_loaded)
    #   Weight_Write(model_loaded)
    #weight_correlation(model_loaded)
    #weight_write_filter_graph_update('Mnist_0.05_0.001_[20, 50]Rand_Trans_Relu2_Begin_Full_2')
    #name1 = 'Mnist_0.05_0.001_[20, 50]Rand_Trans_Relu2_Begin_Full_1'
    #weight_write_filter_graph_update(name1)
    name1 = 'Mnist_0.05_0.001_[20, 50]Rand_Trans_Relu2_Begin_Full_1'
    name11 = 'Mnist_0.05_0.001_[20, 50]Rand_Trans_PT_Full_1'
    name12 = 'Mnist_0.05_[20, 50]pt_wd'
    name13 = 'Mnist_0.05_[20, 50]pt'
    #for ker in range(0,20,1):
    #ker=numpy.arange(0,6,1)
    #turn_off_filter(name1+'.pkl', True, kernum=ker)
    #weight_write_filter_graph_update(name12)
    for off_kernel in range(0,21,1):
        print('Prediction on combination of '+str(off_kernel)+' kernels.')
        turn_off_combinatorial_filter(name13+'.pkl', off_kernel)
