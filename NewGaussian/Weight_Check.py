from __future__ import division
import theano.tensor as T
import theano
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import numpy
import xlsxwriter
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.io
from Gaussian_Process_Model import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, ConvPoolLayer_NoMaxPool,Generate_Set_ez, Generate_Set, Generate_Test_Set, Generate_Test_Set_ez, Generate_Set_ez_fixed_seq



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


def Weight_Open3(model_file):

    with open(model_file, 'rb') as f:
        layer0, layer1_input, layer1, layer2 = pickle.load(f)

    layer_0_weight_matrix = numpy.array(layer0.W.eval())
    layer_0_b_value = numpy.array(layer0.b.eval())

    layer_1_weight_matrix = numpy.array(layer1.W.eval())
    layer_1_b_value = numpy.array(layer1.b.eval())

    layer_2_weight_matrix = numpy.array(layer2.W.eval())
    layer_2_b_value = numpy.array(layer2.b.eval())

    rval = [(layer_2_weight_matrix, layer_2_b_value),
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
    initial_filename = model_name + '_Initial.pkl'
    Weight_3_i, Weight_2_i, Weight_1_i, Weight_0_i = Weight_Open(initial_filename)

    for page_number in range(0, 2, 1):
        page_name = 'Weight_' + str(page_number)
        page_name = eval(page_name)
        initial_page_name = 'Weight_' + str(page_number) + '_i'
        initial_page_name = eval(initial_page_name)

        cross_value_temp = cross_corre_value(page_name[0], initial_page_name[0])
        cross_value = cross_value_temp >= 0.8
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


def weight_write_filter_graph_update_2(model_name):

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
                    plt.axis('off')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.03)
            plt.savefig(filename + str(page_number) + 'filters.jpg')


def weight_write_filter_graph_update_side_mag(model_name):
    # Put the magnitude information on a check on the side
    # Change to sign mean square in stead of mean along
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
            temp_value = numpy.zeros((filter_number, filter_z))
            for number_of_filter in range(0, filter_number):
                print('Layer:' + str(page_number) + ' ; weight_write_filter_graph:' + str(number_of_filter))
                temp = plt.subplot(5, 4, number_of_filter + 1)
                #fig, temp = plt.subplots(5, 4)
                temp.pcolor(page_name[0][number_of_filter, 0, :, :])
                temp_mean = round_significant(numpy.sqrt(numpy.mean((page_name[0][number_of_filter, 0, :, :])**2)))
                temp_value[number_of_filter]=temp_mean

                if cross_value[number_of_filter] == 1:
                    boundary(temp)
                #, vmin=-0.7, vmax=0.7)
                temp.axis('off')

            #temp= plt.subplot(5, 5, 21)
            #temp.pcolor(temp_value.reshape(5,4))

            scipy.io.savemat('test.mat', mdict={'r_value': temp_value})

            plt.savefig(filename + str(page_number) + 'filters.jpg')


        if page_number == 1:

            #   Writting all 50*20 filters for layer 1
            fig = plt.gcf()
            DPI = fig.get_dpi()
            fig.set_size_inches(1500 / float(DPI), 2500 / float(DPI))
            temp_value = numpy.zeros((filter_number, filter_z))

            for number_of_filter in range(0, filter_number):
                print('Layer:' + str(page_number) + ' ; weight_write_filter_graph:' + str(number_of_filter))
                for r_z in range(0, filter_z, 1):
                    r_m_s = round(
                        numpy.sqrt(numpy.sum(page_name[0][number_of_filter, r_z, :, :] ** 2) / filter_x / filter_y)
                        * 100, 1)
                    ax = plt.subplot2grid((filter_number, filter_z), (number_of_filter, r_z))
                    ax.pcolor(page_name[0][number_of_filter, r_z, :, :])

                    temp_mean =numpy.sqrt(numpy.mean((page_name[0][number_of_filter, 0, :, :]) ** 2))
                    temp_value[number_of_filter, r_z] = temp_mean

                    if cross_value[number_of_filter, r_z] == 1:
                        boundary(ax)

                    plt.axis('off')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.03)
            plt.savefig(filename + str(page_number) + 'filters.jpg')
            scipy.io.savemat('test2.mat', mdict={'r_value': temp_value})


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

    return r_value


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
        if page_number == 0:
            weight_4d_correlation_filter(filename, page_name, page_number)
        if page_number == 1:
            #weight_4d_correlation_filter(filename, page_name, page_number)
            #weight_4d_correlation_all(filename, page_name, page_number)
            weight_4d_correlation_filter_z_dim(filename, page_name, page_number)

        if page_number >= 2:
            pass


def weight_correlation_all(model_name):

    #  Plot the overall Correlation
    filename = model_name
    Weight_3, Weight_2, Weight_1, Weight_0 = Weight_Open(filename)
    my_dpi = 96
    plt.figure(figsize=(12, 5), dpi=my_dpi)
    for page_number in range(0, 2, 1):
        page_name = 'Weight_' + str(page_number)
        page_name = eval(page_name)
        page_name = page_name[0]
        # [0] is loaded as the weight
        # [1] is loaded as the bias
        #   print(page_name.shape)
        print('Layer Number :' + str(page_number))
        if page_number <= 1:
            r_value = weight_4d_correlation_all(filename, page_name, page_number)
        plt.subplot(1, 2, page_number + 1)
        plt.pcolor(r_value, vmin=-1, vmax=1, cmap='Greys')
        plt.title('Layer' + str(page_number))
        #    , vmin=-1, vmax=1)
        plt.colorbar()
    plt.savefig(filename + 'Weight_Correlation.jpg')


def test_and_compare(model_name):

    batch_size = 500
    datasets = numpy.load('Gaussian_Data_Set.npy')
    total_size = 10000
    valid_set_x, valid_set_y = Generate_Set(datasets, total_size)

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 28, 28))
    n_valid_batches = n_valid//batch_size

    y = T.fmatrix('y')
    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        [layer3.output],
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    validate_error = theano.function(
        [index],
        [layer3.errors(y)],
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    validate_label = [validate_model(i) for i in range(n_valid_batches)]
    validate_label = numpy.asarray(validate_label)
    validate_label = validate_label.reshape(total_size, 10)
    error = numpy.mean([validate_error(i) for i in range(n_valid_batches)])
    print error
    valid_set_y = valid_set_y.eval()
    #scipy.io.savemat(model_name+'_Ground_Truth.mat', mdict={'Ground_Truth': valid_set_y})
    #scipy.io.savemat(model_name+'_Predicted.mat', mdict={'Predicted': validate_label})


def turn_off_filter(model_name, kernels_switch, kernum):

    batch_size = 500
    total_size = 10000

    datasets = numpy.load('Gaussian_Data_set.npy')
    test_set_x, test_set_y = Generate_Test_Set_ez(datasets, total_size)

    n_valid = test_set_x.get_value(borrow=True).shape[0]

    n_valid_batches = n_valid//batch_size
    test_set_x = test_set_x.reshape((n_valid, 1, 28, 28))
    y = T.fmatrix('y')
    index = T.lscalar()
    dummy = T.ftensor4('dummy')
    ydummy = T.fmatrix('ydummy')

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)
        #   layer0, layer1, layer3 = pickle.load(f)

    if kernels_switch:
        #temp_w[[0, 1, 3, 4, 6, 7, 8, 9, ], :, :, :] = 0
        #temp_w[[2, 4, 5, 10, 13, 14, 19], :, :, :] = 0
        temp_w = layer0.W.get_value()
        temp_w[[kernum], :, :, :] = 0
        # temp_w[:,:,:,:]=0
        layer0.W.set_value(temp_w, borrow=True)

    temp_test_set_y = theano.shared(numpy.zeros(test_set_y.shape.eval(), dtype=theano.config.floatX), borrow=True)

    temp_test_set_x = theano.shared(numpy.zeros(test_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_test_set_xx = T.Rebroadcast((1, True))(temp_test_set_x)

    update = [(temp_test_set_x, dummy),(temp_test_set_y, ydummy)]
    copier = theano.function([dummy, ydummy], [temp_test_set_x, temp_test_set_y] , updates=update)

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: temp_test_set_xx[index * 500: (index + 1) * 500],
            y: temp_test_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    #copier(tran_test_set)
    for x in range(0, 10, 1):
        test_set_x, test_set_y = Generate_Test_Set_ez(datasets, total_size)

        test_set_x = test_set_x.reshape((n_valid, 1, 28, 28)).eval()
        test_set_y = test_set_y.get_value()

        copier(test_set_x,test_set_y)
        validation_losses = [validate_model(i) for i in range(n_valid_batches)]
        #print validation_losses
        this_validation_loss = numpy.mean(validation_losses)
        print(this_validation_loss)


    #validate_label = numpy.asarray(validate_label)
    #validate_label = validate_label.reshape(total_size, 10)

    #valid_set_y = valid_set_y.eval()
    #scipy.io.savemat(model_name+'_Ground_Truth_Data_TurnOff_1.mat', mdict={'Ground_Truth': valid_set_y})
    #scipy.io.savemat(model_name+'_Predicted_Data_TurnOff_1.mat', mdict={'Predicted': validate_label})


if __name__ == "__main__":
    name2 = 'Gaussian_Model_0.001_0.001_[20, 30]_adam_weight.pkl'
    name1 = 'Gaussian_Model_0.05_0.001_[20, 30]'
    name3 = 'Gaussian_Model_0.05_0.001_[20, 30]_WN'
    name12 = 'Gaussian_Model_0.05_0.001_[20, 30]seed2'
    #for x in range(0, 20, 1):
    #    turn_off_filter(name2, 'F')

    #weight_write_filter_graph_update(name2)
    #weight_correlation(name1)
    #name3 = 'Gaussian_Model_Initilization.pkl'
    #name4 = 'Gaussian_Model_WN_0.05_Sigmoid.pkl'
    #name5 = 'Gaussian_Model_0.05_Sigmoid.pkl'
    #weight_write_filter_graph(name1)
    #weight_write_filter_graph_update(name1)
    #weight_correlation(name1 + '.pkl')
    #weight_correlation_all(name1+'.pkl')
    #weight_correlation(name1)
    #normalized_cross_correlation(name1, name2)
    #normalized_cross_correlation(name1, name3)
    #normalized_cross_correlation(name2, name3)
    #test_and_compare(name1)
    weight_write_filter_graph_update(name12)
    #for ker in range(0,20,1):
    #    print 'Kernel:'+ str(ker)
    #    turn_off_filter(name12+'.pkl', True ,kernum=ker)
