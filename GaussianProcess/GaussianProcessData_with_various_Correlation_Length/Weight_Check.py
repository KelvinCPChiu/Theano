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
from Gaussian_Process_Model import LogisticRegression, LogisticRegression_2, LeNetConvPoolLayer, ConvPoolLayer_NoMaxPool, HiddenLayer, shared_dataset_2

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
        #print cross_value_temp

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

            #plt.savefig(filename + str(page_number) + 'filters.jpg')

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


def test_and_compare(model_name, x):

    batch_size = 500

    valid_set_x = numpy.load('Gaussian_Valid_Data_Set_2_11.npy')
    valid_set_y = numpy.load('Gaussian_Valid_Label_Set_2_11.npy')
    label = numpy.nonzero(valid_set_y==x)
    valid_set_x = valid_set_x[label]
    valid_set_y = valid_set_y[label]

    valid_set_x, valid_set_y = shared_dataset_2(valid_set_x, valid_set_y)

    n_valid = valid_set_x.get_value(borrow=True).shape[0]

    valid_set_x = valid_set_x.reshape((n_valid, 1, 40, 40))

    n_valid_batches = n_valid//batch_size


    y = T.fmatrix('y')
    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    validate_error = theano.function(
        [index],
        [layer3.errors(y)],
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]},
        on_unused_input='ignore')

    error = numpy.mean([validate_error(i) for i in range(n_valid_batches)])
    print error
    valid_set_y = valid_set_y.eval()

    #scipy.io.savemat(model_name+'_Ground_Truth.mat', mdict={'Ground_Truth': valid_set_y})
    #scipy.io.savemat(model_name+'_Predicted.mat', mdict={'Predicted': validate_label})


def turn_off_filter(model_name, kernels_switch):

    batch_size = 500

    valid_set_x = numpy.load('Gaussian_Test_Data_Set_2_11.npy')
    valid_set_y = numpy.load('Gaussian_Test_Label_Set_2_11.npy')


    valid_set_x, valid_set_y = shared_dataset_2(valid_set_x, valid_set_y)

    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    valid_set_x = valid_set_x.reshape((n_valid, 1, 40, 40))
    n_valid_batches = n_valid//batch_size

    y = T.fmatrix('y')
    index = T.lscalar()

    with open(model_name, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            layer0.input: valid_set_x[index * 500: (index + 1) * 500],
            y: valid_set_y[index * 500: (index + 1) * 500]})

    if kernels_switch:
        temp_w = layer0.W.get_value()
        temp_w[[10], :, :, :] = 0
        #temp_w[:,:,:,:]=0
        layer0.W.set_value(temp_w,borrow=True)

    validation_losses = [validate_model(i) for i in range(n_valid_batches)]
    this_validation_loss = numpy.mean(validation_losses)
    print(this_validation_loss)

    #validate_label = numpy.asarray(validate_label)
    #validate_label = validate_label.reshape(total_size, 10)

    #valid_set_y = valid_set_y.eval()
    #scipy.io.savemat(model_name+'_Ground_Truth_Data_TurnOff_1.mat', mdict={'Ground_Truth': valid_set_y})
    #scipy.io.savemat(model_name+'_Predicted_Data_TurnOff_1.mat', mdict={'Predicted': validate_label})


if __name__ == "__main__":
    #name2 = 'Gaussian_Model_0.05_0.001_[20, 30]_WN.pkl'
    name1 = 'Gaussian_Model_0.05_0.001_[20, 30]'
    #name1 = 'Gaussian_Model_0.05_0.001_[20, 30].pkl'
    #weight_write_filter_graph(name2)
    #weight_write_filter_graph_update(name1)
    #weight_correlation(name1+'.pkl')
    #weight_correlation_all(name1+'.pkl')
    #weight_correlation(name2)
    #normalized_cross_correlation(name1, name2)
    #normalized_cross_correlation(name1, name3)
    #normalized_cross_correlation(name2, name3)
    #for x in range(0,10,1):
    #   test_and_compare(name1+'.pkl',x)
    weight_write_filter_graph_update(name1)
    #turn_off_filter(name1 +'.pkl', kernels_switch=True)