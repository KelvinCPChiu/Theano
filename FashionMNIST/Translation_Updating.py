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
import timeit
import scipy.io
from convolutional_mlp import LogisticRegression, LeNetConvPoolLayer, HiddenLayer
from fashion_mnist import load_mnist


def shared_dataset(data_x, data_y, borrow=True):
    # 0-9 Label Representation
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def printimage(test_set_x):
    # Print Image from tensor to numpy and plot it
    mm = numpy.squeeze(test_set_x.eval(), axis=(0,))
    # print(mm)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(mm)  # , cmap='gray')
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


def theano_translation_updating(image_tensor_input, horizon_disp, verti_disp):
    tx = image_tensor_input
    def vertical_shift(image_input, displacement):
        #temp1 = numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX)
        #txout1 = theano.shared(temp1)
        txout1 = T.zeros_like(image_input)
        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, 0:27 - displacement, :], image_input[:, :, displacement:27, :])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, -displacement:27, :], image_input[:, :,  0:27 + displacement, :])
        else:
            txout1 = image_input
        return txout1

    def horizontal_shift(image_input, displacement):
        #temp1 = numpy.zeros((10000, 1, 28, 28), dtype=theano.config.floatX)
        #txout1 = theano.shared(temp1)
        txout1 = T.zeros_like(image_input)
        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, :, displacement:27], image_input[:, :, :, 0:27 -displacement])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, :, 0:27 +displacement], image_input[:, :, :, -displacement:27])
        else:
            txout1 = image_input
        return txout1

    if verti_disp != 0 and horizon_disp == 0:
        txout = vertical_shift(tx, verti_disp)

    if horizon_disp != 0 and verti_disp == 0:
        txout = horizontal_shift(tx, horizon_disp)

    if horizon_disp != 0 and verti_disp != 0:
        txout = vertical_shift(tx, verti_disp)
        txout = horizontal_shift(txout, horizon_disp)

    if verti_disp == 0 and horizon_disp == 0:
        txout = tx

    #txout = T.Rebroadcast((1, True))(txout)
    txout = txout.eval()
    return txout


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

    #txout = T.Rebroadcast((1, True))(txout)

    return txout.eval()

    #if TransNum >= 0:
    #    tempim[:, TransNum:27] = im[:, 0:27 - TransNum]  # Left
    #else:
    #   tempim[:, 0:27 - TransNum] = im[:, TransNum:27]  # Right


def translation_prediction_updating(model_file):

    path = os.getcwd()

    y = T.ivector('y')
    index = T.lscalar()
    dummy = T.ftensor4('dummy')

    test_set_x, test_set_y = load_mnist(path=path, kind='t10k')
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    test_set_x = test_set_x.reshape((10000, 1, 28, 28))

    temp_test_set_x = theano.shared(numpy.zeros(test_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_test_set_xx = T.Rebroadcast((1, True))(temp_test_set_x)

    with open(model_file, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    error_spectrum = numpy.zeros((21, 21))

    print('Start Predicting...')
    start_time = timeit.default_timer()
    #### Copier is used to update the testing set ####
    #### Relace the shared varialbe temp_test_set_x by dummy, then, it will update the variable passing to the predict model ####
    #### As function could only input shared variable, therefore, it is done in this way ####
    update = (temp_test_set_x, dummy)
    copier = theano.function([dummy],temp_test_set_x, updates=[update])

    predict_model = theano.function(inputs=[index],
                                    outputs=layer3.errors(y),
                                    givens={layer0.input: temp_test_set_xx[index * 500: (index + 1) * 500],
                                            y: test_set_y[index * 500: (index + 1) * 500]})

    for horizontal in range(-20, 21, 2):
        temp_time_1 = timeit.default_timer()
        for vertical in range(-20, 21, 2):

            predicted_values = 0

            tran_test_set = theano_translation_updating(test_set_x, horizontal, vertical).reshape((-1,1,28,28))

            copier(tran_test_set)
            #print('Horizontal Shift:' + str(horizontal) + '; Vertical Shift:' + str(vertical))
            for batch_value in range(0, 20, 1):
                temp_predicted_values = predict_model(batch_value)
                predicted_values = temp_predicted_values + predicted_values
            predicted_values = predicted_values/20

            error_spectrum[vertical/2 + 10, horizontal/2 + 10] = predicted_values


        temp_time_2 = timeit.default_timer()
        print 'Horizontal :'+str(horizontal)
        print('This loop ran for %.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    scipy.io.savemat(model_file+'error_spectrum.mat', mdict={'Error_Spectrum': error_spectrum})

    return error_spectrum


if __name__ == "__main__":
    for x in range(1, 11, 1):
        name = 'FashionMnist_0.05_0.001_[20, 30]Rand_Trans_tanh_Begin_Full_' + str(x) + '.pkl'
    #name = 'FashionMnist_0.05_0.001_[20, 30]Rand_Trans_tanh_Begin_' + str(x) +'.pkl'
        translation_prediction_updating(name)
