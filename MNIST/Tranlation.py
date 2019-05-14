import os
import numpy
import gzip
import theano.tensor as T
import theano
import six.moves.cPickle as pickle
import timeit
import scipy.io
from mlp import LogisticRegression,HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer


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


def theano_translation_updating(image_tensor_input, horizon_disp, verti_disp):
    tx = image_tensor_input
    def vertical_shift(image_input, displacement):

        txout1 = T.zeros_like(image_input)

        if displacement > 0:
            txout1 = T.set_subtensor(txout1[:, :, 0:27 - displacement, :], image_input[:, :, displacement:27, :])
        elif displacement < 0:
            txout1 = T.set_subtensor(txout1[:, :, -displacement:27, :], image_input[:, :,  0:27 + displacement, :])
        else:
            txout1 = image_input
        return txout1

    def horizontal_shift(image_input, displacement):

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

    txout = txout.eval()
    return txout


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


def translation_prediction(model_file):

    y = T.ivector('y')
    index = T.lscalar()
    dummy = T.ftensor4('dummy')

    dataset = 'mnist.pkl.gz'
    datasets = loaddata_mnist(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.reshape((10000, 1, 28, 28))

    temp_test_set_x = theano.shared(numpy.zeros(test_set_x.shape.eval(), dtype=theano.config.floatX), borrow=True)
    temp_test_set_xx = T.Rebroadcast((1, True))(temp_test_set_x)

    with open(model_file+'.pkl', 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    error_spectrum = numpy.zeros((21, 21))

    #### Copier is used to update the testing set ####
    #### Relace the shared varialbe temp_test_set_x by dummy, then, it will update the variable passing to the predict model ####
    #### As function could only input shared variable, therefore, it is done in this way ####
    update = (temp_test_set_x, dummy)
    copier = theano.function([dummy],temp_test_set_x, updates=[update])

    predict_model = theano.function(inputs=[index],
                                    outputs=layer3.errors(y),
                                    givens={layer0.input: temp_test_set_xx[index * 500: (index + 1) * 500],
                                            y: test_set_y[index * 500: (index + 1) * 500]})

    print('Start Predicting...')
    start_time = timeit.default_timer()
    for horizontal in range(-20, 21, 2):
        temp_time_1 = timeit.default_timer()
        for vertical in range(-20, 21, 2):

            predicted_values = 0

            tran_test_set = theano_translation_updating(test_set_x, horizontal, vertical).reshape((-1, 1, 28, 28))
            copier(tran_test_set)

            #print('Horizontal Shift:' + str(horizontal) + '; Vertical Shift:' + str(vertical))
            for batch_value in range(0, 20, 1):
                temp_predicted_values = predict_model(batch_value)
                predicted_values = temp_predicted_values + predicted_values

            predicted_values = predicted_values/20

            error_spectrum[vertical/2 + 10, horizontal/2 + 10] = predicted_values

        temp_time_2 = timeit.default_timer()
        print 'Horizontal'+str(horizontal)
        print('This loop ran for %.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    scipy.io.savemat(model_file+'error_spectrum.mat', mdict={'Error_Spectrum': error_spectrum})

    return error_spectrum


if __name__ == "__main__":
    path = os.getcwd()
    path = os.path.join(path,'Full')
    for x in range(1, 11, 1):
        name = 'Mnist_0.05_0.001_[20, 50]Rand_Trans_Relu2_Begin_Full_' + str(x)
        path = os.path.join(path,name)
        translation_prediction(model_file=path)