from __future__ import division
import numpy
import scipy.io


def Gaussian_Data(training_set_sample, desire_dimension, correlation_length):

    def kernel(x_1, x_2, width):
        sqdist = numpy.sum(x_1 ** 2, 1).reshape(-1, 1) + numpy.sum(x_2 ** 2, 1) - 2 * numpy.dot(x_1, x_2.T)
        return numpy.exp(-.5 * sqdist/width**2)

    n = desire_dimension

    if numpy.mod(n, 2) == 1:
        n = n+1

    meshmax = numpy.floor(n/2)
    xtest = numpy.linspace(-meshmax, meshmax, n).reshape(-1, 1)

    gaussian_kernel_x = kernel(xtest, xtest, correlation_length[0])
    gaussian_kernel_y = kernel(xtest, xtest, correlation_length[1])

    cholesky_x = numpy.linalg.cholesky(gaussian_kernel_x + 10 ** (-6) * numpy.eye(n))
    cholesky_y = numpy.linalg.cholesky(gaussian_kernel_y + 10 ** (-6) * numpy.eye(n))

    norm = numpy.random.normal(size=(training_set_sample, n, n))
    f_prior = numpy.tensordot(cholesky_y, numpy.tensordot(cholesky_x, norm, axes=(1, 1)), axes=(1, 2))

    f_prior = f_prior + numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)
    f_prior = f_prior / numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)

    return f_prior


def n_type_gaussian_generator(number_of_sample):
    fig_size = 40
    dataset = numpy.zeros((fig_size, fig_size, number_of_sample))
    aver_size = number_of_sample//10

    for length_dim in range(1, 11, 1):
        temp_set = Gaussian_Data(aver_size, fig_size, [length_dim*2, length_dim*2])
        #temp_set = Gaussian_Data(aver_size, fig_size, [length_dim/5+1, length_dim/5+1])
        dataset[:, :, (length_dim-1)*aver_size:length_dim*aver_size] = temp_set

    label_set_1 = numpy.ones(aver_size)
    label_set = numpy.zeros(aver_size)
    for i in range(1, 10, 1):
        label_set = numpy.concatenate((label_set, label_set_1 * i), axis=0)
    set_order = numpy.arange(0, number_of_sample, 1)
    numpy.random.shuffle(set_order)
    dataset = dataset[:, :, set_order]
    label_set = label_set[set_order]

    f_temp = numpy.zeros([number_of_sample, fig_size, fig_size])
    for i in range(0, number_of_sample, 1):
        f_temp[i, :, :] = dataset[:, :, i]

    #numpy.save('Gaussian_Data_Set_2_20.npy', f_temp)
    #numpy.save('Gaussian_Label_Set_2_20.npy', label_set)
    numpy.save('Gaussian_Valid_Data_Set_2_20.npy', f_temp)
    numpy.save('Gaussian_Valid_Label_Set_2_20.npy', label_set)

    return f_temp, label_set

if __name__ == "__main__":
    n_type_gaussian_generator(10000)
