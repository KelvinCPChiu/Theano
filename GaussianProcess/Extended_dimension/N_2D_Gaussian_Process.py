from __future__ import division
import numpy 
import matplotlib.pyplot as plt
import scipy.io


def Gaussian_Data(desire_dimension, correlation_length):

    def kernel(x_1, x_2, width):
        sqdist = numpy.sum(x_1 ** 2, 1).reshape(-1, 1) + numpy.sum(x_2 ** 2, 1) - 2 * numpy.dot(x_1, x_2.T)
        return numpy.exp(-.5 * sqdist/width**2)

    n = desire_dimension

    if numpy.mod(n, 2) == 0:
        n = n+1

    meshmax = numpy.floor(n/2)
    training_set_sample = 10
    xtest = numpy.linspace(-meshmax, meshmax, n).reshape(-1, 1)
    gaussian_kernel = kernel(xtest, xtest, correlation_length)

    cholesky = numpy.linalg.cholesky(gaussian_kernel + 10 ** (-6) * numpy.eye(n))

    norm = numpy.random.normal(size=(training_set_sample, n, n))
    f_prior = numpy.tensordot(cholesky, numpy.tensordot(cholesky, norm, axes=(1, 1)), axes=(1, 2))

    f_prior = f_prior + numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)
    f_prior = f_prior / numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)

    f_temp = numpy.zeros([10, n, n])
    norm_temp = numpy.zeros([n, n, 10])
    for i in range(0, 10, 1):
        f_temp[i, :, :] = f_prior[:, :, i]
        norm_temp[:, :, i] = norm[i, :, :]
    #   plt.subplot(5, 2, i + 1)
    #    plt.pcolor(f_prior[:, :, i])
    numpy.save('Gaussian_Data_Set_' + str(desire_dimension) + '.npy', f_temp)
    #numpy.save('Gaussian_White_Noise.npy', norm)
    scipy.io.savemat('GaussianPlot_' + str(desire_dimension) + '.mat', mdict={'Gaussian_Plot': f_prior})
    #scipy.io.savemat('GaussianWNPlot.mat', mdict={'Gaussian_PlotWN': norm_temp})
    #plt.show()
    return f_temp


def Gaussian_Range_Data(desire_dimension):

    def kernel(x_1, x_2, width):
        sqdist = numpy.sum(x_1 ** 2, 1).reshape(-1, 1) + numpy.sum(x_2 ** 2, 1) - 2 * numpy.dot(x_1, x_2.T)
        return numpy.exp(-.5 * sqdist/width**2)

    n = desire_dimension

    if numpy.mod(n, 2) == 0:
        n = n+1
    training_set_sample = 10
    meshmax = numpy.floor(n/2)
    xtest = numpy.linspace(-meshmax, meshmax, n).reshape(-1, 1)
    f_prior = numpy.zeros((n, n, training_set_sample))
    order = 0
    correlation_length = numpy.linspace(2, 5, 10)

    for indeices in range(0, 10, 1):
        gaussian_kernel = kernel(xtest, xtest, correlation_length[indeices])

        cholesky = numpy.linalg.cholesky(gaussian_kernel + 10 ** (-6) * numpy.eye(n))
        norm = numpy.random.normal(size=(n, n))
        temp_f_prior = numpy.dot(cholesky, numpy.dot(cholesky, norm).T).T
        #temp_f_prior = temp_f_prior/numpy.max(temp_f_prior)
        f_prior[:, :, order] = temp_f_prior
        order += 1

    f_prior = f_prior + numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)
    f_prior = f_prior / numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)
    f_temp = numpy.zeros([10, n, n])
    for i in range(0, 10, 1):
        f_temp[i, :, :] = f_prior[:, :, i]

    numpy.save('Gaussian_Data_Set_Range.npy', f_temp)
    scipy.io.savemat('GaussianPlot_Range.mat', mdict={'Gaussian_Plot': f_prior})
    return f_temp


if __name__ == "__main__":
#    for x in range(40, 60, 4):
    Gaussian_Data(60, 3)

#    x = numpy.load('Gaussian_Data_Set.npy')
#   Gaussian_Range_Data(36)
