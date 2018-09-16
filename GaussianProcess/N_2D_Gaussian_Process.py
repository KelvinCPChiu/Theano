from __future__ import division
import numpy 
import matplotlib.pyplot as plt
import scipy.io


def Gaussian_Data():

    def kernel(x_1, x_2):
        sqdist = numpy.sum(x_1 ** 2, 1).reshape(-1, 1) + numpy.sum(x_2 ** 2, 1) - 2 * numpy.dot(x_1, x_2.T)
        return numpy.exp(-.5 * sqdist)

    n = 28
    training_set_sample = 10
    #xtest = numpy.linspace(-14, 14, n).reshape(-1, 1)
    xtest = numpy.linspace(-5, 5, n).reshape(-1, 1)
    gaussian_kernel = kernel(xtest, xtest)

    cholesky = numpy.linalg.cholesky(gaussian_kernel + 10 ** (-6) * numpy.eye(n))

    norm = numpy.random.normal(size=(training_set_sample, n, n))
    f_prior = numpy.tensordot(cholesky, numpy.tensordot(cholesky, norm, axes=(1, 1)), axes=(1, 2))

    f_prior = f_prior + numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)
    f_prior = f_prior / numpy.amax(numpy.amax(numpy.abs(f_prior), axis=0), axis=0)

    f_temp = numpy.zeros([10, 28, 28])
    for i in range(0, 10, 1):
        f_temp[i, :, :] = f_prior[:, :, i]

    numpy.save('Gaussian_Data_Set_Second.npy', f_temp)
    scipy.io.savemat('GaussianPlot_Second.mat', mdict={'Gaussian_Plot': f_prior})
    return f_temp


if __name__ == "__main__":
    Gaussian_Data()
#    x = numpy.load('Gaussian_Data_Set.npy')