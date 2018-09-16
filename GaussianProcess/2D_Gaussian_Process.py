from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def kernel(x_1, x_2):
    sqdist = np.sum(x_1**2, 1).reshape(-1, 1) + np.sum(x_2**2, 1) - 2*np.dot(x_1, x_2.T)
    return np.exp(-.5*sqdist)

n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
x = np.linspace(0, 28, n)
K_ = kernel(Xtest, Xtest)

L = np.linalg.cholesky(K_+10**(-6)*np.eye(n))

norm = np.random.normal(size=(n, n))
norm2 = np.random.normal(size=(n, n))
f_prior = np.dot(L, np.dot(L, norm).T).T
#f_prior = np.dot(np.dot(L, norm), L)
f_prior = f_prior/np.max(f_prior)
#f_prior_2 = np.dot(np.dot(L, norm2), L)
f_prior_2 = np.dot(L, np.dot(L, norm2).T).T
f_prior_2 = f_prior_2/np.max(f_prior_2)

coeff = 0.5
f_prior_3 = coeff*f_prior + (1-coeff)*f_prior_2
f_prior_3 = f_prior_3/np.max(np.max(f_prior_3))
fig = plt.figure()
plt.subplot(1, 3, 1)
plt.pcolor(x, x, f_prior)
plt.colorbar()
plt.subplot(1, 3, 2)
plt.pcolor(x, x, f_prior_2)
plt.colorbar()
plt.subplot(1, 3, 3)
plt.pcolor(x, x, f_prior_3)
plt.colorbar()
plt.show()
