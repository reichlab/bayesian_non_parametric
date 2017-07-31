import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
# Test data
# Define the kernel function

def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

#plt.ion()
axes = plt.gca()
axes.set_xlim([-5,5])
axes.set_ylim([-5,5])
param =.1
num_sample= 10
for i in range(num_sample):
    n=50

    color = np.random.rand(3,1)
    Xtest = np.linspace(-5, 5, n).reshape(-1,1)
    K_ss = kernel(Xtest, Xtest, param)
    #Cholesky Decomp
    L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))

    #We observe the points [1,1] and [2,2]
    Xtrain = np.array([1]).reshape((-1,1))
    ytrain=Xtrain
    K = kernel(Xtrain, Xtrain, param)
    L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

    # Compute the mean at our test points.
    K_s = kernel(Xtrain, Xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

    # Compute the standard deviation so we can plot it
    print ( type(Lk))
    print (Lk)
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    # Draw samples from the posterior at our test points.
    L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))

    plt.plot(Xtrain, ytrain, 'x', alpha=.5)
    plt.plot(Xtest, f_post)
    plt.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd",alpha=.3)
    plt.plot(Xtest, mu, 'x', alpha=.5)
    plt.show()
plt.show()



