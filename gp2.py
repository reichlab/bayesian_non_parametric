import numpy as np
import matplotlib.pyplot as pl
import sys
from scipy.spatial.distance import pdist, cdist, squareform
import scipy.stats
# Test data

n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1,1)



f = lambda x, theta: scipy.stats.norm.pdf(x, theta, 0.3)

class my_pdf(scipy.stats.rv_continuous):
    def _pdf(self,x):
        return 3*x**2  # Normalized over its range, in this case [0,1]

# Define the kernel function
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

# def kernel(a, b, param):
#     length_scale = .2
#     periodicity = 1
#     dists = cdist(a, b, metric='euclidean')
#     K = np.exp(- 2 * (np.sin(dists)
#                               / length_scale) ** 2)
#     return K
param = 0.1
K_ss = kernel(Xtest, Xtest, param)
# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix

Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
ytrain = np.sin(Xtrain)

# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.000055*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
f = lambda x: scipy.stats.norm.pdf(x, loc=mu[5],scale=stdv[5])


pl.plot(Xtrain, ytrain, 'bs', ms=8)
#pl.plot(Xtest, f_post)
pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.axis([-5, 5, -3, 3])
pl.title('Three samples from the GP posterior')
#pl.show()


alpha = 2.
#N is number of samples 
#K is maximum number of mixture components
N = 10
K = 2
P0 = scipy.stats.norm

x_plot = np.linspace(-3, 3, 200)

beta = scipy.stats.beta.rvs(1, alpha, size=(N, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)

print (w.shape)
dpm_pdfs = (w[..., np.newaxis] * f(x_plot[np.newaxis, np.newaxis, :])).sum(axis=1)

fig, ax = pl.subplots(figsize=(8, 6))

ax.plot(x_plot, dpm_pdfs.T, c='gray');



fig, ax = pl.subplots(figsize=(8, 6))

ix = 0


# ax.plot(x_plot, dpm_pdfs[ix], c='k', label='Density');
# ax.plot(x_plot, (w[..., np.newaxis] * dpm_pdf_components)[ix, 0],
#         '--', c='k', label='Mixture components (weighted)');
# ax.plot(x_plot, (w[..., np.newaxis] * dpm_pdf_components)[ix].T,
#         '--', c='k');

ax.set_yticklabels([]);
ax.legend(loc=1);
pl.show()

