
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import sklearn
import csv
import theano.tensor as tt
import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons

import seaborn as sns
sns.set()
data = []

with open('article-disease-pred-with-kcde/data-raw/San_Juan_Training_Data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(int(row['total_cases']))
        # if int(row['season_week']) in data.keys():
        #     data[int(row['season_week'])] += int(row['total_cases'])
        # else: 
        #     data[int(row['season_week'])] = int(row['total_cases'])

y = data[:500]


x = y[:len(y)-1]
y = y[1:]
size = len(x)

X = np.array(x,dtype=np.float64).reshape((len(x),1))/np.max(x)
Y = np.array(y,dtype=np.float64).reshape((len(y),1))/np.max(y)



import numpy as np
import matplotlib.pyplot as pl

# Test data
n = 499 
Xtest = X#np.linspace(-5, 5, n).reshape(-1,1)

# Define the kernel function
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

param = 0.1
K_ss = kernel(Xtest, Xtest, param)

# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,3)))

# Now let's plot the 3 sampled functions.
pl.plot(Xtest, f_prior)
pl.axis([-5, 5, -3, 3])
pl.title('Three samples from the GP prior')
pl.show()



# Noiseless training data
Xtrain = X#np.array([-4, -3, -2, -1, 1]).reshape(5,1)
ytrain = Y#np.sin(Xtrain)

# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)


# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3000)))

pl.plot(Xtrain, ytrain, 'bs', ms=8)
pl.plot(Xtest, f_post)
pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.axis([0, 1, 0, 1])
pl.title('Three samples from the GP posterior')
pl.show()
# print (X.ndim)
# import GPy, numpy as np

# t_distribution = GPy.likelihoods.StudentT(deg_free=5.0, sigma2=2.0)
# laplace = GPy.inference.latent_function_inference.Laplace()
# error = np.random.normal(0,.2,X.size)


# kern = GPy.kern.MLP(1) + GPy.kern.Bias(1)
# m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)
# m.het_Gauss.variance = .001
# m.het_Gauss.variance.fix()
# m.optimize()

# m.plot_f() #Show the predictive values of the GP.
# pb.errorbar(X,Y,yerr=np.array(m.likelihood.flattened_parameters).flatten(),fmt=None,ecolor='r',zorder=1)
# pb.grid()
# pb.plot(X,Y,'kx',mew=1.5)
# plt.show()

# # X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
# # X = scale(X)
# # X = X.astype(floatX)
# # Y = Y.astype(floatX)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)


# fig, ax = plt.subplots()
# ax.scatter(X, Y, label='Class 0')
# sns.despine(); ax.legend()
# ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');

# sd_ = 100
# def construct_nn(ann_input, ann_output):
#     n_hidden = 10

#     # Initialize random weights between each layer
#     init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
#     init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
#     init_out = np.random.randn(n_hidden).astype(floatX)

#     with pm.Model() as neural_network:
#         # Weights from input to hidden layer
#         weights_in_1 = pm.Normal('w_in_1', 0, sd=sd_,
#                                  shape=(X.shape[1], n_hidden),
#                                  testval=init_1)

#         # Weights from 1st to 2nd layer
#         weights_1_2 = pm.Normal('w_1_2', 0, sd=sd_,
#                                 shape=(n_hidden, n_hidden),
#                                 testval=init_2)

#         # Weights from hidden layer to output
#         weights_2_out = pm.Normal('w_2_out', 0, sd=sd_,
#                                   shape=(n_hidden,),
#                                   testval=init_out)

#         # Build neural-network using tanh activation function
#         act_1 = pm.math.tanh(pm.math.dot(ann_input,
#                                          weights_in_1))
#         act_2 = pm.math.tanh(pm.math.dot(act_1,
#                                          weights_1_2))
#         act_out = pm.math.sigmoid(pm.math.dot(act_2,
#                                               weights_2_out))

#         variance_ = pm.InverseGamma('gam',1,sd_)
#         # Binary classification -> Bernoulli likelihood
#         out = pm.Normal('out',
#                            mu=act_out,sd=variance_,
#                            observed=ann_output
#                            #total_size=Y_train.shape[0] # IMPORTANT for minibatches
#                           )
#     return neural_network

# # Trick: Turn inputs and outputs into shared variables.
# # It's still the same thing, but we can later change the values of the shared variable
# # (to switch in the test-data later) and pymc3 will just use the new data.
# # Kind-of like a pointer we can redirect.
# # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
# ann_input = theano.shared(X_train)
# ann_output = theano.shared(Y_train)
# neural_network = construct_nn(ann_input, ann_output)



# with neural_network:
#     s = theano.shared(pm.floatX(1))
#     inference = pm.ADVI(cost_part_grad_scale=s)
#     # ADVI has nearly converged
#     pm.fit(n=20000, method=inference)
#     # It is time to set `s` to zero
#     s.set_value(0)
#     approx = pm.fit(n=30000)

# trace = approx.sample(draws=5000)
# plt.plot(-inference.hist)
# plt.ylabel('ELBO')
# plt.xlabel('iteration')


# # create symbolic input
# x = T.matrix('X')
# # symbolic number of samples is supported, we build vectorized posterior on the fly
# n = T.iscalar('n')
# # Do not forget test_values or set theano.config.compute_test_value = 'off'
# x.tag.test_value = np.empty_like(X_train[:10])
# n.tag.test_value = 100
# _sample_proba = approx.sample_node(neural_network.out.distribution.mu, size=n,
#                                    more_replacements={ann_input:x})

# sample_proba = theano.function([x, n], _sample_proba)
# pred = sample_proba(X_test, 1000).mean(0)

# print (pred)
# print (Y_test)
# fig, ax = plt.subplots()
# ax.scatter(X_test, pred)
# ax.scatter(X_test,Y_test)
# ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y');
# plt.show()
