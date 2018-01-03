
import numpy as np
import pylab as pl
from pykalman import UnscentedKalmanFilter
import pandas as pd
from ggplot import *
from matplotlib import pyplot as plt
import sys
from sklearn.metrics import mean_squared_error as mse
import pymc3 as pm
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
import theano.tensor as tt
from theano.tensor.nlinalg import det
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theano import shared
from pymc3.math import logsumexp



X_train = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/tmp_data/X_train.csv", "rb"), delimiter=",", skiprows=0)

y_train = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/tmp_data/y_train.csv", "rb"), delimiter=",", skiprows=0)


X_test = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/tmp_data/X_test.csv", "rb"), delimiter=",", skiprows=0)

X_train = X_train[:10,:]
y_train = y_train[:10]
X_test = X_test[:10,:]

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)

def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))

def stick_breaking(v):
    return v * tt.concatenate([tt.ones_like(v[:, :1]),
                               tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                              axis=1)


#N = len(X_train)
K = len(X_train)
#D = 10
#ntd_vector = raw counts of delays in a matrix of size N x D


def gaussian_kernel_2d(x,y,h):
    mat = h*np.eye(len(X_train))
    precision_mat = tt.nlinalg.matrix_inverse(mat)
    tmp = tt.dot(tt.dot(tt.transpose(x-y),precision_mat),(x-y))
    return 1.0/(tt.sqrt(tt.nlinalg.det(2*np.pi*mat)))*tt.exp(-.5*tmp)



X_train = np.transpose(X_train)

lag_4 = shared(X_train[0].reshape((-1,1)), broadcastable=(False, True))
lag_3 = shared(X_train[1].reshape((-1,1)), broadcastable=(False, True))
lag_2 = shared(X_train[2].reshape((-1,1)), broadcastable=(False, True))
lag_1 = shared(X_train[3].reshape((-1,1)), broadcastable=(False, True))

total_data = shared(X_train)


with pm.Model() as model:
    
    v = gaussian_kernel_2d(total_data,total_data,1)
    w = pm.Deterministic('w', stick_breaking(v))


with model:
    gamma = pm.Normal('gamma', 0., 10., shape=K)
    delta_1 = pm.Normal('delta1', 0., 10., shape=K)
    delta_2 = pm.Normal('delta2', 0., 10., shape=K)
    delta_3 = pm.Normal('delta3', 0., 10., shape=K)
    delta_4 = pm.Normal('delta4', 0., 10., shape=K)
    mu = pm.Deterministic('mu', gamma + delta_1 * lag_1 + delta_2 * lag_2 + delta_3 * lag_3 + delta_4 * lag_4)


with model:
   
    tau = pm.Gamma('tau', 1., 1., shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=y_train.reshape((-1,1)))
    
    #p_vector = pm.Dirichlet('dprior',a=np.ones(D))
    #N_tinf = pm.Poisson('ntinft',lambda=obs)
    #N_td = pm.Multinomial('ntd',n=N_tinf,p=p_vector,observed=ntd_vector)

SAMPLES = 2000
BURN = 1000

with model:
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, tune=BURN, progressbar=False)


PP_SAMPLES = 5000

X_test = np.transpose(X_test)
lag_4.set_value(X_test[0].reshape((-1,1)))
lag_3.set_value(X_test[1].reshape((-1,1)))
lag_2.set_value(X_test[2].reshape((-1,1)))
lag_1.set_value(X_test[3].reshape((-1,1)))

total_data.set_value(X_test)

with model:
    pp_trace = pm.sample_ppc(trace, PP_SAMPLES,progressbar= False)

#print (len(np.mean(pp_trace['obs'],axis=0)))
myList = ','.join(map(str, np.mean(pp_trace['obs'],axis=0)))
print (myList)  
