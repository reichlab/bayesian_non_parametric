
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
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/X_train.csv", "rb"), delimiter=",", skiprows=0)

print (X_train[0])
sys.exit()
y_train = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/y_train.csv", "rb"), delimiter=",", skiprows=0)


X_test = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/X_test.csv", "rb"), delimiter=",", skiprows=0)

X_train = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
t_train = np.array([1,1,1,1])


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
def gaussian_kernel(x1,x2):
    return tt.exp(-1*tt.sum((x1-x2)**2, axis=2))




X_train = np.transpose(X_train)

lag_4 = shared(X_train[0].reshape((-1,1)), broadcastable=(False, True))
lag_3 = shared(X_train[1].reshape((-1,1)), broadcastable=(False, True))
lag_2 = shared(X_train[2].reshape((-1,1)), broadcastable=(False, True))
lag_1 = shared(X_train[3].reshape((-1,1)), broadcastable=(False, True))

with pm.Model() as model:
    alpha = pm.Normal('alpha', 0., 5., shape=K)
    beta_1 = pm.Normal('beta1', 0., 5., shape=K)
    beta_2 = pm.Normal('beta2', 0., 5., shape=K)
    beta_3 = pm.Normal('beta3', 0., 5., shape=K)
    beta_4 = pm.Normal('beta4', 0., 5., shape=K)
    v = norm_cdf(alpha + beta_1 * lag_1 + beta_2 * lag_2 + beta_3 * lag_3 + beta_4 * lag_4)
    v = gaussian_kernel
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
    w_print = tt.printing.Print('w')(w)
    obs = pm.NormalMixture('obs', w_print, mu, tau=tau, observed=y_train.reshape((-1,1)))
    
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

with model:
    pp_trace = pm.sample_ppc(trace, PP_SAMPLES,progressbar= False)

#print (len(np.mean(pp_trace['obs'],axis=0)))
myList = ','.join(map(str, np.mean(pp_trace['obs'],axis=0)))
print (myList)  
