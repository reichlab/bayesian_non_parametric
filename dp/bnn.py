

import numpy as np
import pylab as pl
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
import theano
from pymc3.math import logsumexp
floatX = theano.config.floatX



X_train = \
    np.loadtxt(open("/Users/gcgibson/bayesian_non_parametric/tmp_data/X_train.csv", "rb"), delimiter=",", skiprows=0)

y_train = \
    np.loadtxt(open("/Users/gcgibson//bayesian_non_parametric/tmp_data/y_train.csv", "rb"), delimiter=",", skiprows=0)


X_test = \
    np.loadtxt(open("/Users/gcgibson/bayesian_non_parametric/tmp_data/X_test.csv", "rb"), delimiter=",", skiprows=0)


def construct_nn(ann_input, ann_output):
    n_hidden = 5
    
    # Initialize random weights between each layer
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)
    
    with pm.Model() as neural_network:
        # Weights from input to hidden layer
         weights_in_1 = pm.Normal('w_in_1', 0, sd=1,shape=(X_train.shape[1], n_hidden),testval=init_1)

         # Weights from 1st to 2nd layer
         weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                 shape=(n_hidden, n_hidden),
                                 testval=init_2)
         
         # Weights from hidden layer to output
         weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                   shape=(n_hidden,),
                                   testval=init_out)
         
         # Build neural-network using tanh activation function
         act_1 = pm.math.tanh(pm.math.dot(ann_input,
                                          weights_in_1))
         act_2 = pm.math.tanh(pm.math.dot(act_1,
                                          weights_1_2))
         act_out = pm.math.exp(pm.math.dot(act_2,
                                               weights_2_out))
         
         # Binary classification -> Bernoulli likelihood
         out = pm.Poisson('out',
                            act_out,
                            observed=ann_output,
                            total_size=y_train.shape[0] # IMPORTANT for minibatches
                            )
    return neural_network

# Trick: Turn inputs and outputs into shared variables.
# It's still the same thing, but we can later change the values of the shared variable
# (to switch in the test-data later) and pymc3 will just use the new data.
# Kind-of like a pointer we can redirect.
# For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
ann_input = theano.shared(X_train)
ann_output = theano.shared(y_train)
neural_network = construct_nn(ann_input, ann_output)


from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))


with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(n=30000, method=inference,progressbar=False)

trace = approx.sample(draws=5000)
pm.traceplot(trace);

ann_input.set_value(X_test)
with neural_network:
    ppc = pm.sample_ppc(trace, samples=500, progressbar=False)
    
    # Use probability of > 0.5 to assume prediction of class 1
pred = ppc['out'].mean(axis=0)





myList = ','.join(map(str, pred))
print (myList)
