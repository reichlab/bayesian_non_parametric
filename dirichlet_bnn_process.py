
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import ts_utils
import pymc3 as pm
import scipy as sp
import seaborn as sns
from statsmodels.datasets import get_rdataset
from theano import tensor as tt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
from theano import shared, tensor as tt
import numpy.random as nr
import sys
import theano
floatX = theano.config.floatX
import theano.tensor.slinalg as slinalg
from theano import printing
import csv
size = 99
from sklearn.preprocessing import StandardScaler


train_data = []
test_data = []
count = 0
# with open('article-disease-pred-with-kcde/data-raw/San_Juan_Training_Data.csv') as f:
#         for line in f.readlines():
#             if count <= 781:
#                 line_ = line.split(',')
#                 train_data.append(line_[len(line_)-1])
#             else:
#                 test_data.append(line_[len(line_)-1])
#             count +=1

data__full = [4,5,4,3,6,2,4,5,10,6,8,2,6,17,23,13,21,28,24,20,40,27,42,33,43,37,57,71,44,56,53,52,47,26,27,21,21,26,34,37,17,19,25,18,21,17,17,16,16,15,23,16,17,12,17,10,15,19,21,14,18,13,14,18,23,25,62,60,76,66,64,68,89,92,140,116,142,129,140,140,127,129,169,141,108,78,70,81,104,90,85,55,53,65,33,38,59,40,37,29,30,30,28,23,24,29,26,23,20,19,20,26,29,31,28,26,32,35,33,30,52,59,67,65,74,70,61,53,76,61,57,44,34,47,60,60,53,36,31,30,32,28,33,33,35,22,13,13,21,17,11,8,8,6,6,7,12,17,10,10,18,19,12,22,12,21,18,16,16,22,17,25,23,12,25,28,27,18,23,23,29,38,36,43,46,31,25,40,31,38,30,22,31,26,35,36,39,25,31,37,33,25,24,18,23,13,18,14,17,22,13,24,31,34,31,31,38,49,42,49,55,80,84,72,89,115,179,202,272,302,395,426,461,381,333,353,410,364,359,288,221,149,112,154,91,72,56,46,37,26,17,17,20,11,7,16,14,16,5,2,6,5,4,3,4,16,8,7,10,14,7,9,11,23,17,19,24,17,28,40,33,31,33,29,30,36,48,40,28,36,19,34,23,17,17,23,14,20,13,23,20,16,16,23,14,15,4,5,5,11,11,7,4,6,5,2,4,2,4,6,6,4,6,11,16,9,12,13,27,21,19,17,24,27,30,29,25,35,33,30,29,31,29,22,27,24,26,29,22,33,24,30,20,17,24,28,18,13,9,14,11,11,19,10,8,8,9,3,7,14,4,9,14,7,9,3,3,14,12,10,21,26,47,42,31,34,33,52,56,70,112,70,47,48,49,66,56,61,67,64,68,49,50,56,75,63,62,41,50,34,31,38,30,32,26,30,36,35,46,48,44,51,59,71,102,128,127,150,191,256,329,263,220,204,181,99,54,80,102,127,73,68,64,55,67,84,85,67,73,89,68,59,56,77,75,47,50,42,28,37,37,27,12,15,22,8,15,17,10,9,11,20,13,11,16,11,7,17,14,13,15,30,25,40,44,25,21,48,56,60,45,55,32,46,61,42,37,43,34,40,25,16,17,17,16,23,18,18,9,7,7,4,3,2,8,3,1,1,2,3,3,2,0,0,2,2,0,6,3,6,2,3,2,4,5,2,9,2,4,8,6,3,11,14,15,20,9,20,28,38,30,30,23,16,22,28,14,17,20,17,10,13,20,9,18,9,8,19,11,4,6,6,8,13,8,8,5,16,12,11,18,10,22,14,16,18,27,38,35,41,51,65,55,54,62,64,56,65,71,75,71,72,47,27,35,25,19,37,38,34,26,19,18,22,16,18,6,12,6,6,3,7,6,1,3,2,2,1,10,3,3,1,1,2,6,3,3,5,4,7,6,5,7,6,4,4,7,9,5,5,10,6,13,6,5,5,9,3,6,11,7,7,15,9,6,6,6,7,10,8,7,12,3,2,7,5,5,7,7,7,7,10,13,10,14,11,20,25,17,18,25,21,31,32,26,35,28,37,41,34,30,39,39,39,34,30,37,29,26,15,22,15,20,14,10,21,14,14,9,11,5,6,7,11,4,3,2,6,10,7,5,3,12,13,10,13,13,8,21,18,8,7,20,14,14,7,14,10,13,27,13,18,16,16,20,17,4,15,8,6,12,15,11,10,15,17,7,7,8,9,12,12,5,4,11,4,5,7,1,1,4,2,6,3,4,10,12,21,26,21,30,45,56,75,83,82,126,119,137,131,112,82,73,43,55,55,53,46,43,29,22,26,13,17,8,13,10,17,19,9,9,9,3,7,7,0,2,3,3,1,3,3,3,7,3,5,11,5,5,6,6,4,4,8,14,12,16,10,16,18,15,23,17,33,15,13,11,14,17,19,20,12,21,7,19,10,13,10,8,21,11,9,14,14,15,18,16,12,20,8,3,13,4,1,10,8,13,10,21,18,21,34,25,34,33,40,42,36,72,75,76,92,71,112,106,101,170,135,106,68,48,48,26,33,29,17,12,13,17,15,14,15,10,9,2,6,8,5,1,2,3,4,3,1,3,5,2,3,2,3,2,2,3,4,3,4,4,4,7,6,15,11,9,9,12,13,13,13,20,28,45,28,34,41,36,38,48,27,23,28,42,30,18,38,28,36,44,41,35,28,28,22,26,24,9,21,10,15]

SS = StandardScaler().fit(data__full)

blue = sns.color_palette()
SEED = 972915 # from random.org; for reproducibility
np.random.seed(SEED)

#DATA_URI = 'http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat'

def standardize(x):
    return (x - x.mean()) / x.std()


def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))

def stick_breaking(v):
    return v * tt.concatenate([tt.ones_like(v[:, :1]),
                               tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                              axis=1)



N= size
K = 40
hidden_size = 2

data__full = [4,5,4,3,6,2,4,5,10,6,8,2,6,17,23,13,21,28,24,20,40,27,42,33,43,37,57,71,44,56,53,52,47,26,27,21,21,26,34,37,17,19,25,18,21,17,17,16,16,15,23,16,17,12,17,10,15,19,21,14,18,13,14,18,23,25,62,60,76,66,64,68,89,92,140,116,142,129,140,140,127,129,169,141,108,78,70,81,104,90,85,55,53,65,33,38,59,40,37,29,30,30,28,23,24,29,26,23,20,19,20,26,29,31,28,26,32,35,33,30,52,59,67,65,74,70,61,53,76,61,57,44,34,47,60,60,53,36,31,30,32,28,33,33,35,22,13,13,21,17,11,8,8,6,6,7,12,17,10,10,18,19,12,22,12,21,18,16,16,22,17,25,23,12,25,28,27,18,23,23,29,38,36,43,46,31,25,40,31,38,30,22,31,26,35,36,39,25,31,37,33,25,24,18,23,13,18,14,17,22,13,24,31,34,31,31,38,49,42,49,55,80,84,72,89,115,179,202,272,302,395,426,461,381,333,353,410,364,359,288,221,149,112,154,91,72,56,46,37,26,17,17,20,11,7,16,14,16,5,2,6,5,4,3,4,16,8,7,10,14,7,9,11,23,17,19,24,17,28,40,33,31,33,29,30,36,48,40,28,36,19,34,23,17,17,23,14,20,13,23,20,16,16,23,14,15,4,5,5,11,11,7,4,6,5,2,4,2,4,6,6,4,6,11,16,9,12,13,27,21,19,17,24,27,30,29,25,35,33,30,29,31,29,22,27,24,26,29,22,33,24,30,20,17,24,28,18,13,9,14,11,11,19,10,8,8,9,3,7,14,4,9,14,7,9,3,3,14,12,10,21,26,47,42,31,34,33,52,56,70,112,70,47,48,49,66,56,61,67,64,68,49,50,56,75,63,62,41,50,34,31,38,30,32,26,30,36,35,46,48,44,51,59,71,102,128,127,150,191,256,329,263,220,204,181,99,54,80,102,127,73,68,64,55,67,84,85,67,73,89,68,59,56,77,75,47,50,42,28,37,37,27,12,15,22,8,15,17,10,9,11,20,13,11,16,11,7,17,14,13,15,30,25,40,44,25,21,48,56,60,45,55,32,46,61,42,37,43,34,40,25,16,17,17,16,23,18,18,9,7,7,4,3,2,8,3,1,1,2,3,3,2,0,0,2,2,0,6,3,6,2,3,2,4,5,2,9,2,4,8,6,3,11,14,15,20,9,20,28,38,30,30,23,16,22,28,14,17,20,17,10,13,20,9,18,9,8,19,11,4,6,6,8,13,8,8,5,16,12,11,18,10,22,14,16,18,27,38,35,41,51,65,55,54,62,64,56,65,71,75,71,72,47,27,35,25,19,37,38,34,26,19,18,22,16,18,6,12,6,6,3,7,6,1,3,2,2,1,10,3,3,1,1,2,6,3,3,5,4,7,6,5,7,6,4,4,7,9,5,5,10,6,13,6,5,5,9,3,6,11,7,7,15,9,6,6,6,7,10,8,7,12,3,2,7,5,5,7,7,7,7,10,13,10,14,11,20,25,17,18,25,21,31,32,26,35,28,37,41,34,30,39,39,39,34,30,37,29,26,15,22,15,20,14,10,21,14,14,9,11,5,6,7,11,4,3,2,6,10,7,5,3,12,13,10,13,13,8,21,18,8,7,20,14,14,7,14,10,13,27,13,18,16,16,20,17,4,15,8,6,12,15,11,10,15,17,7,7,8,9,12,12,5,4,11,4,5,7,1,1,4,2,6,3,4,10,12,21,26,21,30,45,56,75,83,82,126,119,137,131,112,82,73,43,55,55,53,46,43,29,22,26,13,17,8,13,10,17,19,9,9,9,3,7,7,0,2,3,3,1,3,3,3,7,3,5,11,5,5,6,6,4,4,8,14,12,16,10,16,18,15,23,17,33,15,13,11,14,17,19,20,12,21,7,19,10,13,10,8,21,11,9,14,14,15,18,16,12,20,8,3,13,4,1,10,8,13,10,21,18,21,34,25,34,33,40,42,36,72,75,76,92,71,112,106,101,170,135,106,68,48,48,26,33,29,17,12,13,17,15,14,15,10,9,2,6,8,5,1,2,3,4,3,1,3,5,2,3,2,3,2,2,3,4,3,4,4,4,7,6,15,11,9,9,12,13,13,13,20,28,45,28,34,41,36,38,48,27,23,28,42,30,18,38,28,36,44,41,35,28,28,22,26,24,9,21,10,15]

X_train,y_train,X_test,y_test = ts_utils.create_train_test(SS.transform(data__full),3,781)

n_hidden = 2

init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
init_out = np.random.randn(n_hidden,K).astype(floatX)




X_train1 = np.array(np.transpose(X_train)[0]).reshape((-1,1))
X_train2 = np.array(np.transpose(X_train)[1]).reshape((-1,1))
X_train3 = np.array(np.transpose(X_train)[2]).reshape((-1,1))
# X_train4 = np.array(np.transpose(X_train)[3]).reshape((-1,1))
# X_train5 = np.array(np.transpose(X_train)[4]).reshape((-1,1))
# X_train6 = np.array(np.transpose(X_train)[5]).reshape((-1,1))



ann_input1 = theano.shared(X_train1,broadcastable=(False,True))
ann_input2 = theano.shared(X_train2,broadcastable=(False,True))
ann_input3 = theano.shared(X_train3,broadcastable=(False,True))
# ann_input4 = theano.shared(X_train4,broadcastable=(False,True))
# ann_input5 = theano.shared(X_train5,broadcastable=(False,True))
# ann_input6 = theano.shared(X_train6,broadcastable=(False,True))


ann_output = theano.shared(y_train)
with pm.Model() as model:
    #alpha = pm.Normal('alpha', 0., 100., shape=K)
        alpha = pm.Normal('alpha', 0., 1., shape=K)
        beta = pm.Normal('beta', 0., 1., shape=K)
        beta1 = pm.Normal('beta1', 0., 1., shape=K)
        beta2 = pm.Normal('beta2', 0., 1., shape=K)
        beta3 = pm.Normal('beta3', 0., 1., shape=K)
        #beta4 = pm.Normal('beta4', 0., 1., shape=K)
        #beta5 = pm.Normal('beta5', 0., 1., shape=K)
        #beta6 = pm.Normal('beta6', 0., 1., shape=K)
        # v = norm_cdf(alpha+beta3*ann_input3+\
        #     beta1*ann_input1+\
        #     beta2*ann_input2+\
        #     beta4*ann_input4+\
        #     beta5*ann_input5+\
        #     beta6*ann_input6 )
        v =  tt.nnet.sigmoid(beta* \
             tt.nnet.sigmoid( \
                # beta4*ann_input4+\
                # beta5*ann_input5+\
                beta3*ann_input3+\
                beta1*ann_input1+\
                beta2*ann_input2+\
                # beta6*ann_input6+\
                alpha))
        w = pm.Deterministic('w', stick_breaking(v))




with model:


    gamma = pm.Gamma('gamma', 0., 100., shape=K)
    delta = pm.Normal('gamma1', 0., 100., shape=K)
    delta1 = pm.Normal('gamma2', 0., 100., shape=K)
    delta2 = pm.Normal('gamma3', 0., 100., shape=K)
    # delta3 = pm.Normal('gamma4', 0., 100., shape=K)
    # delta4 = pm.Normal('gamma5', 0., 100., shape=K)
    # delta5 = pm.Normal('gamma6', 0., 100., shape=K)
    mu = pm.Deterministic('mu', \
            gamma +\
            ann_input1*delta+\
            delta1*ann_input2+\
            delta2*ann_input3)#+\
            # delta3*ann_input4+\
            # delta4*ann_input5+\
            # delta5*ann_input6)


with model:
    tau = pm.Gamma('tau', 1., 1., shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=y_train)

SAMPLES = 2000
BURN = 1000
THIN = 10

with model:
    step = pm.Metropolis()
    trace_ = pm.sample(SAMPLES, step, random_seed=SEED)
   # s = theano.shared(pm.floatX(1))
    #inference = pm.ADVI(cost_part_grad_scale=s)
    # ADVI has nearly converged
    #pm.fit(n=20000, method=inference)
    # It is time to set `s` to zero
    #s.set_value(0)
    #approx = pm.fit(n=30000)
trace = trace_[BURN::THIN]
#trace = approx.sample(draws=5000)

PP_SAMPLES = 1000

X_test1 = np.array(np.transpose(X_test)[0]).reshape((-1,1))
X_test2 = np.array(np.transpose(X_test)[1]).reshape((-1,1))
X_test3 = np.array(np.transpose(X_test)[2]).reshape((-1,1))
# X_test4 = np.array(np.transpose(X_test)[3]).reshape((-1,1))
# X_test5 = np.array(np.transpose(X_test)[4]).reshape((-1,1))
# X_test6 = np.array(np.transpose(X_test)[5]).reshape((-1,1))


ann_input1.set_value(np.array(X_test1,dtype=np.float32).reshape((-1,1)))
ann_input2.set_value(np.array(X_test2,dtype=np.float32).reshape((-1,1)))
ann_input3.set_value(np.array(X_test3,dtype=np.float32).reshape((-1,1)))
# ann_input4.set_value(np.array(X_test4,dtype=np.float32).reshape((-1,1)))
# ann_input5.set_value(np.array(X_test5,dtype=np.float32).reshape((-1,1)))
# ann_input6.set_value(np.array(X_test6,dtype=np.float32).reshape((-1,1)))


with model:
    pp_trace = pm.sample_ppc(trace, PP_SAMPLES, random_seed=SEED)


fig, ax = plt.subplots(figsize=(8, 6))

print (trace['alpha'].mean(axis=0))
print (trace['beta1'].mean(axis=0))
print (trace['beta2'].mean(axis=0))
print (trace['beta3'].mean(axis=0))

print (trace['beta'].mean(axis=0))

print (trace['w'].shape)
ax.bar(np.arange(K) + 1,
       trace['w'].mean(axis=0).max(axis=0));

ax.set_xlim(1 - 0.5, K + 0.5);
ax.set_xticks(np.arange(0, K, 2) + 1);
ax.set_xlabel('Mixture component');

ax.set_ylabel('Largest posterior expected\nmixture weight');
plt.show()
# prod = 0
# for elm in range(500):

#     dist1 = np.histogram(pp_trace['obs'][elm],bins=range(-50,100),density=True)
#     probs = dist1[0]
#     vals = dist1[1]

#     for val in range(1,len(vals)-1):
#         if vals[val] < data__full[781+elm] and data__full[781+elm] <= vals[val+1]:
#             prod += np.log(probs[val])
#             #print (data__full[901+elm],np.log(probs[val]))
#             break
# print (prod)
#print (pp_trace['obs'].mean(axis=0))
print (mean_squared_error(SS.inverse_transform(pp_trace['obs'].mean(axis=0)),SS.inverse_transform(y_test)))

