
from matplotlib import pyplot as plt
import numpy as np
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
size = 100
mean = [0, 0]
cov = [[1, 0], [0, 1]] 
n_hidden = 5
x, y = np.random.multivariate_normal(mean, cov, size).T
data = []

with open('../article-disease-pred-with-kcde/data-raw/San_Juan_Training_Data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(int(row['total_cases']))
        # if int(row['season_week']) in data.keys():
        #     data[int(row['season_week'])] += int(row['total_cases'])
        # else: 
        #     data[int(row['season_week'])] = int(row['total_cases'])

y = data[:50]


x = y[:len(y)-1]
y = y[1:]
size = len(x)

x = np.array(x,dtype=np.float64)
y = np.array(y,dtype=np.float64)




blue = sns.color_palette()
SEED = 972915 # from random.org; for reproducibility
np.random.seed(SEED)

DATA_URI = 'http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat'

def standardize(x):
    return (x - x.mean()) / x.std()

df = (pd.read_csv(DATA_URI, sep=' *', engine='python')
        .assign(std_range=lambda df: standardize(df.range),
                std_logratio=lambda df: standardize(df.logratio)))


fig, ax = plt.subplots(figsize=(8, 6))

# ax.scatter(df.std_range, df.std_logratio,
#            c=blue);

ax.set_xticklabels([]);
ax.set_xlabel("Standardized range");

ax.set_yticklabels([]);
ax.set_ylabel("Standardized log ratio");


def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))

def stick_breaking(v):
    return v * tt.concatenate([tt.ones_like(v[:, :1]),
                               tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                              axis=1)
def kernel(a, b, kernel_name='squared_exponential'):

    def squared_exponential(a, b):
        """Squared exponential kernel 
    
        """
        D = np.sum(a**2, axis=1, keepdims=True) + np.sum(b**2, axis=1) - 2 * np.dot(a, b.T)
        return eta * np.exp(- rho * D)

    def periodic(a, b):
        """periodic kernel
    
        """
        a = np.sin(a)
        b = np.sin(b)
        D = np.sum(a**2, axis=1, keepdims=True) + np.sum(b**2, axis=1) - 2 * np.dot(a, b.T)
        return eta * np.exp(- rho * D)
   
    if kernel_name == 'squared_exponential':
        return squared_exponential(a, b)
    elif kernel_name == 'periodic':
        return periodic(a, b)


N= size
K = size

std_range = X =  x.reshape((size,1))#df.std_range.values[:, np.newaxis]
std_logratio = y.reshape((size,1))#df.std_logratio.values[:, np.newaxis]
print std_range.shape
Xtest = np.linspace(0, 10, 100).reshape(-1,1)

x_lidar = shared(std_range, broadcastable=(False, True))
with pm.Model() as model:
    #alpha = pm.Normal('alpha', 0., 1., shape=K)
    #beta = pm.Normal('beta', 0., 1., shape=K)
    alpha = pm.Normal('alpha', 0., 1.0, shape=K)
    beta = pm.Normal('beta', 0., 1.0, shape=K)
    v = norm_cdf(alpha + beta * x_lidar)
    beta_sp  = pm.Beta('beta-sp',1,.2,shape=K)
    w = pm.Deterministic('w', stick_breaking(beta_sp*tt.ones_like(v)))


with model:
    gamma = pm.Normal('gamma', 0., 1., shape=K)
    delta = pm.Normal('delta', 0., 10., shape=K)
    mu = pm.Deterministic('mu', gamma + delta * x_lidar)


with model:
    tau = pm.Gamma('tau', 1., 1., shape=K)    
    obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=std_logratio)

SAMPLES = 20000
BURN = 10000
THIN = 10
    
with model:
    step = pm.Metropolis()
    trace_ = pm.sample(SAMPLES, step, random_seed=SEED)
    
trace = trace_[BURN::THIN]


PP_SAMPLES = 5000

lidar_pp_x = np.linspace(std_range.min() - 0.05, std_range.max() + 0.05, 100)
x_lidar.set_value(lidar_pp_x[:, np.newaxis])

with model:
    pp_trace = pm.sample_ppc(trace, PP_SAMPLES, random_seed=SEED)


fig, ax = plt.subplots()

ax.scatter(std_range, std_logratio,
           c=blue, zorder=10,
           label=None);

low, high = np.percentile(pp_trace['obs'], [2.5, 97.5], axis=0)
ax.fill_between(lidar_pp_x, low, high,
                color='k', alpha=0.35, zorder=5,
                label='95% posterior credible interval');

ax.plot(lidar_pp_x, pp_trace['obs'].mean(axis=0),
        c='k', zorder=6,
        label='Posterior expected value');

ax.set_xticklabels([]);
ax.set_xlabel('Standardized range');

ax.set_yticklabels([]);
ax.set_ylabel('Standardized log ratio');

ax.legend(loc=1);
ax.set_title('LIDAR Data');

ax.set_autoscaley_on(False)
ax.set_autoscalex_on(False)

ax.set_ylim([0,100])
ax.set_xlim([0,100])
fig, ax = plt.subplots(figsize=(8, 6))
print (trace['w'].shape)

# for elm in  trace['w'].mean(axis=0):
#     fig, ax = plt.subplots()
#     ax.bar(np.arange(K) + 1,
#          elm );

#     ax.set_xlim(1 - 0.5, K + 0.5);
#     ax.set_xticks(np.arange(0, K, 2) + 1);
#     ax.set_xlabel('Mixture component');


#     ax.set_ylabel('Largest posterior expected\nmixture weight');
#     print ("HELLO")
#     fig.show()
plt.show()
