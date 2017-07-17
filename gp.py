import matplotlib.pyplot as plt
import matplotlib.cm as cmap
cm = cmap.inferno

import numpy as np
import scipy as sp
import theano
import theano.tensor as tt
import theano.tensor.nlinalg
import sys
sys.path.insert(0, "../../..")
import pymc3 as pm
import csv
data = []

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining
with open('article-disease-pred-with-kcde/data-raw/San_Juan_Training_Data.csv') as csvfile:
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

X = np.array(x,dtype=np.float64).reshape((len(x),1))
y = np.array(y,dtype=np.float64)

n = len(y)
plt.scatter(X,y)
plt.show()

with pm.Model() as model:
    # f(x)
    l_true = 0.3
    s2_f_true = 1.0
    cov = s2_f_true * pm.gp.cov.ExpQuad(1, l_true)

    # noise, epsilon
    s2_n_true = 0.1
    K_noise = s2_n_true**2 * tt.eye(n)
    K = cov(X) + K_noise

# evaluate the covariance with the given hyperparameters
K = theano.function([], cov(X) + K_noise)()

# generate fake data from GP with white noise (with variance sigma2)
# y = np.random.multivariate_normal(np.zeros(n), K)

Z = np.linspace(0,100,100)[:,None]

with pm.Model() as model:
    # priors on the covariance function hyperparameters
    l = pm.Uniform('l', 0, 10)

    # uninformative prior on the function variance
    log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=5)
    s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

    # uninformative prior on the noise variance
    log_s2_n = pm.Uniform('log_s2_n', lower=-10, upper=5)
    s2_n = pm.Deterministic('s2_n', tt.exp(log_s2_n))

    # covariance functions for the function f and the noise
    f_cov = s2_f * pm.gp.cov.ExpQuad(1, l)

    y_obs = pm.gp.GP('y_obs', cov_func=f_cov, sigma=s2_n, observed={'X':X, 'Y':y})
    
with model:
    trace = pm.sample(2000, burn=1000)
with model:
    gp_samples = pm.gp.sample_gp(trace, y_obs, Z, samples=50, random_seed=42)



fig, ax = plt.subplots(figsize=(14,5))

[ax.plot(Z, x, color=cm(0.3), alpha=0.3) for x in gp_samples]
# overlay the observed data
ax.plot(X, y, 'ok', ms=10);
ax.set_xlabel("x");
ax.set_ylabel("f(x)");
ax.set_title("Posterior predictive distribution");
plt.show()