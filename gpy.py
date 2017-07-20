#!/usr/bin/env python
# -*- coding: utf8 -*- 
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import matplotlib.pyplot as plt
import theano
theano.config.compute_test_value = "ignore"

n = 5

X = np.array([0, 1, 2, 3, 4]).reshape(5,1)
Y = np.sin(X)
m = 100
X0 = np.linspace(0, 3, m)[:, None]

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining



with pm.Model() as model:
    noise = 0.1
    lengthscale = 0.3
    f_scale = 1

    cov = f_scale * pm.gp.cov.ExpQuad(1, lengthscale)
    K = cov(X)
    K_s = cov(X, X0)
    K_noise = K + noise * tt.eye(n)

    # Add very slight perturbation to the covariance matrix diagonal to improve numerical stability
    K_stable = K + 1e-12 * tt.eye(n)

    # Observed data
    f = np.random.multivariate_normal(mean=np.zeros(n), cov=K_noise.eval())



    # The actual distribution of f_sample doesn't matter as long as the shape is right since it's only used
    # as a dummy variable for slice sampling with the given prior
    f_sample = pm.Flat('f_sample', shape=(n, ))

    # Likelihood
    y = pm.MvNormal('y', observed=f, mu=f_sample, cov=noise * tt.eye(n), shape=n)

    # Interpolate function values using noisy covariance matrix
    L = tt.slinalg.cholesky(K_noise)
    f_pred = pm.Deterministic('f_pred', tt.dot(K_s.T, tt.slinalg.solve(L.T, tt.slinalg.solve(L, f_sample))))
    L = np.linalg.cholesky(K_noise.eval())
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, f))
    post_mean = np.dot(K_s.T.eval(), alpha)
    
    beta = pm.Beta('beta', 1., alpha, shape=n)
    w = pm.Deterministic('w', stick_breaking(beta))


    obs = pm.NormalMixture('obs', w, post_mean, tau=1,
                           observed=Y)

    # Use elliptical slice sampling
    ess_step = pm.EllipticalSlice(vars=[f_sample], prior_cov=K_stable)

    trace = pm.sample(5000, start=model.test_point, step=[ess_step], progressbar=False, random_seed=1)









fig, ax = plt.subplots(figsize=(14, 6));
for idx in np.random.randint(4000, 5000, 500):
    ax.plot(X0, trace['f_pred'][idx],  alpha=0.02, color='navy')
ax.scatter(X, f, s=40, color='k', label='True points');
ax.plot(X0, post_mean, color='g', alpha=0.8, label='Posterior mean');
ax.legend();
ax.set_xlim(-5, 5);
ax.set_ylim(-2, 2);
plt.show()