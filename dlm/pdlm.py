from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
import sys
import numpy as np
from numpy.random import random


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 1))
    particles[:, 0] = mean + (randn(N) * std)
    return particles

def predict(particles,t):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    particles[:, 0] += randn(N)*.01

    return particles

def neff(weights):
    return 1. / np.sum(np.square(weights))


def update(particles, weights,ts,t):
    for p in range(len(particles)):
        weights[p] *= scipy.stats.norm.pdf(ts[t],p,100)
    return weights/sum(weights)  


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var


def multinomal_resample(weights):
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors
    return np.searchsorted(cumulative_sum, random(len(weights)))


def run_pf1(N, iters=18, sensor_std_err=.1, 
            do_plot=True, plot_particles=False,
            xlim=(0, 20), ylim=(0, 20),
            initial_x=None):
    
    time_series = 100*np.ones(100)
    initial_x = 100
    particles = create_gaussian_particles(
            mean=initial_x, std=( 100), N=N)
  

    
    weights = (1.0)/(1.0*N)*np.ones(N)


    
    xs = []

    initial_state = np.array([0])
    
    for t in range(len(time_series)):

        particles = predict(particles,t)
        
        # incorporate measurements
        weights = update(particles, weights,time_series, t)

        resample = multinomal_resample(weights)
        mu, var = estimate(particles, weights)
        xs.append(mu)

    return xs,particles

from numpy.random import seed
 
estimated_states, particles = run_pf1(N=5000, plot_particles=False)   

