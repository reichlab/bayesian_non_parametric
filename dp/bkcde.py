import numpy as np
import numpy.matlib as nm
import emcee
import matplotlib.pyplot as plt
import copy
from random import shuffle
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import sys
from pyhmc import hmc



X_train = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/X_train.csv", "rb"), delimiter=",", skiprows=0)


y_train = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/y_train.csv", "rb"), delimiter=",", skiprows=0)


X_test = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/X_test.csv", "rb"), delimiter=",", skiprows=0)




X_train = X_train[:10]
y_train = y_train[:10]



def gaussian_kernel_2d(x,y,h):
    mat = h*np.eye(len(x))
    precision_mat = np.linalg.inv(mat)
    tmp = np.dot(np.dot(np.transpose(x-y),precision_mat),(x-y))
    return 1.0/(np.sqrt(np.linalg.det(2*np.pi*mat)))*np.exp(-.5*tmp)






lambda_ = 1

joint_data = np.hstack((X_train,y_train.reshape((-1,1))))

print (joint_data.shape)


data_x = X_train



SAMPLE = True
if SAMPLE == True:
    #lnprobability of loo likelihood over (x,y)
    def lnprob(theta, y):
        # Rob Hyndman's prior stated here https://robjhyndman.com/papers/mcmckernel.pdf
        ln_prior = np.log(1/(1+lambda_*theta**2))
        #LOO likelihood averaged over all data points
        likelihood_proposal = 0
        for i1 in range(len(joint_data)):
            loo_likelihood_proposal = 0
            data_mod = copy.copy(joint_data)
            test_x = joint_data[i1]
            data_mod = np.delete(data_mod,(i1),axis=0)
            print (len(joint_data),len(data_mod))
            for i in range(len(data_mod)):
                loo_likelihood_proposal += \
                   1.0/(theta[0])*gaussian_kernel_2d(np.array([test_x[0]]),data_mod[i][0],theta[0])* \
                   1.0/(theta[1])*gaussian_kernel_2d(np.array([test_x[1]]),data_mod[i][1],theta[1]) * \
                   1.0/(theta[2])*gaussian_kernel_2d(np.array([test_x[2]]),data_mod[i][2],theta[2])*\
                   1.0/(theta[3])*gaussian_kernel_2d(np.array([test_x[3]]),data_mod[i][3],theta[3]) *\
                   1.0/(theta[4])*gaussian_kernel_2d(np.array([test_x[4]]),data_mod[i][4],theta[4])
                            
            likelihood_proposal += np.log(loo_likelihood_proposal)
        likelihood_proposal = likelihood_proposal/(1.0*len(joint_data))
        if math.isnan(np.log(likelihood_proposal)) == False:
            return np.log(likelihood_proposal) + ln_prior.sum()
        return -np.inf


    #start emcee sampler
    ndim, nwalkers = 5, 50
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]
    joint_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[joint_data])
    joint_sampler.run_mcmc(p0, 100)
    joint_samples = joint_sampler.chain[:, 50:, :].reshape((-1, ndim))





#[ 0.50393207,  0.49386658 ,0.55553895  ,0.50597886 ,0.57495547]
expected_joint_bandwith =  joint_samples.mean(axis=0)
print (expected_joint_bandwith)

def get_expected_value(x_test,joint_data,data_x,expected_joint_bandwith):
    expected_value = 0
    joint_density_sum = 0
    marginal_density_sum = 0
    delta_t = .1
    for y_test1 in np.arange(-10,1000,delta_t):
        tmp1 = 0
        tmp2 = 0
        for i in range(len(data_x)):
            what_tmp = x_test + [y_test1]
            tmp1 +=  (delta_t**2)*1.0/(len(joint_data))*gaussian_kernel_2d(what_tmp,joint_data[i],expected_joint_bandwith)
            tmp2  +=  (delta_t)*1.0/len(data_x)*gaussian_kernel_2d(x_test,data_x[i],expected_joint_bandwith[:4])
        expected_value += y_test1*tmp1/tmp2

#expected_value += y_test1*density
    return expected_value

full_season_ahead_predictions = []
for i in range(len(X_test)):
    ex = get_expected_value(X_test[i].tolist(),joint_data,data_x,expected_joint_bandwith)
    full_season_ahead_predictions.append(ex)

print (full_season_ahead_predictions)


