from autograd import numpy as np
from autograd import grad, jacobian
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
from svgd import SVGD



X_train = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/X_train.csv", "rb"), delimiter=",", skiprows=0)


y_train = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/y_train.csv", "rb"), delimiter=",", skiprows=0)


X_test = \
    np.loadtxt(open("/Users/gcgibson/Desktop/bayesian_non_parametric/X_test.csv", "rb"), delimiter=",", skiprows=0)




X_train = X_train
y_train = y_train



def gaussian_kernel_2d(x,y,h):
    mat = h*np.eye(len(x))
    precision_mat = np.linalg.inv(mat)
    tmp = np.dot(np.dot(np.transpose(x-y),precision_mat),(x-y))
    return 1.0/(np.sqrt(np.linalg.det(2*np.pi*mat)))*np.exp(-.5*tmp)

lambda_ = 1
joint_data = np.hstack((X_train,y_train.reshape((-1,1))))
data_x = X_train

def lnprob(theta_i):
    tmp_theta1 = np.exp(theta_i[0])
    tmp_theta2 = np.exp(theta_i[1])
    tmp_theta3 = np.exp(theta_i[2])
    tmp_theta4 = np.exp(theta_i[3])
    tmp_theta5 = np.exp(theta_i[4])
    # Rob Hyndman's prior stated here https://robjhyndman.com/papers/mcmckernel.pdf
    ln_prior = np.log(1/(1+lambda_*theta_i**2))

    #LOO likelihood averaged over all data points
    likelihood_proposal = 0
    for i1 in range(len(joint_data)):
            loo_likelihood_proposal = 0
            data_mod = copy.copy(joint_data)
            test_xy = joint_data[i1]
            data_mod = np.delete(data_mod,(i1),axis=0)
            for i in range(len(data_mod)):
                loo_likelihood_proposal += \
                    1.0/(tmp_theta1)*gaussian_kernel_2d(np.array([test_xy[0]]),data_mod[i][0],tmp_theta1)* \
                    1.0/(tmp_theta2)*gaussian_kernel_2d(np.array([test_xy[1]]),data_mod[i][1],tmp_theta2) * \
                    1.0/(tmp_theta3)*gaussian_kernel_2d(np.array([test_xy[2]]),data_mod[i][2],tmp_theta3)*\
                    1.0/(tmp_theta4)*gaussian_kernel_2d(np.array([test_xy[3]]),data_mod[i][3],tmp_theta4) *\
                    1.0/(tmp_theta5)*gaussian_kernel_2d(np.array([test_xy[4]]),data_mod[i][4],tmp_theta5)
    
    
            likelihood_proposal += loo_likelihood_proposal
    likelihood_proposal = likelihood_proposal/(1.0*len(joint_data)-1 )
    ret_val = np.log(likelihood_proposal) + ln_prior.sum()
    if np.isnan(ret_val) == True:
        ret_val = -np.inf
    return (ret_val)

def dlnprob(theta_i):
    return grad(lnprob)(theta_i)


def grad_overall(theta):
    return_matrix = []
    for theta_i in theta:
        return_matrix.append(dlnprob(theta_i))
    return np.array(return_matrix)



x0 = np.random.normal(0,1, [10,5]);
theta = SVGD().update(x0, grad_overall, n_iter=1000, stepsize=0.01)


print ("result")
print (theta.mean(axis=0))






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


