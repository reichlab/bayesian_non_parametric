from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import copy

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam


def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params



if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    D = 5

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

    def log_density1(x, t):
       # print (x.shape)
        mu, log_sigma = x[:, 0], x[:, 1]
        sigma_density = norm.logpdf(log_sigma, 0, 1.35)
        mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
        print (mu_density.shape)
        return sigma_density + mu_density

    def log_density(theta,t):
        ret_density = []
        for theta_i in theta:
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
            ret_density.append(ret_val)
        return (ret_density)


    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_density, D, num_samples=2000)


   

    print("Optimizing variational parameters...")
    init_mean    = -1 * np.ones(D)
    init_log_std = -5 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=2000)
    print ("Done!")
    print (variational_params)