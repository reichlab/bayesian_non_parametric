from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import copy
import sys
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam
num_samples = 1000

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
        np.loadtxt(open("/home/gcgibson/Desktop/bayesian_non_parametric/tmp_data/X_train.csv", "rb"), delimiter=",", skiprows=0)


    y_train = \
        np.loadtxt(open("/home/gcgibson/Desktop/bayesian_non_parametric/tmp_data/y_train.csv", "rb"), delimiter=",", skiprows=0)


    X_test = \
        np.loadtxt(open("/home/gcgibson/Desktop/bayesian_non_parametric/tmp_data/X_test.csv", "rb"), delimiter=",", skiprows=0)




    X_train = X_train
    y_train = y_train



    def gaussian_kernel_2d(x,y,h):
	ret_density = []
	for i in range(len(h)):
		mat = np.diag(h[i])
		precision_mat = np.linalg.inv(mat)
        	tmp = np.dot(np.dot(np.transpose(x[i]-y[i]),precision_mat),(x[i]-y[i]))
        	tmp2 = 1.0/(np.sqrt(np.linalg.det(2*np.pi*mat)))*np.exp(-.5*tmp)
		ret_density.append(tmp2)
	return ret_density
    lambda_ = 1
    joint_data = np.hstack((X_train,y_train.reshape((-1,1))))

    def log_density1(x, t):
        print (t)
        mu, log_sigma = x[:, 0], x[:, 1]
        sigma_density = norm.logpdf(log_sigma, 0, 1.35)
        mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
        print (mu_density.shape)
        return sigma_density + mu_density

    def log_density(theta_i,t):
	print (t)
	tmp_theta = np.exp(theta_i)
            # Rob Hyndman's prior stated here https://robjhyndman.com/papers/mcmckernel.pdf
        
	ln_prior = np.log(1/(1+lambda_*tmp_theta**2))
            #LOO likelihood averaged over all data points
        likelihood_proposal = np.zeros(num_samples)
	for i1 in range(len(joint_data)):
                    loo_likelihood_proposal = np.zeros(num_samples)
                    data_mod = copy.copy(joint_data)
                    test_xy = joint_data[i1]
                    data_mod = np.delete(data_mod,(i1),axis=0)
                    for i in range(len(data_mod)):
                        tmp_density = gaussian_kernel_2d(np.repeat(test_xy.reshape((1,-1)),num_samples,axis=0),np.repeat(data_mod[i].reshape((1,-1)),num_samples,axis=0),tmp_theta)
			tmp_density = np.array(tmp_density)
			loo_likelihood_proposal += tmp_density
            	
                    likelihood_proposal += loo_likelihood_proposal
	likelihood_proposal = likelihood_proposal+1e-100 #/(1.0*len(joint_data)-1 )
	ret_val = np.log(likelihood_proposal) + ln_prior.sum()
        if np.isnan(ret_val.any()) == True:
	    print ("nanned")    
            ret_val = -1e10
	return (ret_val)


    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_density, D, num_samples=num_samples)


    print("Optimizing variational parameters...")
    init_mean    = 0 * np.ones(D)
    init_log_std = 1 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=2000)
    print (variational_params)
