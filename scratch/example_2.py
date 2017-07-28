import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
# Test data
# Define the kernel function

def kernel(x, y, param):
	K = []
	for elm1 in x:
		row = []
		for elm2 in y:
			row.append(math.exp(-1.0/param*(elm1-elm2)**2))
		K.append(row)
	return np.matrix(K)

plt.ion()
axes = plt.gca()
axes.set_xlim([-5,5])
axes.set_ylim([-5,5])
param =1
num_sample= 10
for i in range(num_sample):
	color = np.random.rand(3,1)	
	Xtest = np.array(np.random.uniform(-5,5,size=100)).reshape((-1,1))
	K_ss = kernel(Xtest, Xtest, param)
	draws = multivariate_normal.rvs(np.zeros(100),K_ss,size=1)
	draws = np.transpose(draws)
	plt.scatter(Xtest,np.zeros(100),color='black',marker='x')
	plt.pause(1)
	list1, list2 = (list(t) for t in zip(*sorted(zip(Xtest.reshape((100)), draws))))	
	plt.plot(list1,list2,color=color)
	plt.pause(2)
plt.show()



sys.exit()


print (K_ss)
# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))


# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix

#f_prior = np.dot(L, np.random.normal(size=(10,1000)))

# Now let's plot the 3 sampled functions.
