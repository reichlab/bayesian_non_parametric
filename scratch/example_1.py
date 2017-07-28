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

f, axarr = plt.subplots(2)
plt.ion()
param =100
num_sample= 10
for i in range(num_sample):
	color = np.random.rand(3,1)	
	Xtest = np.array([np.random.uniform(-5,-4),np.random.uniform(4,5)]).reshape((-1,1))
	K_ss = kernel(Xtest, Xtest, param)
	draws = multivariate_normal.rvs([0,0],K_ss,size=100)
	draws = np.transpose(draws)
	axarr[0].scatter(Xtest,[0,0],color='black',marker='x')
	plt.pause(10)
	tb = axarr[1].table(cellText=K_ss,cellLoc='center',loc='center')
	tb.auto_set_font_size(False)
	tb.set_fontsize(20)
	for key, cell in tb.get_celld().items():
        	cell.set_linewidth(0)        
	for key, cell in tb.get_celld().items():
		cell.set_height(.11)
	axarr[1].table

	axarr[1].set_xticks([])
	axarr[1].set_yticks([])
	axarr[1].set_xlim([0,1])
	plt.pause(10)
	axarr[0].scatter(draws[0],  draws[1],alpha=.4,color=color)
	plt.pause(10)
	axarr[0].plot(Xtest,[draws[0][0],draws[1][0]],color=color)
	plt.pause(2)
	axarr[1].clear()
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
