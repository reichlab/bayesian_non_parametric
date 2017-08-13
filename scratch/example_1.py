import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
from sklearn.manifold import TSNE
# Test data
# Define the kernel function

def kernel1(x, y, param):
    K = []
    for elm1 in x:
        row = []
        for elm2 in y:
            row.append(math.exp(-1.0/param*(elm1-elm2)**2))
        K.append(row)
    return np.matrix(K)

def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

f, axarr = plt.subplots(1)
axarr.set_xlabel("R")
axarr.set_ylabel("R")
#axarr[2].set_xlabel("t-SNE projection ")
param =.1
num_sample= 2
total_size = num_sample


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
plt.ion()
plt.show(block=False)

axarr.set_xlim([-100,100])

for i in np.arange(1000):
    color = np.random.rand(3,1)
    num_sample+=1
    total_size +=1
        #Xtest_ = Xtest = np.append(Xtest,np.array(i).reshape((-1,1)),axis=0)
    Xtest = np.linspace(-100,100,num_sample).reshape((-1,1))
#Xtest = np.append(Xtest,np.random.uniform(0,5,size=num_sample-10)).reshape((-1,1))
    Xtest_ = Xtest
    axarr.scatter(Xtest,np.zeros(total_size),color='black',marker='x',alpha=.01)
    axarr.set_ylim([-5,5])
    K_ss = kernel(Xtest_, Xtest_, param)
    #axarr.scatter(i,0,marker='+')
    draws = multivariate_normal.rvs(np.zeros(total_size),K_ss,size=100)
    draws = np.transpose(draws)

    #tb = axarr[1].table(cellText=[np.round(np.transpose(draws)[0],decimals=1)],cellLoc='center',loc='center')
    #tb.auto_set_font_size(False)
    #tb.set_fontsize(10)
    #for key, cell in tb.get_celld().items():
         #   cell.set_linewidth(0)
  #  for key, cell in tb.get_celld().items():
      #  cell.set_height(.11)
    # axarr[1].table
    # axarr[1].set_xlabel("f(x_i) values for each x_i, from single sample of MVN(0,K(x_i,x_j)) \n which has dimension (num(x),num(x))")
    # axarr[1].set_xticks([])
    # axarr[1].set_yticks([])
    for iter_ in range(20):
        y_out = []
        for elm in range(len(draws)):
            y_out.append(draws[elm][np.random.randint(0,100)])

        axarr.scatter(Xtest,y_out,alpha=.1)

    mean_function_x = []
    mean_function_y = []
    for iter_ in range(20):
        y_out = []
        Xtest_local = np.linspace(-100,100,10)
        for elm in range(len(draws)-1):
            y_out.append(draws[elm][np.random.randint(0,100)])
        yx = zip(Xtest_local,y_out)
        yx = sorted(yx)
        tmp_x = [y for y, x in yx]
        tmp_y = [x for y, x in yx]
        mean_function_x.append(tmp_x)
        mean_function_y.append(tmp_y)
        #axarr.scatter(tmp_x,tmp_y,alpha=.1)
    mean_function_x =np.array(mean_function_x)
    mean_function_y = np.array(mean_function_y)
    axarr.plot(mean_function_x.mean(axis=0),mean_function_y.mean(axis=0))
    #for iter_ in range(10):
    #for elm in range(2):
        #axarr.scatter(i,draws[num_sample][np.random.randint(0,100)],alpha=.5,color='r')

    plt.draw()
    plt.pause(.00001)
    axarr.clear()
    axarr.set_ylim([-5,5])
    axarr.set_xlim([-100,100])

plt.show()


sys.exit()


# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))


# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix

#f_prior = np.dot(L, np.random.normal(size=(10,1000)))

# Now let's plot the 3 sampled functions.
