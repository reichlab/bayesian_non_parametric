import numpy as np
import matplotlib.pyplot as pl
import math
import time

import sys

def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

def softmax(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist
def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def f(neurons):
   
    weight_prior_1 = normpdf(x_next,0,1000)

    weight_prior_2 = normpdf(y_next,0,1000)

    likelihood = 1
    for elm in range(len(train)):
        hidden = np.tanh(x_next*train[elm])
        output = y_next*hidden
        likelihood *= normpdf(targ[elm],output,1)*weight_prior_1*weight_prior_2
    res = likelihood
    return res

train = [.6]
targ = [.6]

def main():
    N = 100000
    x1 = np.arange(N,dtype=np.float)
    x2 = np.arange(N,dtype=np.float)
    x3 = np.arange(N,dtype=np.float)
    x4 = np.arange(N,dtype=np.float)
    x5 = np.arange(N,dtype=np.float)
    x6 = np.arange(N,dtype=np.float)
    
    x1[0] = .5
    x2[0] = .5
    x3[0] = .5
    x4[0] = .5
    x5[0] = .5
    x6[0] = .5

    neurons = [x1,x2,x3,x4,x5,x6]

    counter = 0
    proposal_stdv=1
    for i in range(0, N-1):
        
        move_1 = np.random.normal(x[i], proposal_stdv,size=1)
        move_2 =  np.random.normal(y[i], proposal_stdv,size=1)
        move_3 = np.random.normal(x[i], proposal_stdv,size=1)
        move_4 =  np.random.normal(y[i], proposal_stdv,size=1)
        move_5 = np.random.normal(x[i], proposal_stdv,size=1)
        move_6 =  np.random.normal(y[i], proposal_stdv,size=1)
        moves = [move_1,move_2,move_3,move_4,move_5,move_6]
        for n in range(len(neurons)):
            denom = f([x1[i],x2[i],x3[i],x4[i],x5[i],x6[i]])
            potential_ = neurons
            potential_[n] = moves[n] 
            if np.random.random_sample() <  min(1,f(potential)/ denom
                        ):
                x1[i+1] = move_1
                x2[i+1] = x2[i]
                x3[i+1] = x3[i]
                x4[i+1] = x4[i]
                x5[i+1] = x5[i]
                x6[i+1] = x6[i]
                counter = counter + 1
            else:
                x1[i+1] = x1[i]
                x2[i+1] = x2[i]
                x3[i+1] = x3[i]
                x4[i+1] = x4[i]
                x5[i+1] = x5[i]
                x6[i+1] = x6[i]

        
           
    print("acceptance fraction is ", counter/float(2*N))
    posterior_predictive = []
    for i in range(5000,N,1):
        mean_x = np.array(x[i])
        mean_y = np.array(y[i])
        posterior_predictive.append( mean_y*(np.tanh(mean_x*train[0])))

    print (np.array(x).mean(),np.array(y).mean())
    print (np.array(posterior_predictive).mean())
    pl.xlim([-1,1])
    pl.hist(posterior_predictive, bins=50, color='blue')
    #pl.scatter(range(len(x[1000:])),x[1000:])
    pl.show()

if __name__ == '__main__':
    main()