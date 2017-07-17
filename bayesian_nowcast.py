import pymc3 as pm
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from theano import tensor as TT
import sys
## NOTES:

# 1) incorporate normal
# 2) integrating epi curve 




## SLIDING WINDOW
m = 4
T = 4
t = 1
D = 3
A_t = [(t,d) for t in range(max(T-m,0),T) for d in range(0, min(D,T-t))]

data = np.matrix('1 0 0 0; 2 1 0 0; 3 2 1 0; 4 3 2 1')
observed_data = [(x,y,data[x,y]) for x,y in A_t]



def n_dot_D_i(D,i):
    sum_ = 0
    for j in observed_data:
        sum_ += j[D-i-1]

    return sum_

def  N_dot_D_i(D,i):
    sum_ = 0
    for j in observed_data:
        for k in range(D-i):
            sum_ += j[k]

    return sum_
def n_t_T(t,T):
    sum_ = 0
    for d in range(0,min(T-t,D)):
        sum_ += data[t,d]
    return sum_


def n_t_T_vec(t,T):
    sum_ = []
    for d in range(0,min(T-t,D)):
        sum_.append(data[t,d])
    return sum_


with pm.Model() as rsv_model:
    beta_1 = pm.Normal('beta1',mu=.5,sd=.1)
    beta_2 = pm.Normal('beta2',mu=.5,sd=.1)
    k = 5

    p_1 = pm.Beta('b1',alpha=k+n_dot_D_i(D,0),beta=k+N_dot_D_i(D,0)-n_dot_D_i(D,0))
    p_2 = pm.Beta('b2',alpha=k+n_dot_D_i(D,1),beta=k+N_dot_D_i(D,1)-n_dot_D_i(D,1))
    p_3 = pm.Beta('b3',alpha=k+n_dot_D_i(D,2),beta=k+N_dot_D_i(D,2)-n_dot_D_i(D,2))
    sum_betas = p_1
    p_2 *= (1-sum_betas)
    sum_betas += p_2
    p_3 *= (1-sum_betas)
    sum_betas += p_3
    p_d = [p_1,p_2,p_3x]


    binomial_array = []
    



    for t_ in range(t,T,1):
        lambda_ = np.exp(beta_1+beta_2*t)#pm.Gamma('lambda',alpha=10,beta=1)
        n_total = pm.Poisson('n_total',lambda_)+1
        binomial_array.append((n_total, TT.sum(p_d[:T-t_])))
    
    def logp(observed_data):
        prod_ = 1
        for i in range(len(binomial_array)):
            prod_ +=   observed_data[i]*TT.log(binomial_array[i][1])+ (binomial_array[i][0]-observed_data[i])*TT.log(1-binomial_array[i][1])
            
        return prod_
    total_data = []
    for itr in range(t,T,1):
        total_data.append(n_t_T(itr,T))
    x = pm.DensityDist('x', logp, observed=total_data)

    
    step = pm.Metropolis()
    trace = pm.sample(200000, step=step) 
pm.traceplot(trace)
plt.show()