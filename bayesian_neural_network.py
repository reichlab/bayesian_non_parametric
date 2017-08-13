
import pymc3 as pm
import theano.tensor as T
import theano
import sklearn
import scipy
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
import csv
from matplotlib import pyplot as plt
from scipy.linalg import norm, pinv
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
#data__full = [589,561,640,656,727,697,640,599,568,577,553,582,600,566,653,673,742,716,660,617,583,587,565,598,628,618,688,705,770,736,678,639,604,611,594,634,658,622,709,722,782,756,702,653,615,621,602,635,677,635,736,755,811,798,735,697,661,667,645,688,713,667,762,784,837,817,767,722,681,687,660,698,717,696,775,796,858,826,783,740,701,706,677,711,734,690,785,805,871,845,801,764,725,723,690,734,750,707,807,824,886,859,819,783,740,747,711,751,804,756,860,878,942,913,869,834,790,800,763,800,826,799,890,900,961,935,894,855,809,810,766,805,821,773,883,898,957,924,881,837,784,791,760,802,828,778,889,902,969,947,908,867,815,812,773,813,834,782,892,903,966,937,896,858,817,827,797,843]

data__full = [4,5,4,3,6,2,4,5,10,6,8,2,6,17,23,13,21,28,24,20,40,27,42,33,43,37,57,71,44,56,53,52,47,26,27,21,21,26,34,37,17,19,25,18,21,17,17,16,16,15,23,16,17,12,17,10,15,19,21,14,18,13,14,18,23,25,62,60,76,66,64,68,89,92,140,116,142,129,140,140,127,129,169,141,108,78,70,81,104,90,85,55,53,65,33,38,59,40,37,29,30,30,28,23,24,29,26,23,20,19,20,26,29,31,28,26,32,35,33,30,52,59,67,65,74,70,61,53,76,61,57,44,34,47,60,60,53,36,31,30,32,28,33,33,35,22,13,13,21,17,11,8,8,6,6,7,12,17,10,10,18,19,12,22,12,21,18,16,16,22,17,25,23,12,25,28,27,18,23,23,29,38,36,43,46,31,25,40,31,38,30,22,31,26,35,36,39,25,31,37,33,25,24,18,23,13,18,14,17,22,13,24,31,34,31,31,38,49,42,49,55,80,84,72,89,115,179,202,272,302,395,426,461,381,333,353,410,364,359,288,221,149,112,154,91,72,56,46,37,26,17,17,20,11,7,16,14,16,5,2,6,5,4,3,4,16,8,7,10,14,7,9,11,23,17,19,24,17,28,40,33,31,33,29,30,36,48,40,28,36,19,34,23,17,17,23,14,20,13,23,20,16,16,23,14,15,4,5,5,11,11,7,4,6,5,2,4,2,4,6,6,4,6,11,16,9,12,13,27,21,19,17,24,27,30,29,25,35,33,30,29,31,29,22,27,24,26,29,22,33,24,30,20,17,24,28,18,13,9,14,11,11,19,10,8,8,9,3,7,14,4,9,14,7,9,3,3,14,12,10,21,26,47,42,31,34,33,52,56,70,112,70,47,48,49,66,56,61,67,64,68,49,50,56,75,63,62,41,50,34,31,38,30,32,26,30,36,35,46,48,44,51,59,71,102,128,127,150,191,256,329,263,220,204,181,99,54,80,102,127,73,68,64,55,67,84,85,67,73,89,68,59,56,77,75,47,50,42,28,37,37,27,12,15,22,8,15,17,10,9,11,20,13,11,16,11,7,17,14,13,15,30,25,40,44,25,21,48,56,60,45,55,32,46,61,42,37,43,34,40,25,16,17,17,16,23,18,18,9,7,7,4,3,2,8,3,1,1,2,3,3,2,0,0,2,2,0,6,3,6,2,3,2,4,5,2,9,2,4,8,6,3,11,14,15,20,9,20,28,38,30,30,23,16,22,28,14,17,20,17,10,13,20,9,18,9,8,19,11,4,6,6,8,13,8,8,5,16,12,11,18,10,22,14,16,18,27,38,35,41,51,65,55,54,62,64,56,65,71,75,71,72,47,27,35,25,19,37,38,34,26,19,18,22,16,18,6,12,6,6,3,7,6,1,3,2,2,1,10,3,3,1,1,2,6,3,3,5,4,7,6,5,7,6,4,4,7,9,5,5,10,6,13,6,5,5,9,3,6,11,7,7,15,9,6,6,6,7,10,8,7,12,3,2,7,5,5,7,7,7,7,10,13,10,14,11,20,25,17,18,25,21,31,32,26,35,28,37,41,34,30,39,39,39,34,30,37,29,26,15,22,15,20,14,10,21,14,14,9,11,5,6,7,11,4,3,2,6,10,7,5,3,12,13,10,13,13,8,21,18,8,7,20,14,14,7,14,10,13,27,13,18,16,16,20,17,4,15,8,6,12,15,11,10,15,17,7,7,8,9,12,12,5,4,11,4,5,7,1,1,4,2,6,3,4,10,12,21,26,21,30,45,56,75,83,82,126,119,137,131,112,82,73,43,55,55,53,46,43,29,22,26,13,17,8,13,10,17,19,9,9,9,3,7,7,0,2,3,3,1,3,3,3,7,3,5,11,5,5,6,6,4,4,8,14,12,16,10,16,18,15,23,17,33,15,13,11,14,17,19,20,12,21,7,19,10,13,10,8,21,11,9,14,14,15,18,16,12,20,8,3,13,4,1,10,8,13,10,21,18,21,34,25,34,33,40,42,36,72,75,76,92,71,112,106,101,170,135,106,68,48,48,26,33,29,17,12,13,17,15,14,15,10,9,2,6,8,5,1,2,3,4,3,1,3,5,2,3,2,3,2,2,3,4,3,4,4,4,7,6,15,11,9,9,12,13,13,13,20,28,45,28,34,41,36,38,48,27,23,28,42,30,18,38,28,36,44,41,35,28,28,22,26,24,9,21,10,15]
SS = StandardScaler().fit(data__full)
data__full = SS.transform(data__full)
data__ = data__full[:781]

m = 1
train_target = []
train_data = []
test_data = []
test_target = []
for i in range(1,len(data__)):
    train_data.append(data__[i-1])
    train_target.append(data__[i])


X_train =  np.array(train_data,dtype=np.float32).reshape((-1,1))
y_train =  np.array(train_target,dtype=np.float32).reshape((-1))
X_test = np.array(data__full[780:len(data__full)-1],dtype=np.float32).reshape((-1,1))
y_test = data__full[781:]
ann_input = theano.shared(X_train)
ann_output = theano.shared(y_train)
#hidden_size = 1
beta = 1
def basisfunc( c, d):
        return T.exp(- (c.repeat(len(centers),axis=1)\
            -d.reshape((-1,len(centers))))**2)

hidden_size = 2

kmeans = KMeans(n_clusters=50, random_state=0).fit(X_train)
centers = kmeans.cluster_centers_.reshape((-1))

with pm.Model() as model:
    # Below we require an ordering of the summed weights, thus initialize in this order


    tau = pm.Gamma('tau', 1., 1.)

    #center_arr = []
    #for i in range(hidden_size):
       # center_arr.append(pm.Normal('w_in_'+str(i), 0, sd=10))


    weights = pm.Normal('w_1_out', 0, sd=10,shape=len(centers))
    distances = basisfunc(ann_input \
        ,centers)



    #act_out = T.dot(a1, weights_1_out)
    tmp_ = T.dot(distances,weights)
    out = pm.Normal('out',tmp_
                       ,tau=tau,
                       observed=ann_output)



with model:
    #s = theano.shared(pm.floatX(1))
    #inference = pm.ADVI(cost_part_grad_scale=s)
    # ADVI has nearly converged
    #pm.fit(n=20000, method=inference)
    # It is time to set `s` to zero
    #s.set_value(0)
    #approx = pm.fit(n=30000)

    step = pm.Metropolis()
    trace = pm.sample(50000 ,step=step)

# ann_input.set_value(X_test)

#trace = approx.sample(draws=5000)



# x = T.matrix('X')
# # symbolic number of samples is supported, we build vectorized posterior on the fly
# n = T.iscalar('n')
# # Do not forget test_values or set theano.config.compute_test_value = 'off'
# x.tag.test_value = np.empty_like(X_test[:10])
# n.tag.test_value = 10000
# _sample_proba = approx.sample_node(model.out.distribution.mu, size=n,
#                                    more_replacements={ann_input:x})
# sample_proba = theano.function([x, n], _sample_proba)
# pred = sample_proba(X_test, 1000).mean(axis=0)
PP_SAMPLES = 1000
ann_input.set_value(X_test)
with model:
     pp_trace = pm.sample_ppc(trace, PP_SAMPLES)

pred = pp_trace['out'].mean(axis=0)
print (mean_squared_error(SS.inverse_transform(pred),\
     SS.inverse_transform(y_test)))

# with model:
#     pp_trace = pm.sample_ppc(trace, PP_SAMPLES)
# print (mean_squared_error(SS.inverse_transform(pp_trace['out'].mean(axis=0)),\
#     SS.inverse_transform(y_test)))
#trace = approx.sample(draws=5000)
# pm.traceplot(trace);
# plt.show()