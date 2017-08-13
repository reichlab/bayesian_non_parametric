import theano

floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
GRAD_CLIP = 100
N_HIDDEN = 5
import lasagne
data__full = [4,5,4,3,6,2,4,5,10,6,8,2,6,17,23,13,21,28,24,20,40,27,42,33,43,37,57,71,44,56,53,52,47,26,27,21,21,26,34,37,17,19,25,18,21,17,17,16,16,15,23,16,17,12,17,10,15,19,21,14,18,13,14,18,23,25,62,60,76,66,64,68,89,92,140,116,142,129,140,140,127,129,169,141,108,78,70,81,104,90,85,55,53,65,33,38,59,40,37,29,30,30,28,23,24,29,26,23,20,19,20,26,29,31,28,26,32,35,33,30,52,59,67,65,74,70,61,53,76,61,57,44,34,47,60,60,53,36,31,30,32,28,33,33,35,22,13,13,21,17,11,8,8,6,6,7,12,17,10,10,18,19,12,22,12,21,18,16,16,22,17,25,23,12,25,28,27,18,23,23,29,38,36,43,46,31,25,40,31,38,30,22,31,26,35,36,39,25,31,37,33,25,24,18,23,13,18,14,17,22,13,24,31,34,31,31,38,49,42,49,55,80,84,72,89,115,179,202,272,302,395,426,461,381,333,353,410,364,359,288,221,149,112,154,91,72,56,46,37,26,17,17,20,11,7,16,14,16,5,2,6,5,4,3,4,16,8,7,10,14,7,9,11,23,17,19,24,17,28,40,33,31,33,29,30,36,48,40,28,36,19,34,23,17,17,23,14,20,13,23,20,16,16,23,14,15,4,5,5,11,11,7,4,6,5,2,4,2,4,6,6,4,6,11,16,9,12,13,27,21,19,17,24,27,30,29,25,35,33,30,29,31,29,22,27,24,26,29,22,33,24,30,20,17,24,28,18,13,9,14,11,11,19,10,8,8,9,3,7,14,4,9,14,7,9,3,3,14,12,10,21,26,47,42,31,34,33,52,56,70,112,70,47,48,49,66,56,61,67,64,68,49,50,56,75,63,62,41,50,34,31,38,30,32,26,30,36,35,46,48,44,51,59,71,102,128,127,150,191,256,329,263,220,204,181,99,54,80,102,127,73,68,64,55,67,84,85,67,73,89,68,59,56,77,75,47,50,42,28,37,37,27,12,15,22,8,15,17,10,9,11,20,13,11,16,11,7,17,14,13,15,30,25,40,44,25,21,48,56,60,45,55,32,46,61,42,37,43,34,40,25,16,17,17,16,23,18,18,9,7,7,4,3,2,8,3,1,1,2,3,3,2,0,0,2,2,0,6,3,6,2,3,2,4,5,2,9,2,4,8,6,3,11,14,15,20,9,20,28,38,30,30,23,16,22,28,14,17,20,17,10,13,20,9,18,9,8,19,11,4,6,6,8,13,8,8,5,16,12,11,18,10,22,14,16,18,27,38,35,41,51,65,55,54,62,64,56,65,71,75,71,72,47,27,35,25,19,37,38,34,26,19,18,22,16,18,6,12,6,6,3,7,6,1,3,2,2,1,10,3,3,1,1,2,6,3,3,5,4,7,6,5,7,6,4,4,7,9,5,5,10,6,13,6,5,5,9,3,6,11,7,7,15,9,6,6,6,7,10,8,7,12,3,2,7,5,5,7,7,7,7,10,13,10,14,11,20,25,17,18,25,21,31,32,26,35,28,37,41,34,30,39,39,39,34,30,37,29,26,15,22,15,20,14,10,21,14,14,9,11,5,6,7,11,4,3,2,6,10,7,5,3,12,13,10,13,13,8,21,18,8,7,20,14,14,7,14,10,13,27,13,18,16,16,20,17,4,15,8,6,12,15,11,10,15,17,7,7,8,9,12,12,5,4,11,4,5,7,1,1,4,2,6,3,4,10,12,21,26,21,30,45,56,75,83,82,126,119,137,131,112,82,73,43,55,55,53,46,43,29,22,26,13,17,8,13,10,17,19,9,9,9,3,7,7,0,2,3,3,1,3,3,3,7,3,5,11,5,5,6,6,4,4,8,14,12,16,10,16,18,15,23,17,33,15,13,11,14,17,19,20,12,21,7,19,10,13,10,8,21,11,9,14,14,15,18,16,12,20,8,3,13,4,1,10,8,13,10,21,18,21,34,25,34,33,40,42,36,72,75,76,92,71,112,106,101,170,135,106,68,48,48,26,33,29,17,12,13,17,15,14,15,10,9,2,6,8,5,1,2,3,4,3,1,3,5,2,3,2,3,2,2,3,4,3,4,4,4,7,6,15,11,9,9,12,13,13,13,20,28,45,28,34,41,36,38,48,27,23,28,42,30,18,38,28,36,44,41,35,28,28,22,26,24,9,21,10,15]
#data__ = preprocessing.scale(data__)
data__ = data__full[:50]
data__ = np.array(data__,dtype=np.float32)
#data__ = data__/np.max(data__)
data__ = data__.tolist()


train_target = []
train_data = []
test_data = []
test_target = []
for i in range(1,len(data__)):
    train_data.append(data__[i-1])
    train_target.append(data__[i])


train_X = np.array(train_target).reshape((-1,1,1))
train_y = np.array(train_data).reshape((-1))
#train_X = np.ones((19,52))
test_X =  train_X
#train_y = np.ones((19,52))#np.repeat(2,19).reshape((10,1))
test_y = train_y

class GaussWeights(object):
    def __init__(self):
        self.count = 0
    def __call__(self, shape):
        self.count += 1
        return pm.Normal('w%d' % self.count, mu=0, sd=.1,
                         testval=np.random.normal(size=shape).astype(np.float64),
                         shape=shape)

def construct_nn(ann_input, ann_output,init):


    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1))

    l_forward_1 = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh,
    W = init)

    l_out = lasagne.layers.DenseLayer(l_forward_1,  \
        num_units=10,  \
        W = init,
     nonlinearity=lasagne.nonlinearities.linear)

    # Weights from input to hidden layer

    prediction = lasagne.layers.get_output(l_out)




    # Weights from 1st to 2nd layer

    # Binary classification -> Bernoulli likelihood
    out = pm.Normal('out',
                       prediction,
                       observed=ann_output,
                      # total_size=train_y.shape[0] # IMPORTANT for minibatches
                      )
    return out

# Trick: Turn inputs and outputs into shared variables.
# It's still the same thing, but we can later change the values of the shared variable
# (to switch in the test-data later) and pymc3 will just use the new data.
# Kind-of like a pointer we can redirect.
# For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
ann_input = theano.shared(train_X)
ann_output = theano.shared(train_y)


with pm.Model() as neural_network:
    likelihood = construct_nn(ann_input,ann_output,GaussWeights())
    v_params, trace, ppc, y_pred = run_advi(likelihood)

# create symbolic input
w_1 = trace[10:].get_values('w_in_1')
w_2 = trace[10:].get_values('w_1_2')
pred_vals = []

print (np.array(w_1).shape)
for elm in range(len(w_1)):

    pred_vals.append( \
        np.dot(np.tanh(np.dot(data__full[50],w_1[elm])),w_2[elm]))
print (np.array(pred_vals).mean())

pred_vals = []
for elm in range(len(w_1)):


    pred_vals.append( \
        np.dot(np.tanh(np.dot(data__full[1],w_1[elm])),w_2[elm]))
print (np.array(pred_vals).mean())