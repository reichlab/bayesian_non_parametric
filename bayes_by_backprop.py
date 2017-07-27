import os
os.environ["THEANO_FLAGS"] = "optimizer=None"
import theano
from theano import tensor as T,printing 
import numpy as np
from scipy.misc import imsave
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nlinalg import matrix_inverse, det
from matplotlib import pyplot as plt



srng = RandomStreams(seed=234)

def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - T.log(T.abs_(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)

def log_gaussian_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma / 2. - (x - mu) ** 2 / (2. * T.exp(logsigma))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h, w_o, mu, p):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.sigmoid(T.dot(h, w_o))
    return pyx


'''Training data'''
sigma_prior_val = .6
sigma_prior = T.constant(sigma_prior_val)
sigma_prior_vec =  T.constant(np.repeat(sigma_prior_val,10).reshape((10,1)))

X = T.matrix()
Y = T.fmatrix()

'''Variational Params'''
mu = init_weights((10,1))
p = init_weights((10,1))
rv_n = srng.normal((10,1))
w = T.add(mu,T.log(T.add(np.ones((10,1)),T.exp(p)))*rv_n  )


'''Weight matrices'''
n = 10
h0 = np.zeros((n,), dtype=theano.config.floatX)
# w_h = init_weights((1, 100))
# w_o = init_weights((100, 1))

nin = 1
nout = 1
W = theano.shared(np.random.uniform(size=(n, n), low=-.01, high=.01))
# input to hidden layer weights
W_in = theano.shared(np.random.uniform(size=(nin, n), low=-.01, high=.01))
# hidden to output layer weights
W_out = theano.shared(np.random.uniform(size=(n, nout), low=-.01, high=.01))



def step(u_t, h_tm1, W, W_in, W_out):
    h_t = T.tanh(T.dot(u_t, W_in) + T.dot(h_tm1, W))
    y_t = T.dot(h_t, W_out)
    return h_t, y_t

[h, y], _ = theano.scan(step,
                        sequences=X,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])



'''Cost function'''

cost = log_gaussian(w,mu,T.log(1+T.exp(p)))-log_gaussian(w,0,sigma_prior)- \
log_gaussian(Y, y, sigma_prior_vec)
cost =cost.sum()


params = [W, W_in,W_out,mu,p,w]
updates = sgd(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[y,p], allow_input_downcast=True)

trX = np.arange(50,60,1,dtype=np.float32)
trY = np.add(trX[:5],np.random.normal(0,.01,5))
trY = np.concatenate([trY,np.add(trX[5:],np.random.normal(0,200,5))],axis=0)

trX = trX.reshape((10,1))/np.max(trX)
trY = trY.reshape((10,1))/np.max(trY)

print (trX)
print (trY)

fig, ax = plt.subplots(figsize=(14,5))

for i in range(100):
    for example in range(len(trX)):
        cost = train(trX, trY)
output = predict(trX)
mean = output[0]
stdv = np.log(1+np.exp(output[1]))
ax.plot(trX,mean,color='b')
ax.plot(trX,mean+2*stdv,alpha=.5)
ax.plot(trX,mean-2*stdv,alpha=.5)
ax.plot(trX,trY,color='r')
ax.set_ylim([-5,5])
plt.show()
print (stdv)
print (mean)
print (trY)
