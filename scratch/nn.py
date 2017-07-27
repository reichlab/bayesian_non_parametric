import numpy
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
import math
from lasagne.updates import adam
from lasagne.utils import collect_shared_vars


def log_gaussian_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma / 2. - (x - mu) ** 2 / (2. * T.exp(logsigma))

def relu(x):
    return max(0, x)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - T.log(T.abs_(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)

def nonlinearity(x):
    return T.nnet.relu(x)

def init(shape):
    return np.asarray(
        np.random.normal(0, 0.05, size=shape),
        dtype=theano.config.floatX
    )
def get_random(shape, avg, std):
    return rnd.normal(shape, avg=avg, std=std)

rnd = RandomStreams(seed=123)
gpu_rnd = MRG_RandomStreams(seed=123)

rng = numpy.random




x = T.matrix("x")
y = T.vector("y")

W1_mu = theano.shared(value=init((1, 1)))
W1_logsigma = theano.shared(value=init((1, 1)))
b1_mu = theano.shared(value=init((1,)))
b1_logsigma = theano.shared(value=init((1,)))


W2_mu = theano.shared(value=init((1, 1)))
W2_logsigma = theano.shared(value=init((1, 1)))
b2_mu = theano.shared(value=init((1,)))
b2_logsigma = theano.shared(value=init((1,)))


all_params = [
        W1_mu, W1_logsigma,b1_mu, b1_logsigma,
        W2_mu, W2_logsigma,b1_mu, b1_logsigma
    ]
all_params = collect_shared_vars(all_params)



sigma_prior = T.exp(-3)

lr = .001
num_samples = 3
sample_std = .001

sample_log_pw, sample_log_qw, sample_log_likelihood = 0., 0., 0.

for i in range(num_samples):
    epsilon_w1 = get_random((1, 1), avg=0., std=sample_std)
    W1 = W1_mu + T.log(1. + T.exp(W1_logsigma)) * epsilon_w1

    epsilon_w2 = get_random((1, 1), avg=0., std=sample_std)
    W2 = W2_mu + T.log(1. + T.exp(W2_logsigma)) * epsilon_w2

    epsilon_b1 = get_random((1,), avg=0., std=sample_std)
    b1 = b1_mu + T.log(1. + T.exp(b1_logsigma)) * epsilon_b1

    epsilon_b2 = get_random((1,), avg=0., std=sample_std)
    b2 = b2_mu + T.log(1. + T.exp(b2_logsigma)) * epsilon_b2

    a1 = nonlinearity(T.dot(x, W1)+b1)
    a2 = T.nnet.sigmoid(nonlinearity(T.dot(a1, W2)+b2))

    print (a2.eval({x:[[100]]}))
    sample_log_pw += log_gaussian(W1, 0., sigma_prior).sum()
    sample_log_pw += log_gaussian(W2, 0., sigma_prior).sum()

    # Still a question if this is right
    sample_log_qw += log_gaussian(W1, W1_mu, T.log(1. + T.exp(W1_logsigma))).sum()
    sample_log_qw += log_gaussian(W2, W2_mu,T.log(1. + T.exp(W2_logsigma))).sum()

    sample_log_likelihood += log_gaussian(y, a2, sigma_prior).sum()

sample_log_qw /= num_samples
sample_log_pw /= num_samples
sample_log_likelihood /= num_samples


objective = ((sample_log_qw - sample_log_pw) - sample_log_likelihood).sum()







updates = adam(objective, all_params, learning_rate=lr)



train = theano.function(
    inputs = [x, y],
    outputs = [objective],
    updates = updates)

predict = theano.function(inputs = [x], outputs = [W1_mu,W1_logsigma,W2_mu,W2_logsigma,b1_mu,b1_logsigma,b2_mu,b2_logsigma],on_unused_input='ignore')



for i in range(1000):
    train([[1]],[1])


W1_mu,W1_logsigma,W2_mu,W2_logsigma,b1_mu,b1_logsigma,b2_mu,b2_logsigma = predict([[1]])

W1 = W1_mu #+ T.log(1. + T.exp(W1_logsigma)) * epsilon_w1

W2 = W2_mu #+ T.log(1. + T.exp(W2_logsigma)) * epsilon_w2


a1 = relu(np.dot([[1]], W1)+b1_mu)
a2 = sigmoid(relu(np.dot(a1, W2)+b2_mu))

print (a2)



