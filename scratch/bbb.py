import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from lasagne.updates import adam
from lasagne.utils import collect_shared_vars

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import sys
import numpy as np

data__ = [4,5,4,3,6,2,4,5,10,6,8,2,6,17,23,13,21,28,24,20,40,27,42,33,43,37,57,71,44,56,53,52,47,26,27,21,21,26,34,37,17,19,25,18,21,17,17,16,16,15,23,16,17,12,17,10,15,19,21,14,18,13,14,18,23,25,62,60,76,66,64,68,89,92,140,116,142,129,140,140,127,129,169,141,108,78,70,81,104,90,85,55,53,65,33,38,59,40,37,29,30,30,28,23,24,29,26,23,20,19,20,26,29,31,28,26,32,35,33,30,52,59,67,65,74,70,61,53,76,61,57,44,34,47,60,60,53,36,31,30,32,28,33,33,35,22,13,13,21,17,11,8,8,6,6,7,12,17,10,10,18,19,12,22,12,21,18,16,16,22,17,25,23,12,25,28,27,18,23,23,29,38,36,43,46,31,25,40,31,38,30,22,31,26,35,36,39,25,31,37,33,25,24,18,23,13,18,14,17,22,13,24,31,34,31,31,38,49,42,49,55,80,84,72,89,115,179,202,272,302,395,426,461,381,333,353,410,364,359,288,221,149,112,154,91,72,56,46,37,26,17,17,20,11,7,16,14,16,5,2,6,5,4,3,4,16,8,7,10,14,7,9,11,23,17,19,24,17,28,40,33,31,33,29,30,36,48,40,28,36,19,34,23,17,17,23,14,20,13,23,20,16,16,23,14,15,4,5,5,11,11,7,4,6,5,2,4,2,4,6,6,4,6,11,16,9,12,13,27,21,19,17,24,27,30,29,25,35,33,30,29,31,29,22,27,24,26,29,22,33,24,30,20,17,24,28,18,13,9,14,11,11,19,10,8,8,9,3,7,14,4,9,14,7,9,3,3,14,12,10,21,26,47,42,31,34,33,52,56,70,112,70,47,48,49,66,56,61,67,64,68,49,50,56,75,63,62,41,50,34,31,38,30,32,26,30,36,35,46,48,44,51,59,71,102,128,127,150,191,256,329,263,220,204,181,99,54,80,102,127,73,68,64,55,67,84,85,67,73,89,68,59,56,77,75,47,50,42,28,37,37,27,12,15,22,8,15,17,10,9,11,20,13,11,16,11,7,17,14,13,15,30,25,40,44,25,21,48,56,60,45,55,32,46,61,42,37,43,34,40,25,16,17,17,16,23,18,18,9,7,7,4,3,2,8,3,1,1,2,3,3,2,0,0,2,2,0,6,3,6,2,3,2,4,5,2,9,2,4,8,6,3,11,14,15,20,9,20,28,38,30,30,23,16,22,28,14,17,20,17,10,13,20,9,18,9,8,19,11,4,6,6,8,13,8,8,5,16,12,11,18,10,22,14,16,18,27,38,35,41,51,65,55,54,62,64,56,65,71,75,71,72,47,27,35,25,19,37,38,34,26,19,18,22,16,18,6,12,6,6,3,7,6,1,3,2,2,1,10,3,3,1,1,2,6,3,3,5,4,7,6,5,7,6,4,4,7,9,5,5,10,6,13,6,5,5,9,3,6,11,7,7,15,9,6,6,6,7,10,8,7,12,3,2,7,5,5,7,7,7,7,10,13,10,14,11,20,25,17,18,25,21,31,32,26,35,28,37,41,34,30,39,39,39,34,30,37,29,26,15,22,15,20,14,10,21,14,14,9,11,5,6,7,11,4,3,2,6,10,7,5,3,12,13,10,13,13,8,21,18,8,7,20,14,14,7,14,10,13,27,13,18,16,16,20,17,4,15,8,6,12,15,11,10,15,17,7,7,8,9,12,12,5,4,11,4,5,7,1,1,4,2,6,3,4,10,12,21,26,21,30,45,56,75,83,82,126,119,137,131,112,82,73,43,55,55,53,46,43,29,22,26,13,17,8,13,10,17,19,9,9,9,3,7,7,0,2,3,3,1,3,3,3,7,3,5,11,5,5,6,6,4,4,8,14,12,16,10,16,18,15,23,17,33,15,13,11,14,17,19,20,12,21,7,19,10,13,10,8,21,11,9,14,14,15,18,16,12,20,8,3,13,4,1,10,8,13,10,21,18,21,34,25,34,33,40,42,36,72,75,76,92,71,112,106,101,170,135,106,68,48,48,26,33,29,17,12,13,17,15,14,15,10,9,2,6,8,5,1,2,3,4,3,1,3,5,2,3,2,3,2,2,3,4,3,4,4,4,7,6,15,11,9,9,12,13,13,13,20,28,45,28,34,41,36,38,48,27,23,28,42,30,18,38,28,36,44,41,35,28,28,22,26,24,9,21,10,15]
data__ = np.array(data__,dtype=np.float32)/100

rnd = RandomStreams(seed=123)
gpu_rnd = MRG_RandomStreams(seed=123)


def nonlinearity(x):
    return T.nnet.relu(x)


def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - T.log(T.abs_(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma / 2. - (x - mu) ** 2 / (2. * T.exp(logsigma))


def _shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, shared_y


def init(shape):
    return np.asarray(
        np.random.normal(0, 0.05, size=shape),
        dtype=theano.config.floatX
    )


def get_random(shape, avg, std):
    return gpu_rnd.normal(shape, avg=avg, std=std)


if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    # prepare data
    N = 5000

    data = np.float32(mnist.data[:]) / 255.
    idx = np.random.choice(data.shape[0], N)
    data = data[idx]
    target = np.int32(mnist.target[idx]).reshape(N, 1)




    train_target = []
    train_data = []
    test_data = []
    test_target = []
    for i in range(0,len(data__),4):
        train_data.append(data__[i:i+4])
        train_target.append(data__[i+1])

    train_data = np.array(train_data)/100
    train_target = np.array(train_target).reshape((len(train_target),1))/100

    test_data = train_data
    test_target = test_target
   
    # inputs
    x = T.matrix('x')
    y = T.matrix('y')
    n_input = train_data.shape[1]
    M = train_data.shape[0]
    sigma_prior_1 = T.exp(4)
    sigma_prior  = T.exp(2)
    sigma_prior_2 = T.exp(-1)

    
    scale_prior = np.float32(0.5)
    n_samples = 4
    learning_rate = 0.001
    n_epochs = 500

    # weights
    # L1
    n_hidden_1 = 10
    W1_mu = theano.shared(value=init((n_input, n_hidden_1)))
    W1_logsigma = theano.shared(value=init((n_input, n_hidden_1)))
    b1_mu = theano.shared(value=init((n_hidden_1,)))
    b1_logsigma = theano.shared(value=init((n_hidden_1,)))

    # L2
   
    # L3
    n_output = 1
    W3_mu = theano.shared(value=init((n_hidden_1, n_output)))
    W3_logsigma = theano.shared(value=init((n_hidden_1, n_output)))
    b3_mu = theano.shared(value=init((n_output,)))
    b3_logsigma = theano.shared(value=init((n_output,)))

    all_params = [
        W1_mu, W1_logsigma, b1_mu, b1_logsigma,
        W3_mu, W3_logsigma, b3_mu, b3_logsigma
    ]
    all_params = collect_shared_vars(all_params)

    # building the objective
    # remember, we're evaluating by samples
    log_pw, log_qw, log_likelihood = 0., 0., 0.

    for _ in xrange(n_samples):

        epsilon_w1 = get_random((n_input, n_hidden_1), avg=0., std=1)
        epsilon_b1 = get_random((n_hidden_1,), avg=0., std=1)

        W1 = W1_mu + T.log(1. + T.exp(W1_logsigma)) * epsilon_w1
        b1 = b1_mu + T.log(1. + T.exp(b1_logsigma)) * epsilon_b1

       

        epsilon_w3 = get_random((n_hidden_1, n_output), avg=0., std=1)
        epsilon_b3 = get_random((n_output,), avg=0., std=1)

        W3 = W3_mu + T.log(1. + T.exp(W3_logsigma)) * epsilon_w3
        b3 = b3_mu + T.log(1. + T.exp(b3_logsigma)) * epsilon_b3

        a1 = nonlinearity(T.dot(x, W1) + b1)
        h = T.nnet.sigmoid(nonlinearity(T.dot(a1, W3) + b3))

        sample_log_pw, sample_log_qw, sample_log_likelihood = 0., 0., 0.

        for W, b, W_mu, W_logsigma, b_mu, b_logsigma in [(W1, b1, W1_mu, W1_logsigma, b1_mu, b1_logsigma),
                                                         (W3, b3, W3_mu, W3_logsigma, b3_mu, b3_logsigma)]:

            # first weight prior
            sample_log_pw += scale_prior*log_gaussian(W, 0., sigma_prior_1).sum()+ \
                            (1-scale_prior)*log_gaussian(W, 0., sigma_prior_2).sum()
            
            sample_log_pw += scale_prior*log_gaussian(b, 0., sigma_prior_1).sum()+ \
                            (1-scale_prior)*log_gaussian(b, 0., sigma_prior_2).sum()

            # then approximation
            sample_log_qw += log_gaussian_logsigma(W, W_mu, W_logsigma * 2).sum()
            sample_log_qw += log_gaussian_logsigma(b, b_mu, b_logsigma * 2).sum()

        # then the likelihood
        sample_log_likelihood = log_gaussian(y, h, sigma_prior).sum() 

        log_pw += sample_log_pw
        log_qw += sample_log_qw
        log_likelihood += sample_log_likelihood

    log_qw /= n_samples
    log_pw /= n_samples
    log_likelihood /= n_samples

    batch_size = 1
    n_batches = M / float(batch_size)

    objective = ((1. / n_batches) * (log_qw - log_pw) - log_likelihood).sum() / float(batch_size)

    # updates

    updates = adam(objective, all_params, learning_rate=learning_rate)

    i = T.iscalar()

    train_data = theano.shared(np.asarray(train_data, dtype=theano.config.floatX))
    train_target = theano.shared(np.asarray(train_target, dtype=theano.config.floatX))

    train_function = theano.function(
        inputs=[i],
        outputs=objective,
        updates=updates,
        givens={
            x: train_data[i * batch_size: (i + 1) * batch_size],
            y: train_target[i * batch_size: (i + 1) * batch_size]
        }
    )

    # a1_mu = nonlinearity(T.dot(x, W1_mu) + b1_mu)
    # a2_mu = nonlinearity(T.dot(a1_mu, W2_mu) + b2_mu)
    # h_mu = T.nnet.sigmoid(nonlinearity(T.dot(a2_mu, W3_mu) + b3_mu))

   


    output_function = theano.function([x], [h])
    n_train_batches = int(train_data.get_value().shape[0] / float(batch_size))

    # and finally, training loop
    for e in xrange(n_epochs):
        errs = []
        for b in xrange(n_train_batches):
            batch_err = train_function(b)
            errs.append(batch_err)
    n_posterior_samples = 200
    samples = []
    for posterior_sample in xrange(n_posterior_samples):    
        out = output_function([[.04,.05,.04,.03]])
        print (out)
        out_mean = np.array(out)[0][0]
        samples.append(out_mean)
    print (np.mean(samples))
    print (np.std(samples))


