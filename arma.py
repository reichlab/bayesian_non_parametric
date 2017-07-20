
import pymc3 as pm
from theano import scan, shared
import csv
import numpy as np
import theano
from matplotlib import pyplot as plt
import theano.tensor as TT
data = []
with open('../article-disease-pred-with-kcde/data-raw/San_Juan_Training_Data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(int(row['total_cases']))
        # if int(row['season_week']) in data.keys():
        #     data[int(row['season_week'])] += int(row['total_cases'])
        # else: 
        #     data[int(row['season_week'])] = int(row['total_cases'])
print(data)
y = list(set(data))

y = data[:50]/(np.max(data[:50])+10.)


x = y[:len(y)-1]
y = y[1:]
size = len(x)

# Turn inputs and outputs into shared variables so that we can change them later

X = x.reshape((len(x),1))
Y = y

X_train = X#, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
Y_train = Y
print(X_train)
print(Y_train)
ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)

def build_model():
    y = shared(np.array([1, 4, 16, 16*4], dtype=np.float64))
    with pm.Model() as arma_model:
        init_1 = np.random.randn(X.shape[1], 3)
        init_1 = init_1[:, np.argsort(init_1.sum(axis=0))]
        init_2 = np.random.randn(3)
        init_2 = init_2[np.argsort(init_2)]
        init_3 = np.random.randn(3,3)

        W_in = pm.Normal('w_in_1', 0, sd=1, shape=(X.shape[1], 3), 
                             testval=init_1)
        W_out = pm.Normal('w_1_out', 0, sd=1, shape=(3,), 
                              testval=init_2)
        W = pm.Normal('weights_hidden',0,sd=1,shape=(3,3),testval=init_3)

        sigma = pm.HalfCauchy('sigma', 5.)
        theta = pm.Normal('theta', 0., sd=2.)
        phi = pm.Normal('phi', 0., sd=2.)
        mu = pm.Normal('mu', 0., sd=10.)
        h0_init = np.zeros((3,), dtype=np.float64)
        h0 = theano.shared(value=h0_init, name='h0')

        def step(u_t, h_tm1, W, W_in, W_out):
          h_t = TT.tanh(TT.dot(u_t, W_in) + TT.dot(h_tm1, W))
          y_t = TT.dot(h_t, W_out)
          return h_t, y_t

        [h, y], _ = theano.scan(step,
                        sequences=ann_input,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])

        
        likelihood = pm.Normal('y', mu=y,
                        sd=.1, observed=ann_output)
    return arma_model


def run(n_samples=100):
    model = build_model()
    with model:
        s = theano.shared(pm.floatX(1))
        inference = pm.ADVI(cost_part_grad_scale=s)
        # ADVI has nearly converged
        pm.fit(n=20000, method=inference)
        # It is time to set `s` to zero
        s.set_value(0)
        approx = pm.fit(n=30000)
        trace = approx.sample(draws=5000)
    print (trace['w_in_1'].mean(axis=0))
    h_t = np.ones((3,), dtype=np.float64)
    pred =[]
    for elm in X_train:
        h_t = np.tanh(np.dot(elm, trace['w_in_1'].mean(axis=0)) + np.dot(h_t,trace['weights_hidden'].mean(axis=0)))
       
        y_t = np.dot(h_t, trace['w_1_out'].mean(axis=0))
        pred.append(y_t)
    fig, ax = plt.subplots(figsize=(14,5))
    ax.scatter(X_train,pred,color='g')
    ax.scatter(X_train,Y_train,color='r')

    burn = n_samples // 10
    pm.plots.traceplot(trace[burn:])
    plt.show()

if __name__ == '__main__':
    run()

