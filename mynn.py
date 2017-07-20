
import pymc3 as pm
import theano.tensor as T
import theano
import sklearn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
import csv
from matplotlib import pyplot as plt


data = []
with open('../article-disease-pred-with-kcde/data-raw/San_Juan_Training_Data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(int(row['total_cases']))
        # if int(row['season_week']) in data.keys():
        #     data[int(row['season_week'])] += int(row['total_cases'])
        # else: 
        #     data[int(row['season_week'])] = int(row['total_cases'])


y = data[:50]/(np.max(data[:50])+10.)


x = y[:len(y)-1]
y = y[1:]
size = len(x)

# Turn inputs and outputs into shared variables so that we can change them later

X = x.reshape((len(x),1))
Y = y

X_train = X#, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
Y_train = Y
ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)
with pm.Model() as model:
    # Below we require an ordering of the summed weights, thus initialize in this order
    init_1 = np.random.randn(X.shape[1], 3)
    init_1 = init_1[:, np.argsort(init_1.sum(axis=0))]
    init_2 = np.random.randn(3)
    init_2 = init_2[np.argsort(init_2)]
    
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, shape=(X.shape[1], 3), 
                             testval=init_1)
    # Weights from hidden layer to output
    weights_1_out = pm.Normal('w_1_out', 0, sd=1, shape=(3,), 
                              testval=init_2)

    tau = pm.Gamma('tau', 1., 10.,shape=len(X_train))
    
    # Do forward pass
    a1 = T.dot(ann_input, weights_in_1)
    act_1 = T.nnet.sigmoid(a1)
    act_out = T.dot(act_1, weights_1_out)
    
    out = pm.Normal('out', 
                       T.nnet.sigmoid(act_out),tau,
                       observed=ann_output)
    
    step = pm.Metropolis()
    trace = pm.sample(50000, step=step)

pm.traceplot(trace);

ann_input.set_value(X_train)
ann_output.set_value(Y_train)

# Creater posterior predictive samples
ppc = pm.sample_ppc(trace, model=model, samples=500)
pred = ppc['out'].mean(axis=0)
tau_pred = trace['tau'].mean(axis=0)
fig, ax = plt.subplots(figsize=(14,5))
ax.scatter(	X,pred,color='g')
ax.scatter(X_train,Y_train,color='r')
ax.scatter(X_train,pred + 2*tau_pred,color='b')
ax.scatter(X_train,pred - 2*tau_pred,color='b')
plt.show()
