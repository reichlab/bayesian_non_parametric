#!/usr/bin/env python
"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
This example prettifies some of the tensor naming for visualization in
TensorBoard. To view TensorBoard, run `tensorboard --logdir=log`.

References
----------
http://edwardlib.org/tutorials/bayesian-neural-network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Gamma
import csv

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

X = np.array(x,dtype=np.float64).reshape((len(x),1))
y = np.array(y,dtype=np.float64)


def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
 
  X = range(N)
  print (X)
  y= np.sin(X)
  X = np.array(X).reshape((N,D))
  return X, y


def neural_network(X):
  h = tf.nn.tanh(tf.matmul(X, W_0) + b_0)
  h = tf.nn.tanh(tf.matmul(h, W_1) + b_1)
  h = tf.matmul(h, W_2) + b_2
  return tf.reshape(h, [-1])


ed.set_seed(42)

N = len(X)  # number of data points
D = 1   # number of features

# DATA
X_train, y_train = X,y
num_hidden = 1
# MODEL
with tf.name_scope("model"):
  W_0 = Normal(loc=tf.zeros([D, num_hidden]), scale=tf.ones([D, num_hidden]), name="W_0")
  W_1 = Normal(loc=tf.zeros([num_hidden, num_hidden]), scale=tf.ones([num_hidden, num_hidden]), name="W_1")
  W_2 = Normal(loc=tf.zeros([num_hidden, 1]), scale=tf.ones([num_hidden, 1]), name="W_2")
  b_0 = Normal(loc=tf.zeros(num_hidden), scale=tf.ones(num_hidden), name="b_0")
  b_1 = Normal(loc=tf.zeros(num_hidden), scale=tf.ones(num_hidden), name="b_1")
  b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")
  X = tf.placeholder(tf.float32, [N, D], name="X")
  y = Normal(loc=neural_network(X), scale=.1*tf.ones(N), name="y")

# INFERENCE
with tf.name_scope("posterior"):
  with tf.name_scope("qW_0"):
    qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, num_hidden]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([D, num_hidden]), name="scale")))
  with tf.name_scope("qW_1"):
    qW_1 = Normal(loc=tf.Variable(tf.random_normal([num_hidden, num_hidden]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([num_hidden, num_hidden]), name="scale")))
  with tf.name_scope("qW_2"):
    qW_2 = Normal(loc=tf.Variable(tf.random_normal([num_hidden, 1]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([num_hidden, 1]), name="scale")))
  with tf.name_scope("qb_0"):
    qb_0 = Normal(loc=tf.Variable(tf.random_normal([num_hidden]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([num_hidden]), name="scale")))
  with tf.name_scope("qb_1"):
    qb_1 = Normal(loc=tf.Variable(tf.random_normal([num_hidden]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([num_hidden]), name="scale")))
  with tf.name_scope("qb_2"):
    qb_2 = Normal(loc=tf.Variable(tf.random_normal([1]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([1]), name="scale")))

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
inference.run(logdir='log',n_iter=1000)
X_train= X_train.astype(np.float32)
# fig, ax = plt.subplots()

# ax.scatter(X_train,neural_network(X_train).eval())
# ax.scatter(X_train,y_train,color='r')
def relu_ (x):
  return np.maximum(x, 0)
fig, ax = plt.subplots(figsize=(14,5))

def visualise(X_data, y_data, W_0,W_1,W_2, b_0,b_1,b_2, n_samples=10):
  W_0 = qW_0.sample(n_samples).eval()
  W_1 = qW_1.sample(n_samples).eval()
  W_2 = qW_2.sample(n_samples).eval()
  b_0 = qb_0.sample(n_samples).eval()
  b_1 = qb_1.sample(n_samples).eval()
  b_2 = qb_2.sample(n_samples).eval()
  print (W_0.shape)
  for ns in range(n_samples):
      h = np.tanh(np.matmul(X_train, W_0[ns]) + b_0[ns])
      h = np.tanh(np.matmul(h, W_1[ns]) + b_1[ns])
      h = np.matmul(h, W_2[ns]) + b_2[ns]
      output= h
      ax.plot(X_train, output, alpha=0.1)

  h = np.tanh(np.matmul(X_train, W_0.mean(axis=0)) + b_0.mean(axis=0))
  h = np.tanh(np.matmul(h, W_1.mean(axis=0)) + b_1.mean(axis=0))
  h = np.matmul(h, W_2.mean(axis=0)) + b_2.mean(axis=0)
  output= h
  ax.plot(X_train, output, alpha=0.5,color='g')

visualise(X_train,y_train, W_0,W_1,W_2, b_0,b_1,b_2,200)
ax.scatter(X_train,y_train,color='r')
ax.set_ylim([-1,1])
plt.show()
