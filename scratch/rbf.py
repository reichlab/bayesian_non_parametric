# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:09:31 2016

@author: Rob Romijnders
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
#Load the MNIST data
#X_train and y_train refers to the usual 60.000 by 784 matrix and 60.000 vector
#X_test and y_test refers to the usual 10.000 by 784 and 10.000 vector

data__full = [4,5,4,3,6,2,4,5,10,6,8,2,6,17,23,13,21,28,24,20,40,27,42,33,43,37,57,71,44,56,53,52,47,26,27,21,21,26,34,37,17,19,25,18,21,17,17,16,16,15,23,16,17,12,17,10,15,19,21,14,18,13,14,18,23,25,62,60,76,66,64,68,89,92,140,116,142,129,140,140,127,129,169,141,108,78,70,81,104,90,85,55,53,65,33,38,59,40,37,29,30,30,28,23,24,29,26,23,20,19,20,26,29,31,28,26,32,35,33,30,52,59,67,65,74,70,61,53,76,61,57,44,34,47,60,60,53,36,31,30,32,28,33,33,35,22,13,13,21,17,11,8,8,6,6,7,12,17,10,10,18,19,12,22,12,21,18,16,16,22,17,25,23,12,25,28,27,18,23,23,29,38,36,43,46,31,25,40,31,38,30,22,31,26,35,36,39,25,31,37,33,25,24,18,23,13,18,14,17,22,13,24,31,34,31,31,38,49,42,49,55,80,84,72,89,115,179,202,272,302,395,426,461,381,333,353,410,364,359,288,221,149,112,154,91,72,56,46,37,26,17,17,20,11,7,16,14,16,5,2,6,5,4,3,4,16,8,7,10,14,7,9,11,23,17,19,24,17,28,40,33,31,33,29,30,36,48,40,28,36,19,34,23,17,17,23,14,20,13,23,20,16,16,23,14,15,4,5,5,11,11,7,4,6,5,2,4,2,4,6,6,4,6,11,16,9,12,13,27,21,19,17,24,27,30,29,25,35,33,30,29,31,29,22,27,24,26,29,22,33,24,30,20,17,24,28,18,13,9,14,11,11,19,10,8,8,9,3,7,14,4,9,14,7,9,3,3,14,12,10,21,26,47,42,31,34,33,52,56,70,112,70,47,48,49,66,56,61,67,64,68,49,50,56,75,63,62,41,50,34,31,38,30,32,26,30,36,35,46,48,44,51,59,71,102,128,127,150,191,256,329,263,220,204,181,99,54,80,102,127,73,68,64,55,67,84,85,67,73,89,68,59,56,77,75,47,50,42,28,37,37,27,12,15,22,8,15,17,10,9,11,20,13,11,16,11,7,17,14,13,15,30,25,40,44,25,21,48,56,60,45,55,32,46,61,42,37,43,34,40,25,16,17,17,16,23,18,18,9,7,7,4,3,2,8,3,1,1,2,3,3,2,0,0,2,2,0,6,3,6,2,3,2,4,5,2,9,2,4,8,6,3,11,14,15,20,9,20,28,38,30,30,23,16,22,28,14,17,20,17,10,13,20,9,18,9,8,19,11,4,6,6,8,13,8,8,5,16,12,11,18,10,22,14,16,18,27,38,35,41,51,65,55,54,62,64,56,65,71,75,71,72,47,27,35,25,19,37,38,34,26,19,18,22,16,18,6,12,6,6,3,7,6,1,3,2,2,1,10,3,3,1,1,2,6,3,3,5,4,7,6,5,7,6,4,4,7,9,5,5,10,6,13,6,5,5,9,3,6,11,7,7,15,9,6,6,6,7,10,8,7,12,3,2,7,5,5,7,7,7,7,10,13,10,14,11,20,25,17,18,25,21,31,32,26,35,28,37,41,34,30,39,39,39,34,30,37,29,26,15,22,15,20,14,10,21,14,14,9,11,5,6,7,11,4,3,2,6,10,7,5,3,12,13,10,13,13,8,21,18,8,7,20,14,14,7,14,10,13,27,13,18,16,16,20,17,4,15,8,6,12,15,11,10,15,17,7,7,8,9,12,12,5,4,11,4,5,7,1,1,4,2,6,3,4,10,12,21,26,21,30,45,56,75,83,82,126,119,137,131,112,82,73,43,55,55,53,46,43,29,22,26,13,17,8,13,10,17,19,9,9,9,3,7,7,0,2,3,3,1,3,3,3,7,3,5,11,5,5,6,6,4,4,8,14,12,16,10,16,18,15,23,17,33,15,13,11,14,17,19,20,12,21,7,19,10,13,10,8,21,11,9,14,14,15,18,16,12,20,8,3,13,4,1,10,8,13,10,21,18,21,34,25,34,33,40,42,36,72,75,76,92,71,112,106,101,170,135,106,68,48,48,26,33,29,17,12,13,17,15,14,15,10,9,2,6,8,5,1,2,3,4,3,1,3,5,2,3,2,3,2,2,3,4,3,4,4,4,7,6,15,11,9,9,12,13,13,13,20,28,45,28,34,41,36,38,48,27,23,28,42,30,18,38,28,36,44,41,35,28,28,22,26,24,9,21,10,15]
from sklearn.cluster import KMeans
# data__full = scale(data__full)
data__ = data__full[:50]
# print (data__full[50])
# data__ = data__full[:50]
# data__ = np.array(data__,dtype=np.float32)
# data__ = data__.tolist()


m = 1
train_target = []
train_data = []
test_data = []
test_target = []
for i in range(len(data__)):
    train_data.append(data__[i-1])
    train_target.append(data__[i])

print (np.array(train_data).shape)

X_train = X_test = np.array(train_data,dtype=np.float32).reshape((-1,1))
y_train = y_test =  np.array(train_target,dtype=np.float32).reshape((-1))

kmeans = KMeans(init='k-means++', n_clusters=len(y_train)-2, n_init=10)
kmeans.fit(np.array(y_train).reshape((-1,1)))
centroids = kmeans.cluster_centers_.reshape((-1,))
print (centroids)

"""Hyper-parameters"""
batch_size = 1            # Batch size for stochastic gradient descent
test_size = batch_size      # Temporary heuristic. In future we'd like to decouple testing from batching
num_centr = len(centroids)             # Number of "hidden neurons" that is number of centroids
max_iterations = 100000       # Max number of iterations
learning_rate = 1e-3      # Learning rate
num_classes = 1            # Number of target classes, 10 for MNIST
var_rbf = 225            # What variance do you expect workable for the RBF?

#Obtain and proclaim sizes
N,D = X_train.shape
Ntest = X_test.shape[0]
print('We have %s observations with %s dimensions'%(N,D))

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

#Placeholders for data
x = tf.placeholder('float',shape=[batch_size,D],name='input_data')
y_ = tf.placeholder('float', shape=[batch_size], name = 'Ground_truth')


with tf.name_scope("Hidden_layer") as scope:
  #Centroids and var are the main trainable parameters of the first layer
  centroids = tf.constant(centroids.reshape((-1,1)),dtype=tf.float32,name='centroids')
  var = tf.Variable(tf.truncated_normal([num_centr],mean=var_rbf,stddev=5,dtype=tf.float32),name='RBF_variance')

  #For now, we collect the distanc
  exp_list = []
  for i in range(num_centr):
        exp_list.append(tf.exp((-1*tf.reduce_sum(tf.square(tf.subtract(x,centroids[i,:])),1))/(2*var[i])))
        phi = tf.transpose(tf.stack(exp_list))

with tf.name_scope("Output_layer") as scope:
    w = tf.Variable(tf.truncated_normal([num_centr,num_classes], stddev=.1, dtype=tf.float32),name='weight')
    bias = tf.Variable( tf.constant(0.1, shape=[num_classes]),name='bias')

    h = tf.matmul(phi,w)+bias
    size2 = tf.shape(h)

with tf.name_scope("Softmax") as scope:
  loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(h, y_)))
  cost = tf.reduce_sum(loss)
  loss_summ = tf.summary.scalar("cross entropy_loss", cost)

with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)
#     The following block plots for every trainable variable
#      - Histogram of the entries of the Tensor
#      - Histogram of the gradient over the Tensor
#      - Histogram of the grradient-norm over the Tensor
    numel = tf.constant([[0]])
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient

      numel +=tf.reduce_sum(tf.size(variable))

      h1 = tf.histogram_summary(variable.name, variable)
      h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
      h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
with tf.name_scope("Evaluating") as scope:
    accuracy =tf.sqrt(tf.reduce_mean(tf.squared_difference(h, y_)))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()


# For now, we collect performances in a Numpy array.
# In future releases, I hope TensorBoard allows for more
# flexibility in plotting
perf_collect = np.zeros((4,int(np.floor(max_iterations /100))))

with tf.Session() as sess:
  with tf.device("/cpu:0"):
    print('Start session')
   # writer = tf.train.SummaryWriter("./, sess.graph_def)

    step = 0
    sess.run(tf.initialize_all_variables())

#    #Debugging
#    batch_ind = np.random.choice(N,batch_size,replace=False)
#    result = sess.run([phi],feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind]})
#    print(result[0])


    for i in range(max_iterations):
      batch_ind = np.random.choice(N,batch_size,replace=False)
      if i%100 == 1:
        #Measure train performance
        result = sess.run([cost,accuracy,train_step,h],feed_dict={x:np.array(data__full[0]).reshape((-1,1)), y_: np.array(data__full[1]).reshape((1,))})

        acc = result[1]
        print("Estimated accuracy at iteration %s of %s: %s" % (i,max_iterations, acc))
        print (result[3],data__full[1],data__full[0])
        step += 1
      else:
        sess.run(train_step,feed_dict={x:np.array([X_train[batch_ind]]).reshape((-1,1)) \
          , y_: np.array([y_train[batch_ind]]).reshape((-1))})

"""Additional plots"""


# We can now open TensorBoard. Run the following line from your terminal
# tensorboard --logdir=/home/rob/Dropbox/ml_projects/RBFN_tf/log_tb