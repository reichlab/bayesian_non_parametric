import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
from scipy.stats import norm

y_train = [1.4343897,0.5812728,0.9752642,1.4435222,0.6272415,1.5350055,2.0524251,1.2446703,1.5784264,1.5093078,1.8227201,1.3765860,0.9835593,0.5172897,1.4271145,1.1080890,0.7005765,0.5761012,2.0908005,0.9083144,0.9019680,1.0374890,0.8963419,0.8305956,0.7956729,1.0545629,0.9676514,1.0538474,1.2730516,0.5868709,1.7001184,0.9927210,0.7485983,0.9294269,1.7908196,0.4839141,0.6834663,1.6581032,3.4185652,1.0376642,0.3371482,0.8802055,0.9008538,0.4976946,1.0424277,0.6954531,0.5228425,1.7017606,1.1481733,0.6298479,0.7402941,1.1044740,0.7633736,0.6171344,0.6945606,0.8185196,1.2602043,1.4010858,0.6516594,0.4420171,0.6798522,0.5616979,1.1263608,0.8987007,2.3480708,1.3227814,1.0247199,1.5276863,0.9292929,2.5689573,1.2780135,0.8798412,0.7931086,1.1129425,1.8297001,0.7723241,1.3804836,1.0613372,0.4279842,1.7995757,0.6103783,1.2176640,0.6652558,1.0979659,1.6579506,1.5791826,2.0137412,0.7833415,1.1337972,1.1636457,1.7979757,0.5159113,1.0804260,2.1043583,1.3408635,0.6234596,1.3600895,0.5528448,2.4898298,1.3100504]
x_train = [1.4045492,0.6989161,0.9510627,1.7047190,0.9683409,2.4035547,0.7511683,1.5551469,0.6762053,1.1855954,1.8234196,0.2926205,1.0338832,0.4521064,1.1829740,0.8404029,0.7165140,0.6147296,1.3436606,0.6774859,0.6910263,1.1280514,1.0169330,0.8530184,0.9170171,0.9207020,1.6339146,2.4014797,0.6428915,0.9950586,1.5993236,0.7186894,0.5687548,1.3317113,0.8412805,0.8175485,2.2827293,1.1704571,1.3758034,1.0465211,0.9972902,1.3716838,0.6504551,0.6837009,0.9788499,0.7224157,0.7206308,0.5937835,1.1538085,1.2950379,0.5737379,0.5691650,1.0710501,1.3786235,0.7659387,1.1334639,0.7641966,0.8704595,0.7241809,0.9249249,0.7405532,0.7505255,1.6347606,2.1579407,1.6124271,2.9235001,1.0619669,0.7244556,1.8378322,1.4092371,0.5908489,0.7636589,0.7223787,0.7574973,1.5911549,1.0282644,1.1037701,1.5694230,0.5670305,0.8590597,0.8414620,1.4299847,0.7062031,0.6574490,1.6164585,1.3403497,0.9925906,1.0067950,0.7134453,2.1198477,2.4242799,0.6321120,1.2720978,1.8080349,1.5424188,1.5606643,1.9575598,0.4620136,1.5758459,2.7052811]

x_train = np.array(x_train).reshape((100,1))
y_train = np.array(y_train).reshape((100,1))

NHIDDEN = 24
STDEV = 0.5
KMIX = 2 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer,Wo) + bo

def get_mixture_coef(output):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")

  out_pi, out_sigma, out_mu = tf.split(output, 3, 1)

  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.subtract(out_pi, max_pi)

  out_pi = tf.exp(out_pi)

  normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
  out_pi = tf.multiply(normalize_pi, out_pi)

  out_sigma = tf.exp(out_sigma)

  return out_pi, out_sigma, out_mu

out_pi, out_sigma, out_mu = get_mixture_coef(output)

oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
def tf_normal(y, mu, sigma):
  result = tf.subtract(y, mu)
  result = tf.multiply(result,tf.reciprocal(sigma))
  result = -tf.square(result)/2
  return tf.multiply(tf.exp(result),tf.reciprocal(sigma))*oneDivSqrtTwoPI

def get_lossfunc(out_pi, out_sigma, out_mu, y):
  result = tf_normal(y, out_mu, out_sigma)
  result = tf.multiply(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  result = -tf.log(result)
  return tf.reduce_mean(result)

lossfunc = get_lossfunc(out_pi, out_sigma, out_mu, y)
train_op = tf.train.AdamOptimizer().minimize(lossfunc)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

NEPOCH = 10000
loss = np.zeros(NEPOCH) # store the training progress here.
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_train, y: y_train})
  loss[i] = sess.run(lossfunc, feed_dict={x: x_train, y: y_train})

weights,means,sds = sess.run([out_pi, out_sigma, out_mu],feed_dict={x: [x_train[0]], y: y_train})

def p(x):
  sum_ = 0
  for i in range(len(weights)):
    sum_ += weights[0][i]*norm.pdf(x,loc=means[0][i],scale=sds[0][i])
  return sum_
print (p(1)) 
