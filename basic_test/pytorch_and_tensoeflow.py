import torch as t
from torch.autograd import Variable
import numpy as np
import tensorflow as tf

n, d, h = 3, 4, 5
x = Variable(t.randn(n, d))
w1 = Variable(t.randn(d, h))
w2 = Variable(t.randn(d, h))

z = 10
if z > 0:
    y = x.mm(w1)
else:
    y = x.mm(w2)

x = tf.placeholder(tf.float32, shape=(n, d))
z = tf.placeholder(tf.float32, shape=None)
w1 = tf.placeholder(tf.float32, shape=(d, h))
w2 = tf.placeholder(tf.float32, shape=(d, h))


def f1():
    return tf.matmul(x, w1)


def f2():
    return tf.matmul(x, w2)


y = tf.cond(tf.less(z, 0), f1, f2)
with tf.Session() as sess:
    values = {
        x: np.random.randn(n, d),
        z: 10,
        w1: np.random.randn(d, h),
        w2: np.random.randn(d, h)
    }
    y_val = sess.run(y, feed_dict=values)
