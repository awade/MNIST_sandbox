#!/usr/bin/env python

""" A first look at usring tensor flow for machine lerning

See tutorial: https://www.tensorflow.org/tutorials/mnist/beginners/

Andrew Wade
20170122
"""

import tensorflow as tf

# Setup variables
x = tf.placeholder(tf.float32, [None,784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Implement model
y = tf.nn.softmax(tf.matmul(x,W) + b)


