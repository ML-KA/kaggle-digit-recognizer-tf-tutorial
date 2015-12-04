#!/usr/bin/env python

import tensorflow as tf
from datetime import datetime as dt
import numpy as np
import logging
logging.getLogger("tf").setLevel(logging.WARNING)

# Getting the data in a shape we can work with

# TODO - Getting to know Python: What can you say about the format, without
# looking at the specification / implementation of `input_data.read_data_sets`?
import input_kaggle
mnist = input_kaggle.read_data_sets('input/train.csv', 'input/test.csv')

# The model
x = tf.placeholder("float", [None, 784])  # the input
W = tf.Variable(tf.zeros([784, 10]))  # first layer weights: TODO - why 784/10?
b = tf.Variable(tf.zeros([10]))  # first layer bias: TODO - why 10?
y = tf.nn.softmax(tf.matmul(x, W) + b)  # first layer activation function

# The objective function
y_ = tf.placeholder("float", [None, 10])  # TODO: Why 10?
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Training definition
# TODO: What does 0.01 mean? How do you find it out?
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialization: TODO - why is this necessary? What is done here?
init = tf.initialize_all_variables()

# Tensor flow specific stuff
sess = tf.Session()
sess.run(init)

# Training execution
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Model evaluation
# TODO: What does the '1' stand for?
argmax = tf.argmax(y, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy,
               feed_dict={x: mnist.validation.images,
                          y_: mnist.validation.labels}))
predictions = sess.run(argmax,
                       feed_dict={x: mnist.test.images})
predictions = predictions.transpose().astype(int)
data = zip(range(1, len(predictions) + 1), predictions)
np.savetxt("predictions-%s.csv" % dt.now().strftime("%Y-%m-%d-%H-%M"),
           data,
           header='ImageId,Label',
           comments='',
           fmt='%i,%i')
