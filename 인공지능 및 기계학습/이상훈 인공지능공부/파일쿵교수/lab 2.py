# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:32:27 2017

@author: SH-NoteBook
"""

#lab 2
#Linear Regression

import tensorflow as tf

x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

# try to find values for w and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

hypothesis = W * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
rate = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# before starting, initialize the variables. We will 'run' this first.
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print('{:4} {} {} {}'.format(step, sess.run(cost), sess.run(W), sess.run(b)))

# learns best fit is W: [1] b: [0]

