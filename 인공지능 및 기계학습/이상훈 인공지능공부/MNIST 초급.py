# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 08:42:58 2017

https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/beginners/

@author: SH-NoteBook
"""

#==============================================================================
# MNIST 데이터셋
#==============================================================================
#from tensorflow.examples.tutorials.mnist import input_data
#import pandas as pd
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#==============================================================================
# 회귀 구현하기
#==============================================================================
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))





