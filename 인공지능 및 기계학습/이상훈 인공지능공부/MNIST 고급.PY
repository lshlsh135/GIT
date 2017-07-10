# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:26:41 2017

@author: SH-NoteBook
"""

import tensorflow as tf
sess = tf.InteractiveSession()

#==============================================================================
# 플레이스홀더 (Placeholder
#==============================================================================
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#==============================================================================
# 변수 (Variable)
#==============================================================================
W= tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Variable들은 세션이 시작되기 전에 초기화되어야 합니다. 아래 코드는 모든 Variable들 
#각각에 대해 미리 지정된 초기 값(위에서 지정된 0으로만 구성된 텐서)를 넣어 주는 역할을 합니다.
sess.run(tf.global_variables_initializer())





