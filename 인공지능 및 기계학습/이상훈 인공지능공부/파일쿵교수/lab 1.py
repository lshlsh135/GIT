# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:07:10 2017

@author: SH-NoteBook
"""
import tensorflow as tf

def hello():
    a = tf.constant("hello, tensorflow")
    print(a)
    sess = tf.Session()
    result = sess.run(a)
    
    # 2.x 버전에서는 문자열로 출력되지만, 3.x 버전에서는 byte 자료형
    # 문자열로 변환하기 위해 decode 함수로 변환
    print(result)                                 # b'hello, tensorflow!'
    print(type(result))                           # <class 'bytes'>
    print(result.decode(encoding='utf-8'))        # hello, tensorflow!
    print(type(result.decode(encoding='utf-8')))  # <class 'str'>
    sess.close()

def constant():
    a = tf.constant(2)
    b = tf.constant(3)
    
    with tf.Session() as sess:
        result = sess.run(a+b)
        print(type(result))
        print(result)
        
def placeHolder():
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)
    
    add = tf.add(a,b)
    mul = tf.multiply(a,b)
    
    
    with tf.Session() as sess:
        # {a: 2, b: 3}는 딕셔너리
        # key로 'a'와 'b'를 사용하고, value로 2와 3  사용
        # free_dict를 사용하지 않을 경우 None 기본값 적용
        r1 = sess.run(add,feed_dict={a:2,b:3})
        r2 = sess.run(mul,feed_dict={a:2,b:3})
        
        print(type(r1))
        print(r1,r2)
        
placeHolder()









