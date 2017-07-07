# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:39:07 2017

@author: SH-NoteBook

https://tensorflow.blog/

"""
import numpy as np

from sklearn import datasets        # sklearn에서 제공해주는 당뇨병 관련 datasets 을 받음
import  matplotlib.pyplot as plt# 그래프 그리기 위해 선언
diabetes = datasets.load_diabetes()   # 당뇨병 관련 자료 받아옴. dictionary 형태

plt.scatter(diabetes['data'][:,2],diabetes['target'])  # diabetes의 data 라는곳의 3번째 열은 체지방
diabetes['target']


#==============================================================================
# 
#==============================================================================
from sklearn import linear_model

sgd_regr = linear_model.SGDRegressor(n_iter=30000, penalty='none')
sgd_regr.fit(diabetes['data'][:,2].reshape(-1,1),diabetes['target'])
print('Coefficients: ', sgd_regr.coef_, sgd_regr.intercept_)

#==============================================================================
# 
#==============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes['data'][:,2], diabetes['target'], test_size=0.1, random_state=10)

class SingleNeuron(object):
    def __init__(self):
        self._w = 0
        self._b = 0
        self._x = 0
        
    def set_params(self,w,b):
        self._w = w
        self._b = b
        
    def forpass(self,x):
        self._x = x
        _y_hat = self._w * self._x + self._b
        return _y_hat
    
    def backprop(self, err):
        m = len(self._x)
        self._w_grad = 0.1 * np.sum(err * self._x) / m

    def update_grad(self):
        self.set_params(self._w + self._w_grad, self._b + self._b_grad)
        
    def fit(self, X, y, n_iter=10):
        """정방향 계산을 하고 역방향으로 에러를 전파시키면서 모델을 최적화시킵니다."""
        for i in range(n_iter):
            y_hat = self.forpass(X)
            error = y - y_hat
            self.backprop(error)
            self.update_grad()
                
        
    
        
n1 = SingleNeuron()
n1.set_params(5,1)
n1.forpass(4)

n1.set_params(5, 1)
n1.fit(X_train, y_train, 30000)
print('Final W', n1._w)
print('Final b', n1._b)

from sklearn import linear_model
sgd_regr = linear_model.SGDRegressor(n_iter=30000, penalty='none')
sgd_regr.fit(X_train.reshape(-1, 1), y_train)
print('Coefficients: ', sgd_regr.coef_, sgd_regr.intercept_)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=10)

from sklearn import metrics
learning_rate = [1.2, 1.0, 0.8]
for lr in learning_rate:
    validation_errors = 0
    for train, validation in kf.split(X_train):
        n1.fit(X_train[train], y_train[train], 2000, lr)
        y_hat = n1.forpass(X_train[validation])
        validation_errors += metrics.mean_squared_error(y_train[validation], y_hat)
    print(validation_errors/5)



import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
tf.__version__                            #버전 확인하는 코드

