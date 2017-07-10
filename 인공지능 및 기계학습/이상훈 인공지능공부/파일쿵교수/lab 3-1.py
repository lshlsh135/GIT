# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:48:30 2017

@author: SH-NoteBook
"""

def cost(W, X, y):
    s = 0
    for i in range(len(X)):
        s += (W * X[i] - y[i]) ** 2

    return s / len(X)

X = [1., 2., 3.]
Y = [1., 2., 3.]

W_val, cost_val = [], []
for i in range(-30, 51):
    W = i*0.1
    c = cost(W, X, Y)

    print('{:.1f}, {:.1f}'.format(W, c))

    W_val.append(W)
    cost_val.append(c)


# ------------------------------------------ #


import matplotlib.pyplot as plt

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()

#==============================================================================
# 
#==============================================================================
def gradients(W, X, y):
    nX = []
    for i in range(len(X)):
        nX.append(W * X[i] - y[i])

    s = 0
    for i in range(len(X)):
        s += nX[i] * X[i]

    return s / len(X)

X = [1., 2., 3.]
Y = [1., 2., 3.]

# 양수로 시작하면 값이 줄어들고
# 음수로 시작하면 값이 늘어난다.
W = 100                     # -100으로도 테스트해 볼 것.
for i in range(1000):
    g = gradients(W, X, Y)
    W = W - g*0.01

    if i%20 == 19:
        print('{:4} : {:12.8f} {:12.8f}'.format(i+1, g, W))


