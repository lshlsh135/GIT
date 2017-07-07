import matplotlib.pyplot as plt
import math
N = 100
t = 0.4
loss = []
for i in range(N):
    y = i/N
    loss.append(-y*math.log(t) - (1-t)*math.log(1-t))

