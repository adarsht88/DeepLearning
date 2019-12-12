#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:21:38 2019
@author: TraVisAT
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = [0.5, 2.5] Y = [0.2, 0.9]
def f(w,b,x):
return 1.0 / (1.0 + np.exp(-(w * x + b)))
def error(w,b): err = 0.0
for x,y in zip(X,Y):
fx = f(w,b,x)
err += 0.5 * (fx - y) ** 2
return err
def grad_b(w,b,x,y):
fx = f(w,b,x)
return (fx - y) * fx * (1 - fx)
def grad_w(w,b,x,y):
fx = f(w,b,x)
return (fx - y) * fx * (1 - fx) * x
def do_gradient_descent():
weight = []
bias = []
e = []
w,b,eta,max_epochs = -2, -2 , 1.0 ,1000 for i in range(max_epochs):
weight.append(w) bias.append(b) e.append(error(w,b)) dw, db = 0, 0
for x,y in zip(X, Y):
dw += grad_w(w,b,x,y) db += grad_b(w,b,x,y)
w = w - eta * dw b = b - eta * db
print(weight)
print(bias)
print(e)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') Ws = np.arange(-6,6,0.1)
Bs = np.arange(-6,6,0.1)
Ws,Bs = np.meshgrid(Ws,Bs)
err = error(Ws,Bs)
ax = plt.subplot(projection='3d')
ax.plot_surface(Ws, Bs, err, cmap='plasma') ax.scatter(weight,bias,e,c="black",marker="+") ax.scatter(weight,bias,[-1]*len(weight),c="black",marker="o") ax.set_zlim(-1.0,1.0)
ax.set_xlabel('weight') ax.set_ylabel('bias') ax.set_zlabel('error') plt.show()
do_gradient_descent()