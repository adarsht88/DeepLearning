#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:14:36 2019
@author: root """
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = [0.5, 2.5] Y = [0.2, 0.9] emax=0.004 def f(w,b,x):
return 1.0 / (1.0 + np.exp(-(w * x + b)))
def error(w,b): err = 0.0
for x,y in zip(X,Y):
fx = f(w,b,x)
err += 0.5 * (fx - y) ** 2 return err
def grad_b(w,b,x,y): fx = f(w,b,x)
return (fx - y) * fx * (1 - fx)
def grad_w(w,b,x,y): fx = f(w,b,x)
return (fx - y) * fx * (1 - fx) * x
def vanilla_gradient_descent(): wnew = []
bnew = []
e = []
flag=0
print("vanilla gradent descent") w,b,eta,max_epochs = -2, -2 , 1.0 ,1000 for i in range(max_epochs): wnew.append(w)
bnew.append(b)
g=error(w,b) print(g) dw,db=0,0 e.append(g) c=round(g,2)
if c<=emax: print("steps",i+1) flag=1
break
for x,y in zip(X, Y):
dw += grad_w(w,b,x,y)
db += grad_b(w,b,x,y)
w = w - eta * dw
b = b - eta * db
print(g)
if flag==0:
print("more than",max_epochs," steps") print(g)
plot(wnew,bnew,e)
def momentum_based_gradient(): w,b,eta,max_epochs = -2, -2 , 1.0 ,100 prev_v_w,prev_v_b,gamma=0,0,0.9 wnew=[]
bnew=[]
e=[]
flag=0
print("momentum based gradient descent") for i in range(max_epochs) :
dw, db = 0, 0 wnew.append(w) bnew.append(b) g=error(w,b) print(g) e.append(g) c=round(g,2)
if c<=emax: print("steps",i+1) flag=1
break
for x,y in zip(X, Y):
dw += grad_w(w,b,x,y)
db += grad_b(w,b,x,y) v_w=gamma*prev_v_w+eta*dw v_b=gamma*prev_v_b+eta*db

w=w-v_w
b=b-v_b
prev_v_w=v_w
prev_v_b=v_b
if flag==0:
print("more than",max_epochs," steps") print(g)
#plot(wnew,bnew,e)
def nesterov_accelerated_gradient():
w,b,eta,max_epochs = -2, -2 , 1.0 ,100 prev_v_w,prev_v_b,gamma=0,0,0.9 wnew=[]
bnew=[]
e=[]
flag=0
print("nesterov accelerated gradient descent") for i in range(max_epochs) :
dw, db = 0, 0
v_w=gamma*prev_v_w v_b=gamma*prev_v_b
wnew.append(w)
bnew.append(b)
g=error(w,b)
print(g)
e.append(g)
c=round(g,2)
if c<=emax:
print("steps",i+1)
flag=1
break
for x,y in zip(X, Y):
dw += grad_w(w-v_w,b-v_b,x,y)
db += grad_b(w-v_w,b-v_b,x,y) v_w=gamma*prev_v_w+eta*dw v_b=gamma*prev_v_b+eta*db
w=w-v_w
b=b-v_b
prev_v_w=v_w
prev_v_b=v_b
if flag==0:
print("more than",max_epochs," steps") print(g)
plot(wnew,bnew,e)
def stochastic_gradient():

flag=0
w,b,eta,max_epochs = -2, -2 , 1.0 ,100 prev_v_w,prev_v_b,gamma=0,0,0.9 wnew=[]
bnew=[]
e=[]
print("stochastic gradient descent")
for i in range(max_epochs) :
dw, db = 0, 0
wnew.append(w)
bnew.append(b)
g=error(w,b)
print(g)
e.append(g)
c=round(g,2)
if c<=emax:
print("steps",i+1)
flag=1
break
for x,y in zip(X, Y):
dw += grad_w(w,b,x,y)
db += grad_b(w,b,x,y) v_w=gamma*prev_v_w+eta*dw v_b=gamma*prev_v_b+eta*db w=w-v_w
b=b-v_b
prev_v_w=v_w
prev_v_b=v_b
if flag==0:
print("more than",max_epochs," steps") print(g)
plot(wnew,bnew,e)
def mini_batch_gradent_descent(): wnew = []
bnew = []
e = []
flag=0
print("mini batch gradient descent") w,b,eta,max_epochs = -2, -2 , 1.0 ,1000 mini_batch_size,num_points_seen=2,0 for i in range(max_epochs): wnew.append(w)
bnew.append(b) g=error(w,b)

dw,db=0,0 print(g) e.append(g) c=round(g,2)
if c<=emax: print("steps",i+1) flag=1
break
for x,y in zip(X, Y):
dw += grad_w(w,b,x,y)
db += grad_b(w,b,x,y)
num_points_seen +=1
if num_points_seen % mini_batch_size==0: w = w - eta * dw
b = b - eta * db
dw,db=0,0
if flag==0:
print("more than",max_epochs," steps") print(g)
plot(wnew,bnew,e)
def plot(wnew,bnew,e): fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') wold = np.arange(-6,6,0.25)
bold = np.arange(-6,6,0.25)
wold,bold = np.meshgrid(wold,bold)
err = error(wold,bold)
ax = plt.subplot(projection='3d')
ax.plot_surface(wold, bold, err, cmap='coolwarm') ax.scatter(wnew,bnew,e,c="black",marker="o") ax.scatter(wnew,bnew,[-0.9]*len(wnew),c="black",marker="o")
ax.set_xlabel('weight') ax.set_ylabel('bias') ax.set_zlabel('error') plt.show()
print("vanilla Gradient Descent") vanilla_gradient_descent()
print("Momentum Based Gradient Descent") momentum_based_gradient()
print("Nesterov Accelerated Gradient Descent") nesterov_accelerated_gradient()

print("Stochastic Gradient Descent") stochastic_gradient()
print("Mini Batch Gradient Descent") mini_batch_gradent_descent()