from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter import matplotlib.pyplot as plt
import numpy as np
def function(w1,x1,w2,x2,b):
return 1/(1+np.exp(-(w1*x1+w2*x2+b)))
def sigmoid(x):
return 1/(1+np.exp(-(x*55-108)))
X = range(-100,100)
Y = range(-100,100)
X, Y = np.meshgrid(X, Y)
Z1 = function(150,X,0,Y,1000)
Z2 = function(150,X,0,Y,-2000) Z3 = function(0,X,150,Y,1000) Z4 = function(0,X,150,Y,-2000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = plt.subplot(projection='3d') m=ax.plot_surface(X, Y, Z1,cmap=cm.viridis,
linewidth=0, antialiased=False) ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5) plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') m=ax.plot_surface(X, Y, Z2,cmap=cm.viridis,
linewidth=0, antialiased=False) ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5) plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m=ax.plot_surface(X, Y, Z3,cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m=ax.plot_surface(X, Y, Z4,cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5) plt.show()
Z5 = Z1-Z2 Z6 = Z3 - Z4 Z7 = Z5 + Z6
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') m=ax.plot_surface(X, Y, Z5,cmap=cm.viridis,
linewidth=0, antialiased=False) ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5) plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m=ax.plot_surface(X, Y, Z6,cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5) plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m=ax.plot_surface(X, Y, Z7,cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5) plt.show()
x=sigmoid(X) y=sigmoid(Y)

z=sigmoid(Z7)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m=ax.plot_surface(X, Y, z,cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('X Label') ax.set_ylabel('Y Label') ax.set_zlabel('Z Label') fig.colorbar(m, shrink=0.5, aspect=5) plt.show()
