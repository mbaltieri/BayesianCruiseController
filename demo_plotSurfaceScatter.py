#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:01:19 2017

@author: mb540
"""

from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm


fig = plt.figure()
ax = fig.gca(projection='3d')               # to work in 3d
plt.hold(True)

x_surf=np.arange(0, 1, 0.01)                # generate a mesh
y_surf=np.arange(0, 1, 0.01)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = np.sqrt(x_surf+y_surf)             # ex. function, which depends on x and y
ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot, alpha = .5);    # plot a 3d surface plot

n = 100
seed(0)                                     # seed let us to have a reproducible set of random numbers
x=[random() for i in range(n)]              # generate n random points
y=[random() for i in range(n)]
z=[random() for i in range(n)]
ax.scatter(x, y, z);                        # plot a 3d scatter plot

ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_zlabel('z label')

plt.show()