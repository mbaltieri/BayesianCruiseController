#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:14:10 2017

Solve continuous time Lyapunov equation to find covariance matrix of 
multivariate OU process

C^T * X + X * C = Q

@author: mb540
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dt = .05
T = 600
iterations = int(T / dt)
np.random.seed(42)
    
k = .0001
points = 20
variables = 3

pi_z_true = np.exp(4)                # true values used for gp
pi_w_true = np.exp(6)                # true values used for gp

pi_z_range_min = 10
pi_z_range_max = 70

pi_z_range = np.arange(pi_z_range_min, pi_z_range_max, (pi_z_range_max - pi_z_range_min) / points)

pi_w_range_min = 150
pi_w_range_max = 1500

pi_w_range = np.arange(pi_w_range_min, pi_w_range_max, (pi_w_range_max - pi_w_range_min) / points)

OMEGA_range = np.zeros((points, points, variables, variables))

#for i in range(points):
#    for j in range(points):
#        pi_z = pi_z_range[i]
#        pi_w = pi_w_range[j]
#
#        C = np.array([[- k * (pi_z + pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w, - k * pi_w, 0], [0, 0, - 1]])
#        SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#        #eigValues, eigVectors = np.linalg.eig(C)        
#        OMEGA = sp.linalg.solve_lyapunov(C, Q)
#        
#        OMEGA_range[i, j, :, :] = OMEGA
#
plt.close('all')
#for i in range(variables):
#    for j in range(variables):
#        plt.figure()
#        plt.plot(pi_w_range, OMEGA_range[5, :, i, j])
#        
##for i in range(variables):
##    for j in range(variables):
##        plt.figure()
##        plt.plot(pi_z_range, OMEGA_range[:, 2, i, j])

pi_z = pi_z_true
pi_w = pi_w_true

C = np.array([[- k * (pi_z + pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w, - k * pi_w, 0], [0, 0, - 1]])
SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#eigValues, eigVectors = np.linalg.eig(C)        
OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)

## 3d plot
##pi_z_range, pi_w_range = np.meshgrid(pi_z_range, pi_w_range)
#pi_w_range, pi_z_range = np.meshgrid(pi_w_range, pi_z_range)
#
##for i in range(variables):
##    for j in range(variables):
##        fig = plt.figure()
##        ax = fig.gca(projection='3d')
##        surf = ax.plot_surface(pi_z_range, pi_w_range, X_range[:, :, i, j], rstride=1, cstride=1, cmap='jet', alpha = .9)
##        ax.set_xlabel('Pi_z')
##        ax.set_ylabel('Pi_w')
###        ax.set_zlabel('Covariance ' + str(i) + ', ' + str(j))
##        plt.title('Covariance ' + str(i+1) + ', ' + str(j+1))
#        
#
#for i in range(points):
#    plt.figure()
#    plt.plot(pi_z_range[:, 0], X_range[:, i, 0, 0] - 2 * X_range[:, i, 0, 2] + X_range[:, i, 2, 2], label = 'Pi_w = ' + str(pi_w_range[0, i]))
#    plt.legend()
##    plt.title('Pi_w = ' + str(pi_w_range[0, i]))
#    
#
#for i in range(points):
#    plt.figure()
#    plt.plot(pi_w_range[0, :], X_range[i, :, 0, 0] - 2 * X_range[i, :, 0, 2] + X_range[i, :, 2, 2], label = 'Pi_z = ' + str(pi_z_range[i, 0]))
#    plt.legend()
##    plt.title('Pi_z = ' + str(pi_z_range[i, 0]))
#
#variance_mu = X_range[:, :, 0, 0]
#
#

X = np.zeros((variables, iterations))
X[:, 0] = np.ones((variables, ))
dX = np.zeros((variables,))
dW = np.random.randn(variables, iterations)
sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])

for i in range(iterations - 1):
    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
    X[:, i + 1] = X[:, i] + dt * dX


plt.figure()
plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'b')
plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')



skip_seconds = 20
skip_iterations = int(skip_seconds / dt)
print(np.var(X[0, skip_iterations:]))
print(np.var(X[2, skip_iterations:]))
print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)))











