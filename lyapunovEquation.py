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
T = 60
iterations = int(T / dt)
#np.random.seed(42)
    
k = .001
points = 20
variables = 3

pi_z_true = np.exp(4)                # true values used for gp
pi_w_true = np.exp(6)                # true values used for gp

                  
### test MSE with different combinations of pi_z, pi_w with k fixed

pi_z_range_min = 10
pi_z_range_max = 170

pi_z_range = np.arange(pi_z_range_min, pi_z_range_max, (pi_z_range_max - pi_z_range_min) / points)

pi_w_range_min = 150
pi_w_range_max = 2500

pi_w_range = np.arange(pi_w_range_min, pi_w_range_max, (pi_w_range_max - pi_w_range_min) / points)

OMEGA_range = np.zeros((points, points, variables, variables))
MSE_for_pi = np.zeros((points, points))

for i in range(points):
    for j in range(points):
        pi_z = pi_z_range[i]
        pi_w = pi_w_range[j]

        C = np.array([[- k * (pi_z + pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w, - k * pi_w, 0], [0, 0, - 1]])
        SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
        #eigValues, eigVectors = np.linalg.eig(C)        
        OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
        
        MSE_for_pi[i, j] = OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]
        
        OMEGA_range[i, j, :, :] = OMEGA

plt.close('all')
#for i in range(variables):
#    for j in range(variables):
#        plt.figure()
#        plt.plot(pi_w_range, OMEGA_range[5, :, i, j])
#        
#for i in range(variables):
#    for j in range(variables):
#        plt.figure()
#        plt.plot(pi_z_range, OMEGA_range[:, 2, i, j])


# 3d plot MSE
#pi_z_range, pi_w_range = np.meshgrid(pi_z_range, pi_w_range)
pi_w_range, pi_z_range = np.meshgrid(pi_w_range, pi_z_range)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(pi_z_range, pi_w_range, MSE_for_pi, rstride=1, cstride=1, cmap='jet', alpha = .9)
ax.set_xlabel('Pi_z')
ax.set_ylabel('Pi_w')
plt.title('MSE')

### test MSE with different combinations of pi_z, pi_w with k fixed, 
# using a different encoding in terms of M = k * pi_w and R = pi_z / pi_w

M_min = .2
M_max = 1.8

R_min = .04
R_max = .15

M_range = np.arange(M_min, M_max, (M_max - M_min) / points)
R_range = np.arange(R_min, R_max, (R_max - R_min) / points)

MSE_for_MR = np.zeros((points, points))

for i in range(points):
    for j in range(points):
        M = M_range[i]
        R = R_range[j]

        C = np.array([[- M * (R + 1), 1 - M, M * R], [- M, - M, 0], [0, 0, - 1]])
        SIGMA = - np.array([[(M * R) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
        #eigValues, eigVectors = np.linalg.eig(C)        
        OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
        
        MSE_for_MR[i, j] = OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]
        
#pi_z_range, pi_w_range = np.meshgrid(pi_z_range, pi_w_range)
R_range, M_range = np.meshgrid(R_range, M_range)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(M_range, R_range, MSE_for_MR, rstride=1, cstride=1, cmap='jet', alpha = .7)
ax.set_xlabel('Magnitude')
ax.set_ylabel('Ratio')
plt.title('MSE')

# calc index of min/max Z value
xmin, ymin = np.unravel_index(np.argmin(MSE_for_MR), MSE_for_MR.shape)
xmax, ymax = np.unravel_index(np.argmax(MSE_for_MR), MSE_for_MR.shape)

# min max points in 3D space (x,y,z)
mi = np.array([M_range[xmin,ymin], R_range[xmin,ymin], MSE_for_MR.min()])
ma = (M_range[xmax, ymax], R_range[xmax, ymax], MSE_for_MR.max())

ax.scatter(mi[0], mi[1], mi[2], marker="o", color='r')

## Arrays for plotting, 
## first row for points in xplane, last row for points in 3D space
#Ami = np.array([mi]*4)
#Ama = np.array([ma]*4)
#for i, v in enumerate([-40,40,-100]):
#    Ami[i,i] = v 
#    Ama[i,i] = v 
#
##plot points.
#ax.plot(Ami[:,0], Ami[:,1], Ami[:,2], marker="o", ls="", c=cm.coolwarm(0.))
#ax.plot(Ama[:,0], Ama[:,1], Ama[:,2], marker="o", ls="", c=cm.coolwarm(1.))

ax.view_init(azim=-45, elev=19)
plt.savefig(__file__+".png")
plt.show()

plt.figure()
for i in range(points):
    plt.plot(M_range[:, 0], MSE_for_MR[:, i], label = 'Ratio = ' + str(R_range[0, i]))
    plt.xlabel('Magnitude')
    plt.title('MSE')
    plt.legend()
    
plt.figure()
for i in range(points):
    plt.plot(R_range[0, :], MSE_for_MR[i, :], label = 'Magnitude = ' + str(M_range[i, 0]))
    plt.xlabel('Ratio')
    plt.title('MSE')
    plt.legend()


pi_z = pi_z_true
pi_w = pi_w_true


### test MSE for different k

k_max = .01
k_min = .0
step = .00001

k_range = np.arange(k_min, k_max, step)

k_iterations = len(k_range)
MSE_for_k = np.zeros((k_iterations, ))

pi_w = np.exp(7)

for i in range(k_iterations):
    k = k_range[i]
    
    C = np.array([[- k * (pi_z + pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w, - k * pi_w, 0], [0, 0, - 1]])
    SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
    #eigValues, eigVectors = np.linalg.eig(C)        
    OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
    
    MSE_for_k[i] = OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]
    
plt.figure()
plt.plot(k_range, MSE_for_k)


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



### simulate equations numerically

pi_z = np.exp(4)
pi_w = np.exp(6)

k = .001

C = np.array([[- k * (pi_z + pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w, - k * pi_w, 0], [0, 0, - 1]])
SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])

X = np.zeros((variables, iterations))
X[:, 0] = np.ones((variables, ))
dX = np.zeros((variables,))
dW = np.random.randn(variables, iterations)
sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])

OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)

for i in range(iterations - 1):
    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
    X[:, i + 1] = X[:, i] + dt * dX


plt.figure()
plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'b')
plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')



skip_seconds = 20
skip_iterations = int(skip_seconds / dt)
#print(np.var(X[0, skip_iterations:]))
#print(np.var(X[2, skip_iterations:]))
print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)))



pi_z = np.exp(4)
pi_w = np.exp(7)

k = .001

C = np.array([[- k * (pi_z + pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w, - k * pi_w, 0], [0, 0, - 1]])
SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])

X = np.zeros((variables, iterations))
X[:, 0] = np.ones((variables, ))
dX = np.zeros((variables,))
dW = np.random.randn(variables, iterations)
sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])

OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)

for i in range(iterations - 1):
    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
    X[:, i + 1] = X[:, i] + dt * dX


plt.figure()
plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'b')
plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')



skip_seconds = 20
skip_iterations = int(skip_seconds / dt)
#print(np.var(X[0, skip_iterations:]))
#print(np.var(X[2, skip_iterations:]))
print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)))



pi_z = np.exp(5)
pi_w = np.exp(7)

k = .001

C = np.array([[- k * (pi_z + pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w, - k * pi_w, 0], [0, 0, - 1]])
SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])

X = np.zeros((variables, iterations))
X[:, 0] = np.ones((variables, ))
dX = np.zeros((variables,))
dW = np.random.randn(variables, iterations)
sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])

OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)

for i in range(iterations - 1):
    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
    X[:, i + 1] = X[:, i] + dt * dX


plt.figure()
plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'b')
plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')



skip_seconds = 20
skip_iterations = int(skip_seconds / dt)
#print(np.var(X[0, skip_iterations:]))
#print(np.var(X[2, skip_iterations:]))
print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)))



