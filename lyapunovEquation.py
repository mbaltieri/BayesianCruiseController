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
import sympy as sy

dt = .05
T = 240
iterations = int(T / dt)
#np.random.seed(42)
plt.close('all')

beta = 1.
    
k = .001
points = 20
variables = 3

pi_z_true = np.exp(4)                # true values used for gp
pi_w_true = np.exp(6)                # true values used for gp
#
#                  
#### test MSE with different combinations of pi_z, pi_w with k fixed
#
#pi_z_range_min = 10
#pi_z_range_max = 170
#
#pi_z_range = np.arange(pi_z_range_min, pi_z_range_max, (pi_z_range_max - pi_z_range_min) / points)
#
#pi_w_range_min = 150
#pi_w_range_max = 2500
#
#pi_w_range = np.arange(pi_w_range_min, pi_w_range_max, (pi_w_range_max - pi_w_range_min) / points)
#
#OMEGA_range = np.zeros((points, points, variables, variables))
#MSE_for_pi = np.zeros((points, points))
#
#for i in range(points):
#    for j in range(points):
#        pi_z = pi_z_range[i]
#        pi_w = pi_w_range[j]
#
#        C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
#        SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#        #eigValues, eigVectors = np.linalg.eig(C)        
#        OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#        
#        MSE_for_pi[i, j] = OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]
#        
#        OMEGA_range[i, j, :, :] = OMEGA
#

##for i in range(variables):
##    for j in range(variables):
##        plt.figure()
##        plt.plot(pi_w_range, OMEGA_range[5, :, i, j])
##        
##for i in range(variables):
##    for j in range(variables):
##        plt.figure()
##        plt.plot(pi_z_range, OMEGA_range[:, 2, i, j])
#
#
## 3d plot MSE
##pi_z_range, pi_w_range = np.meshgrid(pi_z_range, pi_w_range)
#pi_w_range, pi_z_range = np.meshgrid(pi_w_range, pi_z_range)
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(pi_z_range, pi_w_range, MSE_for_pi, rstride=1, cstride=1, cmap='jet', alpha = .9)
#ax.set_xlabel('Pi_z')
#ax.set_ylabel('Pi_w')
#plt.title('MSE')
#
#### test MSE with different combinations of pi_z, pi_w with k fixed, 
## using a different encoding in terms of M = k * pi_w and R = pi_z / pi_w
#
#M_min = .2
#M_max = 1.8
#
#R_min = .04
#R_max = .15
#
#M_range = np.arange(M_min, M_max, (M_max - M_min) / points)
#R_range = np.arange(R_min, R_max, (R_max - R_min) / points)
#
#MSE_for_MR = np.zeros((points, points))
#
#for i in range(points):
#    for j in range(points):
#        M = M_range[i]
#        R = R_range[j]
#
#        C = np.array([[- M * (R + 1), 1 - M, M * R], [- M, - M, 0], [0, 0, - beta]])
#        SIGMA = - np.array([[(M * R) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#        #eigValues, eigVectors = np.linalg.eig(C)        
#        OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#        
#        MSE_for_MR[i, j] = OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]
#        
##pi_z_range, pi_w_range = np.meshgrid(pi_z_range, pi_w_range)
#R_range, M_range = np.meshgrid(R_range, M_range)
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(M_range, R_range, MSE_for_MR, rstride=1, cstride=1, cmap='jet', alpha = .7)
#ax.set_xlabel('Magnitude')
#ax.set_ylabel('Ratio')
#plt.title('MSE')
#
## calc index of min/max Z value
#xmin, ymin = np.unravel_index(np.argmin(MSE_for_MR), MSE_for_MR.shape)
#xmax, ymax = np.unravel_index(np.argmax(MSE_for_MR), MSE_for_MR.shape)
#
## min max points in 3D space (x,y,z)
#mi = np.array([M_range[xmin,ymin], R_range[xmin,ymin], MSE_for_MR.min()])
#ma = (M_range[xmax, ymax], R_range[xmax, ymax], MSE_for_MR.max())
#
#ax.scatter(mi[0], mi[1], mi[2], marker="o", color='r')
#
#ax.view_init(azim=-45, elev=19)
#plt.show()
#
#plt.figure()
#for i in range(points):
#    plt.plot(M_range[:, 0], MSE_for_MR[:, i], label = 'Ratio = ' + str(R_range[0, i]))
#    plt.xlabel('Magnitude')
#    plt.title('MSE')
#    plt.legend()
#    
#plt.figure()
#for i in range(points):
#    plt.plot(R_range[0, :], MSE_for_MR[i, :], label = 'Magnitude = ' + str(M_range[i, 0]))
#    plt.xlabel('Ratio')
#    plt.title('MSE')
#    plt.legend()
#
#
#pi_z = pi_z_true
#pi_w = pi_w_true
#
#
#### test MSE for different k
#
#k_max = .01
#k_min = .0
#step = .00001
#
#k_range = np.arange(k_min, k_max, step)
#
#k_iterations = len(k_range)
#MSE_for_k = np.zeros((k_iterations, ))
#
#pi_w = np.exp(7)
#
#for i in range(k_iterations):
#    k = k_range[i]
#    
#    C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
#    SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#    #eigValues, eigVectors = np.linalg.eig(C)        
#    OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#    
#    MSE_for_k[i] = OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]
#    
#plt.figure()
#plt.plot(k_range, MSE_for_k)
#plt.plot(k_range[MSE_for_k.argmin()], MSE_for_k.min(), marker="o", color='r')
#plt.title('Min = ' + str(k_range[MSE_for_k.argmin()]))
#
#
#C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
#SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
##eigValues, eigVectors = np.linalg.eig(C)        
#OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#
### 3d plot
###pi_z_range, pi_w_range = np.meshgrid(pi_z_range, pi_w_range)
##pi_w_range, pi_z_range = np.meshgrid(pi_w_range, pi_z_range)
##
###for i in range(variables):
###    for j in range(variables): 
###        fig = plt.figure()
###        ax = fig.gca(projection='3d')
###        surf = ax.plot_surface(pi_z_range, pi_w_range, X_range[:, :, i, j], rstride=1, cstride=1, cmap='jet', alpha = .9)
###        ax.set_xlabel('Pi_z')
###        ax.set_ylabel('Pi_w')
####        ax.set_zlabel('Covariance ' + str(i) + ', ' + str(j))
###        plt.title('Covariance ' + str(i+1) + ', ' + str(j+1))
##        
##
##for i in range(points):
##    plt.figure()
##    plt.plot(pi_z_range[:, 0], X_range[:, i, 0, 0] - 2 * X_range[:, i, 0, 2] + X_range[:, i, 2, 2], label = 'Pi_w = ' + str(pi_w_range[0, i]))
##    plt.legend()
###    plt.title('Pi_w = ' + str(pi_w_range[0, i]))
##    
##
##for i in range(points):
##    plt.figure()
##    plt.plot(pi_w_range[0, :], X_range[i, :, 0, 0] - 2 * X_range[i, :, 0, 2] + X_range[i, :, 2, 2], label = 'Pi_z = ' + str(pi_z_range[i, 0]))
##    plt.legend()
###    plt.title('Pi_z = ' + str(pi_z_range[i, 0]))
##
##variance_mu = X_range[:, :, 0, 0]
##
##
#
#
#
#### simulate equations numerically
#
#pi_z = np.exp(3)
#pi_w = np.exp(5)
#
#k = .0001
#
#C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
#SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#
#X = np.zeros((variables, iterations))
##X[:, 0] = np.ones((variables, ))
#dX = np.zeros((variables,))
#dW = np.random.randn(variables, iterations)
#sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])
#
#OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#
#for i in range(iterations - 1):
#    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
#    X[:, i + 1] = X[:, i] + dt * dX
#
#
#plt.figure()
#plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'b')
#plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')
#plt.title('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#
#
#skip_seconds = 20
#skip_iterations = int(skip_seconds / dt)
##print(np.var(X[0, skip_iterations:]))
##print(np.var(X[2, skip_iterations:]))
#print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)) + '\n')
#



pi_z = np.exp(4)
pi_w = np.exp(6)

k = .001
#k = 1 / pi_z 

C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])

X = np.zeros((variables, iterations))
X[:, 0] = np.ones((variables, ))
dX = np.zeros((variables, iterations))
dW = np.random.randn(variables, iterations)
sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])

OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)

sigma_w = 1 / np.sqrt(pi_w_true)
sigma_z = 1 / np.sqrt(pi_z_true)
alpha = beta

omega31 = pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2 * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w)
omega32 = - k * beta * pi_w / (alpha + k * pi_w) * omega31
omega33 = sigma_w ** 2 / (2 * alpha)
omega11 = 1 / (2 * k * (pi_z + beta * pi_w)) * (2 * pi_z * (1 - k * pi_w) * omega32 / (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w) + 2 * k * pi_z * omega31 + k ** 2 * pi_z ** 2 * 
               sigma_z ** 2) / (1 + 2 * beta * pi_w * (1 - k * pi_w) / (2 * k * (pi_z + beta * pi_w) * (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w)))
omega21 = (pi_z * omega32 - beta * pi_w * omega11) / (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w)
omega22 = - beta * omega21

OMEGA_TEST = np.array([[omega11, omega21, omega31], [omega21, omega22, omega32], [omega31, omega32, omega33]])

F = np.zeros((iterations, ))
E = np.zeros((iterations, ))
rho = np.zeros((iterations, ))
eps_z = np.zeros((iterations, ))
eps_w = np.zeros((iterations, ))

kappa = .05
phi = 0.
pi_z_history = np.zeros((iterations, ))

eta = .01
psi = 0.
pi_w_history = np.zeros((iterations, ))

aa = np.zeros((iterations, ))
bb = np.zeros((iterations, ))

for i in range(iterations - 1):
    dX[:, i] = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
    X[:, i + 1] = X[:, i] + dt * dX[:, i]

#    dX[2, i] = - beta * X[2, i] + sigma[2, 2] * dW[2, i] / np.sqrt(dt)
#    X[2, i + 1] = X[2, i] + dt * dX[2, i]
#    
##    dX[:-1, i] = np.dot(C[:-1, :-1], X[:-1, i]) + np.dot(sigma[:-1, :-1], dW[:-1, i]) / np.sqrt(dt)
##    dX[0, i] = np.dot(C[0, :-1], X[:-1, i]) + np.dot(sigma[0, :-1], dW[:-1, i]) / np.sqrt(dt)
##    dX[1, i] = np.dot(C[1, :-1], X[:-1, i]) + np.dot(sigma[1, :-1], dW[:-1, i]) / np.sqrt(dt)
#    dX[0, i] = X[1, i] - k * (pi_z * (X[0, i] - X[2, i]) + pi_w * (X[1, i] + X[0, i]) - pi_z / np.sqrt(pi_z_true) * dW[0, i] / np.sqrt(dt)) 
#    dX[1, i] = 0 - k * (pi_w * (X[1, i] + X[0, i])) # + pi_w * (X[1, i])) 
                                                    # + pi_z * (X[1, i] - dX[2, i]) - pi_z / np.sqrt(pi_z_true) * dW[0, i] / np.sqrt(dt))
    
    aa[i] = pi_z * (X[1, i] - dX[2, i])
    bb[i] = pi_w * (X[1, i] + X[0, i])
    X[:-1, i + 1] = X[:-1, i] + dt * dX[:-1, i]
    
#    if i > iterations / 3:
#        pi_w = np.exp(6)
#        if i > 2 * iterations / 3:
#            pi_w = np.exp(7)
    
    rho[i] = X[2, i] + dW[0, i] / np.sqrt(pi_z_true)
    eps_z[i] = rho[i] - X[0, i]
    eps_w[i] = X[1, i] + beta * X[0, i]
    F[i] = .5 * (pi_z * eps_z[i] ** 2 + pi_w * eps_w[i] ** 2 - np.log(pi_z * pi_w))
    E[i] = (X[2, i] - X[0, i]) ** 2
    
#    dphi = - .5 * (eps_z[i] ** 2 - 1 /  pi_z) - kappa * phi
#    phi += dt * dphi
#    
#    dpi_z = phi
#    pi_z += dt * dpi_z
    
    pi_z_history[i] = pi_z
    
    
#    dpsi = - .5 * (eps_w[i] ** 2 - 1 /  pi_w) - eta * psi
#    psi += dt * dpsi
#    
#    dpi_w = psi
#    pi_w += dt * dpi_w
    
    pi_w_history[i] = pi_w

skip_seconds = 20
skip_iterations = int(skip_seconds / dt)

#print(np.var(eps_z[skip_iterations:]))
#print(1/pi_z)
#print(1/pi_z_true)
print(np.var(X[2, skip_iterations:]))
print(np.var(eps_w[skip_iterations:]))
print(OMEGA[0, 0] + 2 * OMEGA[1, 0] + OMEGA[1, 1])
print(1/pi_w)
#print(1/pi_w_true)


plt.figure()
plt.plot(pi_z_history[:-1])
plt.title('Pi_z')
plt.grid(True)

plt.figure()
plt.plot(pi_w_history[:-1])
plt.title('Pi_w')
plt.grid(True)

plt.figure()
plt.plot(np.arange(0, iterations*dt, dt), rho, 'b')
plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'g')
plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')
plt.title('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))

plt.figure()
plt.plot(np.arange(0, iterations*dt, dt), dX[2, :], 'b')
plt.plot(np.arange(0, iterations*dt, dt), dX[0, :], 'g')
plt.plot(np.arange(0, iterations*dt, dt), X[1, :], 'r')

#plt.figure()
#plt.plot(aa)
#
#plt.figure()
#plt.plot(bb)

#
#plt.figure()
#plt.semilogy(F[:-1], 'b')
##plt.semilogy(E[:-1], 'r')
#
#plt.figure()
##plt.plot(eps_z[:-1], 'b')
#plt.plot(eps_w[:-1], 'r')
#
##plt.figure()
##plt.plot(eps_z[:-1] ** 2, 'b')
##plt.plot(eps_w[:-1] ** 2, 'r')
#
#
#
#skip_seconds = 20
#skip_iterations = int(skip_seconds / dt)
##print(np.var(X[0, skip_iterations:]))
##print(np.var(X[2, skip_iterations:]))
#print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)))
#print('MSE_rho = ' + str(np.mean((rho[:-1] - X[0, :-1])**2)))
#print('MFE = ' + str(np.mean(F)) + '\n')



#pi_z = np.exp(5)
#pi_w = np.exp(6)
#
#k = .001
#
#C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
#SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#
#X = np.zeros((variables, iterations))
##X[:, 0] = np.ones((variables, ))
#dX = np.zeros((variables,))
#dW = np.random.randn(variables, iterations)
#sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])
#
#OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#
#F = np.zeros((iterations, ))
#E = np.zeros((iterations, ))
#rho = np.zeros((iterations, ))
#eps_z = np.zeros((iterations, ))
#eps_w = np.zeros((iterations, ))
#
#for i in range(iterations - 1):
##    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
##    X[:, i + 1] = X[:, i] + dt * dX
#
#    dX[2] = - beta * X[2, i] + sigma[2, 2] * dW[2, i] / np.sqrt(dt)
#    X[2, i + 1] = X[2, i] + dt * dX[2]
#    
#    dX[:-1] = np.dot(C[:-1, :-1], X[:-1, i]) + np.dot(sigma[:-1, :-1], dW[:-1, i]) / np.sqrt(dt)
#    X[:-1, i + 1] = X[:-1, i] + dt * dX[:-1]
#    
#    rho[i] = X[2, i] + dW[0, i] / np.sqrt(pi_z_true)
#    eps_z[i] = rho[i] - X[0, i]
#    eps_w[i] = X[1, i] + beta * X[0, i]
#    F[i] = .5 * (pi_z * eps_z[i] ** 2 + pi_w * eps_w[i] ** 2 - np.log(pi_z * pi_w))
#    E[i] = (X[2, i] - X[0, i]) ** 2
#
#plt.figure()
#plt.plot(np.arange(0, iterations*dt, dt), X[2, :] + dW[0, :] / np.sqrt(pi_z_true), 'b')
#plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'g')
#plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')
#plt.title('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#
#skip_seconds = 20
#skip_iterations = int(skip_seconds / dt)
##print(np.var(X[0, skip_iterations:]))
##print(np.var(X[2, skip_iterations:]))
#print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)))
#print('MSE_rho = ' + str(np.mean((rho[:-1] - X[0, :-1])**2)))
#print('MFE = ' + str(np.mean(F)) + '\n')

#pi_z = np.exp(4)
#pi_w = np.exp(7)
#
#k = .001
#
#C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
#SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#
#X = np.zeros((variables, iterations))
##X[:, 0] = np.ones((variables, ))
#dX = np.zeros((variables,))
#dW = np.random.randn(variables, iterations)
#sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])
#
#OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#
#for i in range(iterations - 1):
#    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
#    X[:, i + 1] = X[:, i] + dt * dX
#
#
#plt.figure()
#plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'b')
#plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')
#plt.title('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#
#
#skip_seconds = 20
#skip_iterations = int(skip_seconds / dt)
##print(np.var(X[0, skip_iterations:]))
##print(np.var(X[2, skip_iterations:]))
#print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)) + '\n')



#pi_z = np.exp(5)
#pi_w = np.exp(7)
#
#k = .001
#
#C = np.array([[- k * (pi_z + beta * pi_w), 1 - k * pi_w, k * pi_z], [- k * pi_w * beta, - k * pi_w, 0], [0, 0, - beta]])
#SIGMA = - np.array([[(k * pi_z) ** 2 / pi_z_true, 0, 0], [0, 0, 0], [0, 0, 1 / pi_w_true]])
#
#X = np.zeros((variables, iterations))
##X[:, 0] = np.ones((variables, ))
#dX = np.zeros((variables,))
#dW = np.random.randn(variables, iterations)
#sigma = np.array([[k * pi_z / np.sqrt(pi_z_true), 0, 0], [0, 0, 0], [0, 0, 1 / np.sqrt(pi_w_true)]])
#
#OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#
#for i in range(iterations - 1):
#    dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
#    X[:, i + 1] = X[:, i] + dt * dX
#
#
#plt.figure()
#plt.plot(np.arange(0, iterations*dt, dt), X[2, :], 'b')
#plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')
#plt.title('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#
#
#skip_seconds = 20
#skip_iterations = int(skip_seconds / dt)
##print(np.var(X[0, skip_iterations:]))
##print(np.var(X[2, skip_iterations:]))
#print('MSE = ' + str(OMEGA[0, 0] - 2 * OMEGA[0, 2] + OMEGA[2, 2]))
#print('MSE = ' + str(1/(iterations - skip_iterations) * np.sum((X[2, skip_iterations:] - X[0, skip_iterations:])**2)) + '\n')

#sigma_w = 1


#pi_z = np.exp(4)
#k = 0.001
#J = np.array([[0, 1.], [- 1 / ( 2 * pi_z ** 2), - k]])
#eigValues, eigVectors = np.linalg.eig(J)
#
#pi_w = np.exp(6)
#k = 0.00001
#J = np.array([[0, 1.], [- 1 / ( 2 * pi_w ** 2), - k]])
#eigValues2, eigVectors2 = np.linalg.eig(J)


#pi_z, pi_w, alpha, beta, k, sigma_z, sigma_w = sy.symbols('pi_z pi_w alpha beta k sigma_z sigma_w')

omega31 = pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2 * alpha ** 2
                                * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w)
omega32 = - k * beta * pi_w / (alpha + k * pi_w) * pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2
                              * k * alpha * pi_z * pi_w + 2 * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w)
omega33 = sigma_w ** 2 / (2 * alpha)
omega11 = 1 / (2 * k * (pi_z + beta * pi_w)) * (2 * pi_z * (1 - k * pi_w) * - k * beta * pi_w / (alpha + k * pi_w) * pi_z * sigma_w ** 2 *
               (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2 * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k
               + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) / (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w) + 2 * k * pi_z * 
               pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2 * alpha ** 2 * beta * pi_w
               + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) + k ** 2 * pi_z ** 2 * 
               sigma_z ** 2) / (1 + 2 * beta * pi_w * (1 - k * pi_w) / (2 * k * (pi_z + beta * pi_w) * (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w)))
omega21 = (pi_z * - k * beta * pi_w / (alpha + k * pi_w) * pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2
           * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) - beta * pi_w *  1 / (2 * k * (pi_z + beta * pi_w))
           * (2 * pi_z * (1 - k * pi_w) * - k * beta * pi_w / (alpha + k * pi_w) * pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k *
           alpha * pi_z * pi_w + 2 * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) / (pi_z + beta * pi_w +
           pi_w + beta / k - beta * pi_w) + 2 * k * pi_z * pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2 
           * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) + k ** 2 * pi_z ** 2 * sigma_z ** 2) / (1 + 2 * 
           beta * pi_w * (1 - k * pi_w) / (2 * k * (pi_z + beta * pi_w) * (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w)))) / (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w)
omega22 = - beta * (pi_z * - k * beta * pi_w / (alpha + k * pi_w) * pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2
           * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) - beta * pi_w *  1 / (2 * k * (pi_z + beta * pi_w))
           * (2 * pi_z * (1 - k * pi_w) * - k * beta * pi_w / (alpha + k * pi_w) * pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k *
           alpha * pi_z * pi_w + 2 * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) / (pi_z + beta * pi_w +
           pi_w + beta / k - beta * pi_w) + 2 * k * pi_z * pi_z * sigma_w ** 2 * (alpha + k * pi_w) / (2 * alpha ** 2 * pi_z + 2 * k * alpha * pi_z * pi_w + 2 
           * alpha ** 2 * beta * pi_w + 2 * alpha ** 3 / k + 2 * alpha ** 2 * pi_w + 2 * alpha * beta * pi_w) + k ** 2 * pi_z ** 2 * sigma_z ** 2) / (1 + 2 * 
           beta * pi_w * (1 - k * pi_w) / (2 * k * (pi_z + beta * pi_w) * (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w)))) / (pi_z + beta * pi_w + pi_w + beta / k - beta * pi_w)

OMEGA_TEST = np.array([[omega11, omega21, omega31], [omega21, omega22, omega32], [omega31, omega32, omega33]])


def dMsedpi_z(pi_z):
    return -2*pi_z*sigma_w**2*(alpha + k*pi_w)*(-2*alpha**2 - 2*alpha*k*pi_w)/(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z)**2 - 2*sigma_w**2*(alpha + k*pi_w)/(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z) + (beta*pi_w*(-k*pi_w + 1)/(k*(beta*pi_w + pi_z)*(beta/k + pi_w + pi_z)**2) + beta*pi_w*(-k*pi_w + 1)/(k*(beta*pi_w + pi_z)**2*(beta/k + pi_w + pi_z)))*(-2*beta*k*pi_w*pi_z**2*sigma_w**2*(-k*pi_w + 1)/((beta/k + pi_w + pi_z)*(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z)) + k**2*pi_z**2*sigma_z**2 + 2*k*pi_z**2*sigma_w**2*(alpha + k*pi_w)/(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z))/(2*k*(beta*pi_w + pi_z)*(beta*pi_w*(-k*pi_w + 1)/(k*(beta*pi_w + pi_z)*(beta/k + pi_w + pi_z)) + 1)**2) + (-2*beta*k*pi_w*pi_z**2*sigma_w**2*(-2*alpha**2 - 2*alpha*k*pi_w)*(-k*pi_w + 1)/((beta/k + pi_w + pi_z)*(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z)**2) + 2*beta*k*pi_w*pi_z**2*sigma_w**2*(-k*pi_w + 1)/((beta/k + pi_w + pi_z)**2*(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z)) - 4*beta*k*pi_w*pi_z*sigma_w**2*(-k*pi_w + 1)/((beta/k + pi_w + pi_z)*(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z)) + 2*k**2*pi_z*sigma_z**2 + 2*k*pi_z**2*sigma_w**2*(alpha + k*pi_w)*(-2*alpha**2 - 2*alpha*k*pi_w)/(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z)**2 + 4*k*pi_z*sigma_w**2*(alpha + k*pi_w)/(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z))/(2*k*(beta*pi_w + pi_z)*(beta*pi_w*(-k*pi_w + 1)/(k*(beta*pi_w + pi_z)*(beta/k + pi_w + pi_z)) + 1)) - (-2*beta*k*pi_w*pi_z**2*sigma_w**2*(-k*pi_w + 1)/((beta/k + pi_w + pi_z)*(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z)) + k**2*pi_z**2*sigma_z**2 + 2*k*pi_z**2*sigma_w**2*(alpha + k*pi_w)/(2*alpha**3/k + 2*alpha**2*beta*pi_w + 2*alpha**2*pi_w + 2*alpha**2*pi_z + 2*alpha*beta*pi_w + 2*alpha*k*pi_w*pi_z))/(2*k*(beta*pi_w + pi_z)**2*(beta*pi_w*(-k*pi_w + 1)/(k*(beta*pi_w + pi_z)*(beta/k + pi_w + pi_z)) + 1))

sigma_z_min = 0.00001
sigma_z_max = .02
steps = 100

sigma_z_range = np.arange(sigma_z_min, sigma_z_max, (sigma_z_max - sigma_z_min) / steps)

y = np.zeros((steps, ))

for i in range(steps):
    sigma_z = sigma_z_range[i]
    y[i] = sp.optimize.fsolve(dMsedpi_z, 100)

plt.figure()
plt.plot(sigma_z_range, y)
plt.plot(sigma_z_range[y.argmax()], y.max(), marker="o", color='r')
plt.title('Sigma_z = ' + str(sigma_z_range[y.argmax()]) + ', pi_z = ' + str(y.max()))

#dMSEdpi_z = sy.diff(omega11 - 2 * omega31 + omega33, pi_z, 1)
#dMSEdpi_z_simplified = sy.simplify(dMSEdpi_z)
#dMsedpi_z_expanded = sy.expand(dMSEdpi_z)
#dMSEdpi_z_common = sy.ratsimp(dMSEdpi_z)
#dMsedpi_z_collect = sy.collect(dMSEdpi_z, pi_z)

#opt_pi_z = sy.solve(dMSEdpi_z, pi_z)

#print(dMSEdpi_z)
#print(dMsedpi_z_expanded)
#print(dMSEdpi_z_common)
#print(dMsedpi_z_collect)
#print(opt_pi_z)


















