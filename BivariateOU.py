#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:51:55 2017

Bi-variate OU process to check what the best strategy is for one process to
track the other one (get the mean vs. follow instantaneously)

Conclusions:
- follow instantaneously is better than just track the mean

@author: mb540
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 200
iterations = int(T / dt)
plt.close('all')

variables = 2

w = 1.

alpha_min = 1
alpha_max = 10.
steps = 10

alpha_range = np.arange(alpha_min, alpha_max, (alpha_max - alpha_min) / steps)

beta = 1.0

MSE = np.zeros((steps))

for i in range(steps):
    sigma_x = .1
    sigma_y = .1
    dW = np.random.randn(variables, iterations)
    
    X = np.zeros((variables, iterations))
    C = np.array([[- alpha_range[i], w], [0., - beta]])
    sigma = np.array([[sigma_x, .0], [.0, sigma_y]])
    SIGMA = - np.dot(sigma.transpose(), sigma)
    
    OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
    
    MSE[i] = OMEGA[0, 0] - 2 * OMEGA[0, 1] + OMEGA[1, 1]

plt.figure()
plt.plot(alpha_range, MSE)
plt.plot(alpha_range[MSE.argmin()], MSE.min(), marker="o", color='r')

#for j in range(steps):
#    C = np.array([[- alpha_range[j], w], [0., - beta]])
#    X = np.zeros((variables, iterations))
#    for i in range(iterations - 1):
#        dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
#        X[:, i + 1] = X[:, i] + dt * dX
#    
##    plt.figure()
##    plt.plot(np.arange(0, iterations*dt, dt), X[1, :], 'b')
##    plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')
#    MSE[j] = np.mean((X[1, :] - X[0, :]) ** 2)
#
#
#plt.figure()
#plt.plot(alpha_range, MSE)
#plt.plot(alpha_range[MSE.argmin()], MSE.min(), marker="o", color='r')


#tau_min = .001
#tau_max = 5.
#steps = 10
#
#tau_range = np.arange(tau_min, tau_max, (tau_max - tau_min) / steps)
#
#
#for i in range(steps):
#    sigma_x = .1
#    sigma_y = .1
#    dW = np.random.randn(variables, iterations)
#    
#    X = np.zeros((variables, iterations))
#    C = np.array([[- 1., w] / tau_range[i], [0., - beta]])
#    sigma = np.array([[sigma_x, .0], [.0, sigma_y]])
#    SIGMA = - np.dot(sigma.transpose(), sigma)
#    
#    OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)
#    
#    MSE[i] = OMEGA[0, 0] - 2 * OMEGA[0, 1] + OMEGA[1, 1]
#
#plt.figure()
#plt.plot(alpha_range, MSE)
#plt.plot(alpha_range[MSE.argmin()], MSE.min(), marker="o", color='r')
#
#
#for j in range(steps):
#    C = np.array([[- 1., w] / tau_range[j], [0., - beta]])
#    X = np.zeros((variables, iterations))
#    for i in range(iterations - 1):
#        dX = np.dot(C, X[:, i]) + np.dot(sigma, dW[:, i]) / np.sqrt(dt)
#        X[:, i + 1] = X[:, i] + dt * dX
#    
##    plt.figure()
##    plt.plot(np.arange(0, iterations*dt, dt), X[1, :], 'b')
##    plt.plot(np.arange(0, iterations*dt, dt), X[0, :], 'r')
#    MSE[j] = np.mean((X[1, :] - X[0, :]) ** 2)
#
#
#plt.figure()
#plt.plot(alpha_range, MSE)
#plt.plot(alpha_range[MSE.argmin()], MSE.min(), marker="o", color='r')





