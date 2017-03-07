#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:45:03 2017

Testing the range of noise acceptable to perform non trivial inference in a 
system of SDE with 3 variables, not very dissimilar to the model I want to use
for inference in a free energy framework


@author: mb540
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 200
iterations = int(T / dt)
plt.close('all')

variables = 3

w = 1.

alpha_min = 1
alpha_max = 1000.
steps = 100

alpha_range = np.arange(alpha_min, alpha_max, (alpha_max - alpha_min) / steps)

beta = 1.0

MSE = np.zeros((steps))


sigma_x = 0.12
sigma_y = 1
dW = np.random.randn(variables, iterations)

X = np.zeros((variables, iterations))
P = np.array([[1., 1., 1.], [1., 1., 0.], [0., 0., 1.]])                                            # connectivity parameters without sign
C = np.array([[- P[0, 0], P[0, 1], P[0, 2]], [- P[1, 0], - P[1, 1], 0.], [0., 0., - P[2, 2]]])        # connectivity with sign
sigma = np.array([[sigma_x, 0., 0.], [0., 0., 0.], [0., 0., sigma_y]])
SIGMA = - np.dot(sigma.transpose(), sigma)

OMEGA = sp.linalg.solve_lyapunov(C, SIGMA)



OMEGA_TEST = np.array([[1 / (2 * P[0, 0] + P[0, 1] * P[1, 0] / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1])) *
                     (sigma_x ** 2 + P[0, 2] ** 2 * sigma_y ** 2 / (P[2, 2] * ((P[0, 0] + P[2, 2]) * (P[1, 1] + P[2, 2]) + P[0, 1] * P[1, 0])) *
                     (P[1, 1] + P[2, 2] - P[0, 1] * P[1, 0] / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1]))),
                     (- P[1, 0] * 1 / (2 * P[0, 0] + P[0, 1] * P[1, 0] / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1])) *
                     (sigma_x ** 2 + P[0, 2] ** 2 * sigma_y ** 2 / (P[2, 2] * ((P[0, 0] + P[2, 2]) * (P[1, 1] + P[2, 2]) + P[0, 1] * P[1, 0])) *
                     (P[1, 1] + P[2, 2] - P[0, 1] * P[1, 0] / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1]))) - 
                     P[0, 2] ** 2 * sigma_y ** 2 * P[1, 0] / (2 * P[2, 2] * ((P[0, 0] + P[2, 2]) * (P[1, 1] + P[2, 2]) + 
                     P[0, 1] * P[1, 0]))) * 1 / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1]),
                     P[0, 2] * sigma_y ** 2 * (P[1, 1] + P[2, 2]) / (2 * P[2, 2] * ((P[0, 0] + P[2, 2]) * (P[1, 1] + P[2, 2]) + P[0, 1] * P[1, 0]))],
                     [(- P[1, 0] * 1 / (2 * P[0, 0] + P[0, 1] * P[1, 0] / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1])) *
                     (sigma_x ** 2 + P[0, 2] ** 2 * sigma_y ** 2 / (P[2, 2] * ((P[0, 0] + P[2, 2]) * (P[1, 1] + P[2, 2]) + P[0, 1] * P[1, 0])) *
                     (P[1, 1] + P[2, 2] - P[0, 1] * P[1, 0] / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1]))) - 
                     P[0, 2] ** 2 * sigma_y ** 2 * P[1, 0] / (2 * P[2, 2] * ((P[0, 0] + P[2, 2]) * (P[1, 1] + P[2, 2]) + 
                     P[0, 1] * P[1, 0]))) * 1 / (P[0, 0] + P[1, 1] + P[0, 1] * P[1, 0] / P[1, 1]),
                     - P[1, 0] / P[1, 1] * OMEGA[1, 0], 
                     - P[1, 0] / (P[1, 1] + P[2, 2]) * P[0, 2] * sigma_y ** 2 * (P[1, 1] + P[2, 2]) / (2 *
                     P[2, 2] * ((P[0, 0] + P[2, 2]) * (P[1, 1] + P[2, 2]) + P[0, 1] * P[1, 0]))],
                     [P[0, 2] * sigma_y ** 2 / (2 * P[2, 2] * (P[0, 0] + P[2, 2] + P[0, 1] * P[1, 0] / (P[1, 1] + P[2, 2]))),
                     - P[1, 0] / (P[1, 1] + P[2, 2]) * P[0, 2] * sigma_y ** 2 / (2 * P[2, 2] * (P[0, 0] + P[2, 2] + P[0, 1] * P[1, 0] / (P[1, 1] + P[2, 2]))),
                     sigma_y ** 2 / (2 * P[2, 2])]])
                    
#    OMEGA = OMEGA_TEST
#MSE[i] = OMEGA[0, 0] - 2 * OMEGA[0, 1] + OMEGA[1, 1]