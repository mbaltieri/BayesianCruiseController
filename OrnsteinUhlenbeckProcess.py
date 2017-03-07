#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:58:09 2017

Ornstein-Uhlenbeck process as gradient descent of a linear system, simulations

@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 100
iterations = int(T / dt)

x_dot = 0
x = 0

k = .1
pi_1 = np.exp(4)
pi_2 = np.exp(6)

pi_1_real = np.exp(4)
pi_2_real = np.exp(6)
rho = pi_2_real * np.random.randn(iterations,)
rho = np.random.randn(iterations,)

tau = 1. / k * (pi_1 + pi_2)
sigma = k * pi_1 / (np.sqrt(2 * k * (pi_1 + pi_2)))

x_dot_history = np.zeros((iterations,))
x_history = np.zeros((iterations,))
sin_history = np.zeros((iterations,))

#plt.close('all')

# simplified OU process
tau = .001
sigma = 1

for i in range(iterations):
    sin = np.sin(i / 100)
    x_dot = - x / tau + rho[i] * np.sqrt(2 * sigma**2 / tau) / np.sqrt(dt)
#    x_dot = - k * x * (pi_1 + pi_2) + k * pi_1 * rho[i] / np.sqrt(dt)
#    x_dot = - k * (x * (pi_1 + pi_2) - pi_1 * rho[i])
#    x_dot = - k * (np.exp(pi_1) * (rho[i] - x) * - 1 + np.exp(pi_2) * x)
    x_dot = - x + rho[i] / np.sqrt(dt)
    x += dt *  x_dot
    
    x_dot_history[i] = x_dot
    x_history[i] = x
    sin_history[i] = sin
             
plt.figure()
plt.plot(range(iterations), x_history)

#plt.figure()
#plt.plot(range(iterations), sin_history)

print(np.var(x_history))