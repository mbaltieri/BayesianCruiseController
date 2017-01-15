#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:11:30 2017

@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 10
iterations = int(T / dt)

t_dot = np.random.randn()
t = .5
t_dot_history = np.zeros((iterations,))
t_history = np.zeros((iterations,))

for i in range(iterations):
    t_dot = - .75 * t + .15 / (np.exp(- np.abs(600)) + 1)
    t += t_dot * dt
    
    t_dot_history[i] = t_dot
    t_history[i] = t

plt.close('all')

plt.figure(0)
plt.plot(t_history)
plt.title('Temperature')