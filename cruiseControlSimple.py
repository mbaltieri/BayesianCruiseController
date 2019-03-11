#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:45:19 2018

Simplified code for passive tracker, chapter 4

@author: manuelbaltieri
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin

### define font size for plots ###
#
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)            # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)       # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title
#

dt = .05
T = 10
iterations = int(T / dt)
plt.close('all')

alpha = 1

def cruiseControlSimple(simulation, precision_strength):
    gamma_w = 2
    pi_w = np.exp(gamma_w)
    sigma_w = np.sqrt(1/pi_w)
    
    gamma_z = 1*np.array([[1, 0], [0, 1]])
    pi_z = np.exp(gamma_z)
    pi_z = np.diag(np.diag(pi_z))
    sigma_z = np.linalg.inv(splin.sqrtm(pi_z))
    
    w = np.random.randn(iterations, 1) * sigma_w
    z = np.dot(np.random.randn(iterations, 2), sigma_z)
    
    x = np.zeros((iterations, 2))
    x[0,0] = 20.0
    
    if simulation == 0 or simulation == 2:
        mu_gamma_w = -12
    elif simulation == 1 or simulation == 3:
        mu_gamma_w = 2
    mu_pi_w = np.exp(mu_gamma_w)
    
    if simulation == 0:
        if precision_strength == 0:
            mu_gamma_z = 3*np.array([[1, 0], [0, 1]])
        elif precision_strength == 1:
            mu_gamma_z = 1*np.array([[1, 0], [0, 1]])
        elif precision_strength == 2:
            mu_gamma_z = 0*np.array([[1, 0], [0, 1]])
    if simulation == 2:
        mu_gamma_z = 1*np.array([[1, 0], [0, 1]])
    elif simulation == 1 or simulation == 3:
        mu_gamma_z = -12*np.array([[1, 0], [0, 1]])
    mu_pi_z = np.exp(mu_gamma_z)
    mu_pi_z = np.diag(np.diag(mu_pi_z))
    
    psi = np.zeros((iterations, 2))
    a = np.zeros((iterations, 1))
    
    mu_x = np.zeros((iterations, 2))
    mu_v = np.zeros((iterations, 2))
    mu_v = 10*np.ones((iterations, 2))
    
    xi_z = np.zeros((iterations, 2))
    xi_w = np.zeros((iterations, 1))
    
    F = np.zeros((iterations, 1))
    
    if simulation == 3:
        k_a = np.exp(15)
    else:
        k_a = 1.
    
    for i in range(iterations-1):
        x[i, 1] = - alpha * x[i, 0] + a[i] + w[i]
        x[i+1, 0] = x[i, 0] + dt * (- alpha * x[i, 0] + a[i] + w[i]/np.sqrt(dt))
        
        psi[i, :] = x[i, :] + z[i, :]
        
        # perception
        dFdmu_x = np.array([[-mu_pi_z[0,0]*(psi[i,0]-mu_x[i,0]) + mu_pi_w*alpha*(mu_x[i,1]+alpha*mu_x[i,0]-mu_v[i,0]),
                            -mu_pi_z[1,1]*(psi[i,1]-mu_x[i,1]) + mu_pi_w*(mu_x[i,1]+alpha*mu_x[i,0]-mu_v[i,0])]])
        Dmu_x = np.array([[mu_x[i,1], 0]])
        mu_x[i+1, :] = mu_x[i, :] + dt * (Dmu_x - dFdmu_x)
        
        # action
        if simulation > 1:
            dFda = mu_pi_z[0,0]*(psi[i,0]-mu_x[i,0]) + mu_pi_z[1,1]*(psi[i,1]-mu_x[i,1])
            a[i+1] = a[i] + dt * k_a * - dFda
        
        # weighted predictions errors
        xi_z[i, :] = np.dot(psi[i,:]-mu_x[i,:], mu_pi_z[0,0])
        xi_w[i] = mu_pi_w*(mu_x[i,1]+alpha*mu_x[i,0]-mu_v[i,0])
        
        F[i] = .5 * (np.dot(np.dot(xi_z[i, :], mu_pi_z), xi_z[i, :].transpose()) + xi_w[i]*mu_pi_w*xi_w[i] - np.log(mu_pi_z[0,0]*mu_pi_z[0,0]*mu_pi_w))
    
    return psi, x, a, mu_x, xi_z, xi_w, F

# simulations:
# 0: passive tracker
#    0: strong sensory expected precision
#    1: intermediate sensory expected precision
#    2: weak sensory expected precision
# 1: passive dreamer
# 2: active tracker
# 3: active dreamer

simulation = 3
precision_strength = 1          # only simulation 0

psi, x, a, mu_x, xi_z, xi_w, F = cruiseControlSimple(simulation, precision_strength)


plt.figure(figsize=(9, 6))
plt.title('Block velocity')
plt.plot(np.arange(0, T-dt, dt), psi[:-1, 0], 'b', label='Observed velocity')
plt.plot(np.arange(0, T-dt, dt), x[:-1, 0], 'k', label='Real velocity')
plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], 'r', label='Estimated velocity')
plt.xlim(0, T)
plt.xlabel('Time ($s$)')
plt.ylabel('Velocity ($km/h$)')
plt.legend(loc=1)
plt.savefig("figures/cruiseControlActiveInferenceVelocity.pdf")

plt.figure(figsize=(9, 6))
plt.title('Block acceleration')
plt.plot(np.arange(0, T-dt, dt), psi[:-1, 1], 'b', label='Observed acceleration')
plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 1], 'r', label='Real acceleration')
plt.plot(np.arange(0, T-dt, dt), x[:-1, 1], 'k', label='Estimated acceleration')
plt.xlim(0, T)
plt.xlabel('Time ($s$)')
plt.ylabel('Acceleration ($km/h^2$)')
plt.legend(loc=4)
plt.savefig("figures/cruiseControlActiveInferenceAcceleration.pdf")
 
plt.figure(figsize=(9, 6))
plt.title('Sensory prediction error on velocity')
plt.plot(np.arange(0, T-dt, dt), xi_z[:-1, 0])
plt.xlim(0, T)
plt.xlabel('Time ($s$)')
plt.ylabel('a.u.')
plt.savefig("figures/cruiseControlActiveInferenceSensoryPEVelocity.pdf")

plt.figure(figsize=(9, 6))
plt.title('Sensory prediction error on acceleration')
plt.plot(np.arange(0, T-dt, dt), xi_z[:-1, 1])
plt.xlim(0, T)
plt.xlabel('Time ($s$)')
plt.ylabel('a.u.')
plt.savefig("figures/cruiseControlActiveInferenceSensoryPEAcceleration.pdf")

plt.figure(figsize=(9, 6))
plt.title('System prediction error')
plt.plot(np.arange(0, T-dt, dt), xi_w[:-1])
plt.xlim(0, T)
plt.xlabel('Time ($s$)')
plt.ylabel('a.u.')
plt.savefig("figures/cruiseControlActiveInferenceDynamicPE.pdf")

plt.figure(figsize=(9, 6))
plt.title('Variational free energy')
plt.semilogy(np.arange(0, T-dt, dt), F[:-1])
plt.xlim(0, T)
plt.xlabel('Time ($s$)')
plt.ylabel('a.u.')
plt.savefig("figures/cruiseControlActiveInferenceFE.pdf")

plt.figure(figsize=(9, 6))
plt.title('Action')
plt.plot(np.arange(0, T-dt, dt), a[:-1])
plt.xlim(0, T)
plt.xlabel('Time ($s$)')
plt.ylabel('Acceleration ($km/h^2$)')
plt.savefig("figures/cruiseControlActiveInferenceAction.pdf")