#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:29:36 2017

Passive dreamer, thesis chapter.
Simulation of an active inference agent-based model for perception (no action) with bias for top-down processing.
The generative process producing data is a simple OU process.

@author: mb540
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:43:31 2017



@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 10
iterations = int(T / dt)
dx = np.exp(-8)

obs_states = 1
hidden_states = 1                               # x, in Friston's work
hidden_causes = 1                               # v, in Friston's work
states = obs_states + hidden_states
temp_orders_states = 2
temp_orders_causes = 2

small_pi = np.exp(-30)                          # virtually 0 precision

# model details #

x = np.zeros((hidden_states, temp_orders_states))
x = np.random.randn(hidden_states, temp_orders_states)
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
v = np.zeros((hidden_causes, temp_orders_states - 1))
y = np.zeros((obs_states, temp_orders_states - 1))
eta = np.zeros((hidden_causes, temp_orders_states - 1))
eta[0, 0] = 0.

# Free Energy definition #
FE = np.zeros((iterations,))

mu_x = np.random.randn(hidden_states, temp_orders_states)
#mu_x = np.zeros((hidden_states, temp_orders_states))
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
mu_v = np.random.randn(hidden_causes, temp_orders_states)
#mu_v = np.zeros((hidden_causes, temp_orders_states))
mu_gamma_z = 10 * np.ones((obs_states, temp_orders_states))
mu_gamma_z = np.random.randn(obs_states, temp_orders_states)
mu_gamma_z = np.zeros((obs_states, temp_orders_states))
mu_gamma_z_dot = np.zeros((obs_states, temp_orders_states))

mu_gamma_w = np.zeros((obs_states, temp_orders_states))

a = 0

# minimisation variables and parameters
dFdmu_x = np.zeros((hidden_states, temp_orders_states))
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
dFdmu_v = np.zeros((hidden_causes, temp_orders_states))
dFdmu_gamma_z = np.zeros((obs_states, temp_orders_states))
Dmu_x = np.zeros((hidden_states, temp_orders_states))
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
Dmu_v = np.zeros((hidden_causes, temp_orders_states))
eta_mu_x = .001 * np.ones((hidden_states, temp_orders_states))
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
eta_mu_v = .00001 * np.ones((hidden_causes, temp_orders_states))
eta_a = .01
eta_mu_gamma_z = .1

# noise on sensory input
gamma_z = -16 * np.ones((obs_states, temp_orders_states - 1))  # log-precisions
gamma_z[0, 0] = 8
gamma_z = 2 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
pi_z = np.exp(gamma_z) * np.ones((obs_states, temp_orders_states - 1))
# knock out terms that are not directly sensed
# pi_z[:,1:] = np.exp(-16)*np.ones((obs_states,temp_orders_states-2))
sigma_z = 1 / (np.sqrt(pi_z))
z = np.zeros((iterations, obs_states, temp_orders_states - 1))
for i in range(obs_states):
    for j in range(temp_orders_states - 1):
        z[:, i, j] = sigma_z[i, j] * np.random.randn(1, iterations)

# noise on motion of hidden states
gamma_w = 8                                                 # log-precision
pi_w = np.exp(gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
sigma_w = 1 / (np.sqrt(pi_w))
w = np.zeros((iterations, hidden_states, temp_orders_states - 1))
for i in range(hidden_states):
    for j in range(temp_orders_states - 1):
        w[:, i, j] = sigma_w[i, j] * np.random.randn(1, iterations)

# noise on causes
gamma_n = 15                                                  # log-precision
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
pi_n = np.exp(gamma_n) * np.ones((hidden_causes, temp_orders_states - 1))
pi_n[:, temp_orders_causes - 1:] = small_pi * np.ones((hidden_causes, temp_orders_states - temp_orders_causes))
sigma_n = 1 / (np.sqrt(pi_n))
n = np.zeros((iterations, hidden_causes, temp_orders_causes - 1))
for i in range(hidden_causes):
    for j in range(temp_orders_causes - 1):
        n[:, i, j] = sigma_n[i, j] * np.random.randn(1, iterations)

# history
x_history = np.zeros((iterations, hidden_states, temp_orders_states))
y_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
v_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))

rho_history = np.zeros((iterations, obs_states, temp_orders_states - 1))

mu_x_history = np.zeros((iterations, hidden_states, temp_orders_states))
# keep temp_orders_states-temp_orders_causes empty (= 0) to ease calculations
mu_v_history = np.zeros((iterations, hidden_states, temp_orders_states))
mu_gamma_z_history = np.zeros((iterations, obs_states, temp_orders_states))

a_history = np.zeros((iterations,))

eta_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))

xi_z_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
xi_w_history = np.zeros((iterations, hidden_states, temp_orders_states - 1))
# keep temp_orders_states-temp_orders_causes empty (=0) to ease calculations
xi_n_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))

# functions #
# free energy functions
def g(x, v):
    return x


def f(x, v, w, a):
    action = np.zeros((1, temp_orders_states - 1))
    action[0, 0] = a
#    return x - np.power(x,3) + w/np.sqrt(dt) + v
    return - x + v + w/np.sqrt(dt)


def g_gm(x, v):
#    return g(x, v)
    return g(x, v)


def f_gm(x, v, w):
    # return (force_drive(x, v) - force_disturbance(x, theta)) / m
#    return - x + v
    return f(x, v, w, .0)


def getObservation(x, v, w, a):
    x[:, 1:] = f(x[:, :-1], v, w, a)  # + w[i, 1:, :]
    x[:, 0] += dt * x[:, 1]
    return g(x[:, :-1], v)  # + np.squeeze(z[i, :, :])


def sensoryErrors(y, mu_x, mu_v, mu_gamma_z):
    eps_z = y - g_gm(mu_x[:, :-1], mu_v)
    pi_gamma_z = np.exp(mu_gamma_z[:, :-1]) * np.ones((obs_states, temp_orders_states - 1))
    xi_z = pi_gamma_z * eps_z
    return eps_z, xi_z


def dynamicsErrors(mu_x, mu_v, mu_gamma_w):
    eps_w = mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], mu_gamma_w)
    # eps_w = np.zeros((hidden_states, temp_orders_states - 1)) - f_gm(mu_x[:, :-1],mu_v[:, :-1])
    xi_w = pi_w * eps_w
    return eps_w, xi_w


def priorErrors(mu_v, eta):
    eps_n = mu_v[:, :-1] - eta
    xi_n = pi_n * eps_n
    return eps_n, xi_n


def FreeEnergy(y, mu_x, mu_v, mu_gamma_z, mu_gamma_w, eta):
    eps_z, xi_z = sensoryErrors(y, mu_x, mu_v, mu_gamma_z)
    eps_w, xi_w = dynamicsErrors(mu_x, mu_v, mu_gamma_w)
    eps_n, xi_n = priorErrors(mu_v, eta)
    return .5 * (np.trace(np.dot(eps_z, np.transpose(xi_z))) +
                 np.trace(np.dot(eps_w, np.transpose(xi_w))) +
                 np.trace(np.dot(eps_n, np.transpose(xi_n))) -
                 np.log(np.prod(np.exp(mu_gamma_z[:, :-1])) *
                        np.prod(pi_w) * np.prod(pi_n)))


# forget about precisions momentarely
mu_gamma_z[:, :-1] = gamma_z[0, 0] * np.ones((obs_states, temp_orders_states - 1))
mu_gamma_w = gamma_w
mu_v[:, :-1] = eta
#mu_gamma_z[0, 0] = -16.
# mu_gamma_z[0, 1] = -16.

for i in range(iterations):
    print(i)

#    v[:,0] = np.exp(-(i-iterations/4)**2/100**2)
    if (temp_orders_states > 2):
        for j in range(1, temp_orders_causes - 1):
            v[:, j] = (v[:, j - 1] - v_history[i - 1, :, j - 1]) / dt
            # analytic noise
            # z[i, :, j] = (z[i, :, j - 1] - z[i - 1, :, j - 1]) / dt
    x_temp = np.copy(x)                      # save y for derivative over a
    y = getObservation(x, v, w[i,:,:], a)

    # free energy #
    # sensory input
    rho = y + z[i, :, :]

    FE[i] = FreeEnergy(rho, mu_x, mu_v, mu_gamma_z, mu_gamma_w, eta)

    # minimisation
    eps_z, xi_z = sensoryErrors(rho, mu_x, mu_v, mu_gamma_z)
    eps_w, xi_w = dynamicsErrors(mu_x, mu_v, mu_gamma_w)
    eps_n, xi_n = priorErrors(mu_v, eta)

    # hidden states
    for j in range(hidden_states):
        for k in range(temp_orders_states):
            mu_x_temp = np.copy(mu_x)
            mu_x_temp[j, k] += dx
            dFdmu_x[j, k] = (FreeEnergy(rho, mu_x_temp, mu_v, mu_gamma_z, mu_gamma_w, eta) - FE[i]) / dx

    for j in range(hidden_states):
        for k in range(temp_orders_states - 1):
            Dmu_x[j, k] = np.copy(mu_x[j, k + 1])

    # causes
    for j in range(hidden_causes):
        for k in range(temp_orders_causes):
            mu_v_temp = np.copy(mu_v)
            mu_v_temp[j, k] += dx
            dFdmu_v[j, k] = (FreeEnergy(rho, mu_x, mu_v_temp, mu_gamma_z, mu_gamma_w, eta) - FE[i]) / dx

    for j in range(hidden_causes):
        for k in range(temp_orders_causes - 1):
            Dmu_v[j, k] = np.copy(mu_v[j, k + 1])

    # precisions (only on sensors at the moment)
    for j in range(obs_states):
        for k in range(temp_orders_states-1):
            mu_gamma_z_temp = np.copy(mu_gamma_z)
            mu_gamma_z_temp[j, k] += dx
            dFdmu_gamma_z[j, k] = (FreeEnergy(rho, mu_x, mu_v, mu_gamma_z_temp, mu_gamma_w, eta) - FE[i]) / dx

    # action
    # y_adx = getObservation(x_temp, v, a + dx)
    # rho_adx = y_adx + z[i, :, :]
    # FE_adx = FreeEnergy(rho_adx, mu_x, mu_v, eta)
    # dFda = (FE_adx - FE[i]) / dx
    dFda = np.dot(xi_z, np.ones((temp_orders_states - 1, 1)))

    # update system
    mu_x += dt * (Dmu_x - eta_mu_x * dFdmu_x)
    mu_v[:, :-1] += dt * (Dmu_v[:, :-1] - eta_mu_v[:, :-1] * dFdmu_v[:, :-1])
#    a += dt * - eta_a * dFda
#    mu_gamma_z_dot += dt*-eta_mu_gamma_z*(dFdmu_gamma_z + 1*mu_gamma_z_dot)
#    mu_gamma_z += dt*(mu_gamma_z_dot)
#    mu_gamma_z_dot[0, 0] += dt * - eta_mu_gamma_z * (dFdmu_gamma_z[0, 0] + (i / iterations + .01) * mu_gamma_z_dot[0, 0])
#    mu_gamma_z_dot[0, 0] += dt * - eta_mu_gamma_z * (.5 * (mu_gamma_z[0, 0] * xi_z[0, 0]*eps_z[0, 0] - 1) + (i / iterations + .01) * mu_gamma_z_dot[0, 0])
#    mu_gamma_z[0, 0] += dt * (mu_gamma_z_dot[0, 0])
    # mu_gamma_z_dot[0, 1] += dt * - eta_mu_gamma_z * (dFdmu_gamma_z[0, 1] + (i / iterations) * mu_gamma_z_dot[0, 1])
    # mu_gamma_z[0, 1] += dt * (mu_gamma_z_dot[0, 1])

    # save history
    x_history[i, :, :] = x
    y_history[i, :] = y
    rho_history[i, :] = rho
    v_history[i] = v
    mu_x_history[i, :, :] = mu_x
    mu_v_history[i, :, :] = mu_v
    mu_gamma_z_history[i, :, :] = mu_gamma_z
    a_history[i] = a
    eta_history[i] = eta

    xi_z_history[i, :, :] = xi_z
    xi_w_history[i, :, :] = xi_w
    xi_n_history[i, :, :] = xi_n


plt.close('all')

fig0 = plt.figure(0)
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, iterations*dt, dt), y_history[:, :, 0])
plt.title('Observed data')
plt.subplot(2, 2, 2)
plt.plot(np.arange(0, iterations*dt, dt), x_history[:, :, 0])
plt.title('Hidden states')
plt.subplot(2, 2, 3)
plt.plot(np.arange(0, iterations*dt, dt), v_history[:, :, 0])
plt.title('Hidden cause')
fig0.savefig('fig0.eps', format='eps', dpi=1200)

fig2 = plt.figure(2)
plt.semilogy(np.arange(0, iterations*dt, dt), FE)
#plt.title('Free energy')
plt.ylabel('Free energy')
plt.xlabel('Time')
fig2.savefig('fig2.eps', format='eps', dpi=1200)

fig3 = plt.figure(3)
#plt.suptitle('Beliefs about observable states')
y_reconstructed = np.zeros((iterations, obs_states, temp_orders_states - 1))
for i in range(iterations):
    y_reconstructed[i, :, :] = g_gm(mu_x_history[i, :, :-1], mu_v_history[i, :, :-1])
for i in range(hidden_states):
    for j in range(temp_orders_states - 1):
        plt.subplot(hidden_states, temp_orders_states - 1, (temp_orders_states - 1) * i + j + 1)
        plt.plot(np.arange(0, iterations*dt, dt), rho_history[:, i, j], 'b', label='Sensory data')
        plt.plot(np.arange(0, iterations*dt, dt), y_reconstructed[:, i, j], 'r', label='Agent\'s belief')
plt.xlabel('Time')
plt.legend()
fig3.savefig('fig3.eps', format='eps', dpi=1200)

fig4 = plt.figure(4)
#plt.suptitle('Beliefs about hidden states')
for i in range(hidden_states):
    for j in range(temp_orders_states - 1):
        plt.subplot(hidden_states, temp_orders_states - 1, (temp_orders_states - 1) * i + j + 1)
        plt.plot(np.arange(0, iterations*dt, dt), x_history[:, i, j], 'b', label='Agent\'s belief')
        plt.plot(np.arange(0, iterations*dt, dt), mu_x_history[:, i, j], 'r', label='Agent\'s prior')
plt.xlabel('Time')
plt.legend()
fig4.savefig('fig4.eps', format='eps', dpi=1200)

plt.figure(5)
plt.suptitle('Beliefs about hidden causes')
for i in range(hidden_causes):
    for j in range(temp_orders_causes - 1):
        plt.subplot(hidden_causes, temp_orders_causes - 1,(temp_orders_causes - 1) * i + j + 1)
        plt.plot(range(iterations), v_history[:, i, j], 'b', range(iterations), mu_v_history[:, i, j], 'r', range(iterations), eta_history[:, i, j], 'k')
#
#plt.figure(6)
#plt.suptitle('Beliefs about hidden causes + action')
#plt.plot(range(iterations), v_history[:, 0, 0], 'b', range(iterations), mu_v_history[:, 0, 0], 'r', range(iterations), eta_history[:, i, j], 'k', a_history, 'g')
#
#plt.figure(7)
#plt.suptitle('Action')
#plt.plot(a_history)

fig8 = plt.figure(8)
plt.suptitle('(Weighted) Bottom-up errors, xi_z')
for i in range(obs_states):
    for j in range(temp_orders_states - 1):
        plt.subplot(obs_states, (temp_orders_states - 1), (temp_orders_states - 1) * i + j + 1)
        plt.plot(np.arange(0, iterations*dt, dt), xi_z_history[:, i, j], 'b')
plt.xlabel('Time')
fig8.savefig('fig8.eps', format='eps', dpi=1200)

fig9 = plt.figure(9)
plt.suptitle('(Weighted) Top-down errors, hidden states, xi_w')
for i in range(hidden_states):
    for j in range(temp_orders_states - 1):
        plt.subplot(hidden_states, (temp_orders_states - 1), (temp_orders_states - 1) * i + j + 1 )
        plt.plot(np.arange(0, iterations*dt, dt), xi_w_history[:, i, j], 'b')
plt.xlabel('Time')
fig9.savefig('fig9.eps', format='eps', dpi=1200)

#plt.figure(10)
#plt.suptitle('(Weighted) Top-down errors, hidden causes, xi_n')
#for i in range(hidden_causes):
#    for j in range(temp_orders_causes - 1):
#        plt.subplot(hidden_causes, (temp_orders_causes - 1) , (temp_orders_causes - 1) * i + j + 1)
#        plt.plot(range(iterations), xi_n_history[:, i, j], 'b')

#plt.figure(11)
#plt.suptitle('Log-precisions, gamma_z')
#for i in range(obs_states):
#    for j in range(temp_orders_causes - 1):
#        plt.subplot(obs_states,(temp_orders_causes - 1),(temp_orders_causes - 1) * i + j + 1)
#        plt.plot(range(iterations), gamma_z[i, j] * np.ones(iterations,), 'b', range(iterations), mu_gamma_z_history[:, i, j], 'r')

plt.show()
