#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from system import WaveFunction
from vmc import VMC


def exact_energy(omega, n_particles, dim):
    """
    Minimal local energy found by setting alpha = omega / 2, which yields
    E_L = (omega * dim * n_particles) / 2
    """
    return (omega * dim * n_particles) / 2


# System
#N = int(1e4)
N = 20
d = 3
omega = 1
psi = WaveFunction(omega, N, d)

exact_E = exact_energy(omega, N, d)
print(f"Exact energy: {exact_E}")

# Sampler
vmc_sampler = VMC(psi)
ncycles = 100000
alpha_step = 0.05
alphas = np.arange(0.1, 1 + alpha_step, alpha_step)


energies, variances = vmc_sampler.sample(ncycles,
                                         alphas,
                                         proposal_dist='normal',
                                         scale=0.8)

'''
for alpha, energy in zip(alphas, energies):
    print(f"{alpha=:.2f}, {energy=:.2f}")
'''

E_min = np.min(energies)
alpha_min = alphas[np.argmin(energies)]
print(f"{alpha_min=:.2f}, {E_min=:.2f}")

fig, ax = plt.subplots()
ax.plot(alphas, energies, label='VMC')
ax.axhline(exact_E, ls='--', color='r', label='Exact')
ax.set(xlabel=r'$\alpha$', ylabel='Energy')
ax.legend()
plt.show()
