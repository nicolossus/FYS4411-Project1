#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from jax import grad, jit, lax, pmap, random, vmap
from metropolis_jax import vmc_jax


class JaxWF:
    def __init__(self, n_particles, dim, omega):
        self._N = n_particles
        self._d = dim
        self._omega = omega
        # precompute
        self._Nd = self._N * self._d
        self._omega2 = self._omega**2

    @partial(jit, static_argnums=(0,))
    def __call__(self, r, alpha):
        return jnp.exp(-alpha * r**2)

    @partial(jit, static_argnums=(0,))
    def pdf(self, r, alpha):
        return self(r, alpha)**2

    @partial(jit, static_argnums=(0,))
    def logpdf(self, r, alpha):
        return jnp.log(self(r, alpha)**2)

    @partial(jit, static_argnums=(0,))
    def locE(self, r, alpha):
        E_L = self._Nd * alpha + \
            (0.5 * self._omega2 - 2 * alpha**2) * jnp.sum(r**2)
        return E_L

    @property
    def N(self):
        return self._N

    @property
    def dim(self):
        return self._d


def exact_energy(n_particles, dim, omega):
    return 0.5 * omega * dim * n_particles


def safe_initial_state(wavefunction, alpha, seed):
    rng_key = random.PRNGKey(seed)
    N = wavefunction.N
    dim = wavefunction.dim

    positions = random.uniform(rng_key, shape=(N, dim))

    # safe initialization
    wf2 = wavefunction.pdf(positions, alpha)
    while jnp.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wavefunction.pdf(positions, alpha)

    return positions


# Configure wave function
N = 500        # Number of particles
d = 3        # Dimeansion
omega = 1.   # Oscillator frequency
wf = JaxWF(N, d, omega)

# Set variational parameter
alpha_step = 0.1
alphas = jnp.arange(0.1, 1 + alpha_step, alpha_step)

# Configure sampler
seed_sampler = 42
seed_init = 7
n_samples = int(1e4)
n_chains = 1
scale = 1.0

# Run sampler
energies = []

start = time.time()
for i, alpha in enumerate(alphas):
    initial_positions = safe_initial_state(wf, alpha, seed_init)
    energy, _ = vmc_jax(seed_sampler,
                        n_samples,
                        n_chains,
                        initial_positions,
                        wf,
                        scale,
                        alpha)
    energies.append(energy)
end = time.time()
print("Many alpha sampling elapsed time:", end - start)

energies = onp.array(energies)
E_min = onp.min(energies)
alpha_min = alphas[onp.argmin(energies)]
print(f"{alpha_min=:.2f}, {E_min=:.2f}")
exact_energy = exact_energy(N, d, omega)

fig, ax = plt.subplots()
ax.plot(alphas, energies, label='VMC')
ax.axhline(exact_energy, ls='--', color='r', label='Exact')
ax.set(xlabel=r'$\alpha$', ylabel='Energy')
ax.legend()
plt.show()
