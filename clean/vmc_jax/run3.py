#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from functools import partial

import jax as jax
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
        self.grad_wf = grad(self.wf_val, argnums=0, holomorphic=False)
        if dim == 3:
            self.lap = self.spherical_laplacian
        if dim == 2:
            self.lap = self.polar_lap

    @partial(jit, static_argnums=(0,))
    def __call__(self, r, alpha):
        return jnp.exp(-alpha * r**2)

    def wf_val(self, r, alpha):
        return jnp.exp(-alpha * jnp.sum(r**2))

    @partial(jit, static_argnums=(0,))
    def pdf(self, r, alpha):
        return self(r, alpha)**2

    @partial(jit, static_argnums=(0,))
    def logpdf(self, r, alpha):
        return jnp.log(self(r, alpha)**2)

    def convert_to_distance(self, r):
        distance_squared = jnp.sum(r**2, axis=1)
        distance = jnp.sqrt(distance_squared)
        return distance

    @partial(jit, static_argnums=(0,))
    def spherical_laplacian(self, r, alpha):
        grad2_wf = jax.jacfwd(self.spherical_middle,
                              argnums=0, holomorphic=False)
        laplacian = jnp.sum(jnp.diag(grad2_wf(r, alpha) / (r * r)))
        return laplacian

    def spherical_middle(self, r, alpha):
        return r * r * self.grad_wf(r, alpha)

    def polar_lap(self, r, alpha):
        grad2_wf = jax.jacfwd(self.polar_middle, argnums=0, holomorphic=False)
        laplacian = jnp.sum(jnp.diag(grad2_wf(r, alpha) / r))
        return laplacian

    def polar_middle(self, r, alpha):
        return r * self.grad_wf(r, alpha)

    def hamiltonian(self, r, alpha):
        #kinetic = -0.5*jnp.sum(jnp.diag(jnp.sum(jnp.sum(self.hessian(r, alpha), axis=1), axis=-1)))
        kinetic = -0.5 * self.lap(r, alpha)
        return kinetic + 0.5 * self._omega2 * jnp.sum(r**2) * self.wf_val(r, alpha)

    @partial(jit, static_argnums=(0,))
    def local_energy(self, r, alpha):
        r = self.convert_to_distance(r)
        H = self.hamiltonian(r, alpha)
        return H / self.wf_val(r, alpha)

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
N = 500     # Number of particles
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
n_chains = 4
scale = 1.0

# Run sampler
energies = []

start = time.time()
for i, alpha in enumerate(alphas):
    initial_positions = safe_initial_state(wf, alpha, seed_init)
    #print("Init pos: ", initial_positions.shape)
    #print(wf.local_energy(initial_positions, alpha))
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
