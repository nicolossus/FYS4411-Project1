import matplotlib.pyplot as plt
import jax.numpy as jnp
from samplers import MetropolisVMC2jax
from wavefunction import SimpleGaussian, SG
import numpy as np
from jax import random
from jax.config import config

config.update("jax_enable_x64", True)

def wavefunction(r, alpha):
    return jnp.exp(-alpha*jnp.sum(r**2))

def potential(r, _omega):
    return 0.5*_omega*_omega*jnp.sum(r**2)

def exact_energy(n_particles, dim, omega):
    return (omega * dim * n_particles) / 2


def safe_initial_state(wavefunction, alpha, seed=0):
    key = random.PRNGKey(seed)
    N = wavefunction.n_particles
    dim = wavefunction.dim
    #positions = jnp.asarray(rng.random(size=(N, dim)))
    positions = random.uniform(key, shape=(N, dim))

    # safe initialization
    wf2 = wavefunction.density(positions, alpha)
    while jnp.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wavefunction.density(positions, alpha)

    return positions


N = 1
d = 1
omega = 1
wf = SG(wavefunction, potential, N, d, omega)


exact_E = exact_energy(N, d, omega)
print(f"Exact energy: {exact_E}")

vmc_sampler = MetropolisVMC2jax(wf)
ncycles = 10000
alpha_step = 0.1
alphas = jnp.arange(0.1, 1 + alpha_step, alpha_step)

energies = jnp.zeros(alphas.size)

#fig, ax = plt.subplots()

for i, alpha in enumerate(alphas):
    initial_state = safe_initial_state(wf, alpha)
    energies = energies.at[i].set(vmc_sampler.sample(ncycles,
                                     initial_state,
                                     alpha,
                                     scale=0.5,
                                     burn=0,
                                     tune=True,
                                     tune_iter=8000,
                                     tune_interval=250,
                                     optimize=False,
                                     tol_scale=1e-5
                                     ))
    accept_rate = vmc_sampler.accept_rate
    print(f"{alpha=:.2f}, E={energies[i]:.2f}, {accept_rate=:.2f}")
    #ax.plot(vmc_sampler.energy_samples, label=f'{alpha=:.1f}')
    # ax.legend()

E_min = jnp.min(energies)
alpha_min = alphas[jnp.argmin(energies)]
print(f"{alpha_min=:.2f}, {E_min=:.2f}")

fig, ax = plt.subplots()
ax.plot(alphas, energies, label='VMC')
ax.axhline(exact_E, ls='--', color='r', label='Exact')
ax.set(xlabel=r'$\alpha$', ylabel='Energy')
ax.legend()
plt.show()
