#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src import vmc

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


def exact_energy(N, dim, omega):
    return (omega * dim * N) / 2


def non_interact_initial_positions(wavefunction, alpha, N, dim):
    rng = np.random.default_rng()
    positions = rng.random(size=(N, dim))
    # safe initialization
    wf2 = wavefunction.pdf(positions, alpha)
    while np.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wavefunction.pdf(positions, alpha)
    return positions


def interact_initial_positions(wf, alpha, N, dim, a=0.00433):

    rng = np.random.default_rng()

    def corr_factor(r1, r2):
        rij = np.linalg.norm(r1 - r2)
        if rij <= a:
            return 0.
        else:
            return 1 - (a / rij)

    scale = 2.
    r = np.random.randn(N, dim) * scale
    #r = rng.random(size=(N, dim))

    rerun = True
    while rerun:
        rerun = False
        for i in range(N):
            for j in range(i + 1, N):
                corr = corr_factor(r[i, :], r[j, :])
                if corr == 0.:
                    print("corr=0 encountered")
                    rerun = True
                    r[i, :] = np.random.randn() * scale
                    r[j, :] = np.random.randn() * scale
        scale *= 1.5

    return r, scale


N = 2      # Number of particles
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

# Config
nsamples = 10000
initial_alpha = 0.2

# NON-INTERACTING

#wf = vmc.ASHONIB(N, dim, omega)
#wf = vmc.SHONIB(omega)
#wf = vmc.EHONIB()
#initial_positions = non_interact_initial_positions(wf, initial_alpha, N, dim)

# INTERACTING
#wf = vmc.SHOIB(omega)
wf = vmc.ASHOIB(N, dim, omega)
#wf = vmc.EHOIB()
initial_positions, scale = interact_initial_positions(wf,
                                                      initial_alpha,
                                                      N,
                                                      dim)


# Instantiate sampler
#sampler = vmc.Metropolis(wf)
sampler = vmc.MetropolisHastings(wf)

#print(wf.logprob(initial_positions, initial_alpha))
#print(wf.drift_force(initial_positions, initial_alpha))
#print(wf.local_energy(initial_positions, initial_alpha))


start = time.time()
results = sampler.sample(nsamples,
                         initial_positions,
                         initial_alpha,
                         #scale=1.0,  # METROPOLIS
                         dt=1e-10,     # METROPOLIS-HASTINGS
                         nchains=4,
                         warm=True,
                         warmup_iter=1000,
                         tune=False,
                         tune_iter=5000,
                         tune_interval=250,
                         tol_tune=1e-5,
                         optimize=False,
                         max_iter=70000,
                         batch_size=500,
                         gradient_method='adam',
                         eta=0.01,
                         tol_optim=1e-5,
                         )

end = time.time()
print("Sampler elapsed time:", end - start)

exact_E = exact_energy(N, dim, omega)
print(f"Exact energy spherical HO NIB: {exact_E}")
print(results)
# print(sampler.results_all)
