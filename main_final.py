#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os
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
    # r = rng.random(size=(N, dim))

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


N = 500 # Number of particles
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

# Config

nsamples = 10000
alpha_vals = [0.4, 0.5, 0.6]

wf = vmc.ASHONIB(N, dim, omega)
#wf = vmc.EHONIB()

sampler = vmc.RWM(wf)

num_chains = 4
for i, alpha in enumerate(alpha_vals):
    initial_positions, scale = interact_initial_positions(wf,
                                                          alpha,
                                                          N,
                                                          dim)
    start = time.time()
    results = sampler.sample(nsamples,
                         initial_positions,
                         alpha,
                         scale=1.0,  # METROPOLIS
                         #dt=1e-10,     # METROPOLIS-HASTINGS
                         nchains=num_chains,
                         warm=True,
                         warmup_iter=20000,
                         tune=True,
                         tune_iter=30000,
                         tune_interval=1000,
                         tol_tune=1e-5,
                         optimize=False,
                         max_iter=200000,
                         batch_size=2000,
                         gradient_method='adam',
                         eta=0.01,
                         tol_optim=1e-8,
                         )

    end = time.time()
    print("Sampler elapsed time:", end - start)

    print(results)


wf = vmc.SHONIB(omega)


sampler = vmc.RWM(wf)


num_chains = 4
for i, alpha in enumerate(alpha_vals):
    initial_positions, scale = interact_initial_positions(wf,
                                                          alpha,
                                                          N,
                                                          dim)
    start = time.time()
    results = sampler.sample(nsamples,
                         initial_positions,
                         alpha,
                         scale=1.0,  # METROPOLIS
                         #dt=1e-10,     # METROPOLIS-HASTINGS
                         nchains=num_chains,
                         warm=True,
                         warmup_iter=20000,
                         tune=True,
                         tune_iter=30000,
                         tune_interval=1000,
                         tol_tune=1e-5,
                         optimize=True,
                         max_iter=200000,
                         batch_size=2000,
                         gradient_method='adam',
                         eta=0.01,
                         tol_optim=1e-8,
                         )

    end = time.time()
    print("Sampler elapsed time:", end - start)

    print(results)
