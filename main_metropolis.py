#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src import vmc


def exact_energy(N, dim, omega):
    return (omega * dim * N) / 2


def safe_initial_positions(wavefunction, alpha, N, dim, seed=None):
    rng = np.random.default_rng(seed=seed)
    positions = rng.random(size=(N, dim))
    # safe initialization
    wf2 = wavefunction.pdf(positions, alpha)
    while np.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wavefunction.pdf(positions, alpha)
    return positions


N = 500        # Number of particles
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

# Instantiate wave function
# Analytical
wf = vmc.SHOIB(omega)
# Numerical
#wf = vmc.LogNIB(omega)

# Instantiate sampler
sampler = vmc.Metropolis(wf)

# Config
nsamples = 10000
initial_alpha = 0.2

# Set intial positions
initial_positions = safe_initial_positions(wf, initial_alpha, N, dim)

start = time.time()
results = sampler.sample(nsamples,
                         initial_positions,
                         initial_alpha,
                         scale=1.0,
                         nchains=4,
                         warm=True,
                         warmup_iter=500,
                         tune=True,
                         tune_iter=5000,
                         tune_interval=250,
                         tol_tune=1e-5,
                         optimize=True,
                         max_iter=70000,
                         batch_size=500,
                         gradient_method='adam',
                         eta=0.01,
                         tol_optim=1e-5,
                         )

end = time.time()
print("Sampler elapsed time:", end - start)

exact_E = exact_energy(N, dim, omega)
print(f"Exact energy: {exact_E}")
print(results)

wf = vmc.ASHOIB(omega)
# Numerical
#wf = vmc.LogNIB(omega)

# Instantiate sampler
sampler = vmc.Metropolis(wf)

# Config
nsamples = 10000
initial_alpha = 0.2

# Set intial positions

start = time.time()
results = sampler.sample(nsamples,
                         initial_positions,
                         initial_alpha,
                         scale=1.0,
                         nchains=4,
                         warm=True,
                         warmup_iter=500,
                         tune=True,
                         tune_iter=5000,
                         tune_interval=250,
                         tol_tune=1e-5,
                         optimize=True,
                         max_iter=70000,
                         batch_size=500,
                         gradient_method='adam',
                         eta=0.01,
                         tol_optim=1e-5,
                         )

end = time.time()
print("Sampler elapsed time:", end - start)

exact_E = exact_energy(N, dim, omega)
print(f"Exact energy: {exact_E}")
print(results)
