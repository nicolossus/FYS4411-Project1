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
        positions *= 2.0
        wf2 = wavefunction.pdf(positions, alpha)
    return positions


N = 5 # Number of particles
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

# Instantiate wave function
# Analytical
#wf = vmc.AIB(N, dim, omega)
wf = vmc.LogIB(omega)
wf2 = vmc.NIBWF(N, dim, omega)

# Numerical
#wf = vmc.LogNIB(omega)

# Instantiate sampler
sampler = vmc.Metropolis(wf)

# Config
nsamples = 100000
initial_alpha = 0.5

# Set intial positions
initial_positions = safe_initial_positions(wf, initial_alpha, N, dim)

print("wf jax : ", wf.wf(initial_positions, initial_alpha))
print("wf drift : ", wf.drift_force(initial_positions, initial_alpha))
print("wf jax : ", wf.wf_scalar(initial_positions, initial_alpha))
print("wf LE : ", wf.local_energy(initial_positions, initial_alpha))
"""
print("AIBWF: ", wf(initial_positions, 0.5))
print("NIBWF: ", wf2(initial_positions, 0.5))
print("Positions: ", initial_positions)
print("Distances: ", wf.distance_matrix(initial_positions))
print("Shape distances: ", wf.distance_matrix(initial_positions).shape)
print("Unit matrix: ", np.sum(wf.unit_matrix(initial_positions), axis=0))
print("Dudr: ", wf.dudr(initial_positions))
print("AIBWF drift: ", wf.drift_force(initial_positions, 0.5))
print("NIBWF drift: ", wf2.drift_force(initial_positions, 0.5))
print("AIBWF local energy: ", wf.local_energy(initial_positions, 0.5))
print("NIBWF local energy: ", wf2.local_energy(initial_positions, 0.5))
"""
"""
start = time.time()
results = sampler.sample(nsamples,
                         initial_positions,
                         initial_alpha,
                         scale=1.0,
                         nchains=1,
                         warm=True,
                         warmup_iter=500,
                         tune=True,
                         tune_iter=10000,
                         tune_interval=500,
                         tol_tune=1e-8,
                         optimize=True,
                         max_iter=10000,
                         batch_size=500,
                         gradient_method='adam',
                         eta=0.01,
                         tol_optim=1e-7,
                         )

end = time.time()
print("Sampler elapsed time:", end - start)
"""
"""

print("Positions: ", initial_positions)
print("Distances: ", wf.distance_matrix(initial_positions))
print("Shape distances: ", wf.distance_matrix(initial_positions).shape)
print("Unit matrix: ", wf.unit_matrix(initial_positions))
print("Dudr: ", wf.dudr(initial_positions))
print("AIBWF drift: ", wf.drift_force(initial_positions, 0.5))
print("NIBWF drift: ", wf2.drift_force(initial_positions, 0.5))
print("AIBWF local energy: ", wf.local_energy(initial_positions, 0.5))
print("NIBWF local energy: ", wf2.local_energy(initial_positions, 0.5))
"""
exact_E = exact_energy(N, dim, omega)
print(f"Exact energy: {exact_E}")
print(results)
