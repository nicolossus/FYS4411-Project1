#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os
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


N = 100 # Number of particles
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

# Instantiate wave function
# Analytical
wf = vmc.AIB(N, dim, omega)
#wf = vmc.LogIB(omega)
#wf = vmc.NIBWF(N, dim, omega)

# Numerical
#wf = vmc.LogNIB(omega)

# Instantiate sampler
sampler = vmc.Metropolis(wf)

# Config
nsamples = 10000
initial_alpha = 0.5

# Set intial positions
initial_positions = safe_initial_positions(wf, initial_alpha, N, dim)
wf.test_terms_in_lap(initial_positions, wf.dudr_faster(initial_positions), initial_alpha)
print(f"Local energy={wf.local_energy(initial_positions, initial_alpha)}")
'''
print("wf jax : ", wf.wf(initial_positions, initial_alpha))
print("wf drift : ", wf.drift_force(initial_positions, initial_alpha))
print("wf scalar : ", wf.wf_scalar(initial_positions, initial_alpha))
print("wf grad alpha: ", wf.grad_alpha(initial_positions, initial_alpha))
print("wf LE : ", wf.local_energy(initial_positions, initial_alpha))
print("wf grad alpha: ", wf.grad_alpha(initial_positions, initial_alpha))
'''
"""
start = time.time()
distance_matrix = wf.distance_matrix(initial_positions)
end = time.time()
print("Time distance matrix: ", end-start)
start = time.time()
dudr = wf.dudr(initial_positions)
end = time.time()
print("Time dudr: ", end-start)
start = time.time()
dudr_faster = wf.dudr_faster(initial_positions)
end = time.time()
print("Time dudr faster: ", end-start)
print("Dudr: ", dudr_faster)
print("Shape dudr: ", dudr_faster.shape)
"""
"""
start = time.time()
unit_matrix = wf.unit_matrix(initial_positions)
end = time.time()
print("Unit matrix slow: ", unit_matrix)
print("Time unit matrix: ", end-start)
start = time.time()
#unit_matrix_faster = wf.unit_matrix_faster(initial_positions)
end = time.time()
print("unit_matrix_faster: ", unit_matrix_faster)
print("Faster? unit matrix time: ", end-start)
start = time.time()
fourth_term = wf.fourth_term(initial_positions)
end = time.time()
print("Fourth term normal: ", end-start)
start = time.time()
fourth_term_fast = wf.fourth_term_faster(initial_positions)
end = time.time()
print("Fourth term faster: ", end-start)
"""
"""
#dudr = wf.dudr(initial_positions)
#dudr_faster = wf.dudr_faster(initial_positions)
#print("fourth term: ", fourth_term)
#print("fourth term faster: ", fourth_term_fast)
#print("Dudr: ", dudr)
#print("Dudr faster: ", dudr_faster)
start = time.time()
unit_matrix = wf.unit_matrix(initial_positions)
end = time.time()
print("Unit matrix slow: ", unit_matrix)
print("Speed: ", end-start)
start = time.time()
unit_matrix_faster = wf.unit_matrix_faster(initial_positions)
end = time.time()
print("Unit matrix fast: ", unit_matrix_faster)
print("Speed: ", end-start)
start = time.time()
dudr = wf.dudr(initial_positions)
end = time.time()
print("Dudr: ", dudr)
print("Sum dudr: ", np.sum(dudr, axis=1))
print("TIme: ", end-start)
start = time.time()
dudr_faster = wf.dudr_faster(initial_positions)
end = time.time()
print("Dudr fast: ", dudr_faster)
print("time: ", end-start)
#start = time.time()
#dudr_jastrow, grad_jastrow = wf._gradient_jastrow(initial_positions, 0.5)
#end = time.time()
#print("Gradient jastrow: ", grad_jastrow)
#print("dudr_jastrow: ", dudr_jastrow)
#print("Speed: ", end-start)
#print(initial_positions)
#wf.test_terms_in_lap(initial_positions, dudr, 0.5)
#wf.test_terms_in_lap(initial_positions, dudr_faster, 0.5)
#print(unit_matrix_faster)
#LE = wf.local_energy(initial_positions, 0.5)
"""
"""
print("AIBWF: ", wf(initial_positions, initial_alpha))
print("NIBWF: ", wf2(initial_positions, initial_alpha))
print("Positions: ", initial_positions)
print("Distances: ", wf.distance_matrix(initial_positions))
print("Shape distances: ", wf.distance_matrix(initial_positions).shape)
print("Unit matrix: ", np.sum(wf.unit_matrix(initial_positions), axis=0))
print("Dudr: ", wf.dudr(initial_positions))
print("AIBWF drift: ", wf.drift_force(initial_positions, initial_alpha))
print("NIBWF drift: ", wf2.drift_force(initial_positions, initial_alpha))
print("AIBWF local energy: ", wf.local_energy(initial_positions, initial_alpha))
print("NIBWF local energy: ", wf2.local_energy(initial_positions, initial_alpha))
"""
print(wf.logprob(initial_positions, 0.5))
start = time.time()
results = sampler.sample(nsamples,
                         initial_positions,
                         initial_alpha,
                         scale=1.0,
                         nchains=1,
                         warm=True,
                         warmup_iter=1000,
                         tune=True,
                         tune_iter=5000,
                         tune_interval=500,
                         tol_tune=1e-7,
                         optimize=False,
                         max_iter=20000,
                         batch_size=500,
                         gradient_method='adam',
                         eta=0.1,
                         tol_optim=1e-9,
                         early_stop=False,
                         )

end = time.time()
print("Sampler elapsed time:", end - start)


"""
# Instantiate sampler
sampler = vmc.MetropolisHastings(wf)

# Config
nsamples = 10000
initial_alpha = 0.2

# Set intial positions
initial_positions = safe_initial_positions(wf, initial_alpha, N, dim)

start = time.time()
results = sampler.sample(nsamples,
                         initial_positions,
                         initial_alpha,
                         dt=1.0,
                         nchains=1,
                         warm=True,
                         warmup_iter=100,
                         tune=False,
                         tune_iter=5000,
                         tune_interval=250,
                         tol_tune=1e-6,
                         optimize=False,
                         max_iter=30000,
                         batch_size=500,
                         gradient_method='adam',
                         eta=0.1,
                         tol_optim=1e-7,
                         )

end = time.time()
print("Sampler elapsed time:", end - start)
"""
"""
wf.test_terms_in_lap(initial_positions, wf.dudr(initial_positions), 0.5)
print("AIBWF log(wf^2): ", wf.logprob(initial_positions, 0.5))
#print("Positions: ", initial_positions)
#print("Distances: ", wf.distance_matrix(initial_positions))
#print("Shape distances: ", wf.distance_matrix(initial_positions).shape)
#print("Unit matrix: ", wf.unit_matrix(initial_positions))
#print("Dudr: ", wf.dudr(initial_positions))
#print("AIBWF drift: ", wf.drift_force(initial_positions, 0.5))
#print("NIBWF drift: ", wf2.drift_force(initial_positions, 0.5))
print("AIBWF local energy: ", wf.local_energy(initial_positions, 0.5))
print("NIBWF local energy: ", wf2.local_energy(initial_positions, 0.5))
#print("AIBWF u: ", np.exp(np.sum(wf.u(initial_positions))))
#print("AIBWF f: ", wf.f(initial_positions))
"""

exact_E = exact_energy(N, dim, omega)
print(f"Exact energy: {exact_E}")
print(results)
#cwd = os.getcwd()
#filename = "/FiftyNInteractions.csv"
#data_path = "/data/interactions"
#os.makedirs(cwd+data_path, exist_ok=True)
#sampler._results_full.to_csv(cwd+data_path+filename)
