#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import code from src
sys.path.insert(0, '../src/')
import vmc  # noqa

N = 50      # Number of particles
dim = 3      # Dimensionality
output_filename = "../data/ashonib100.csv"

if os.path.exists(output_filename):
    os.remove(output_filename)
# Config
nsamples = int(2**15)
nchains = 4
initial_alpha = 0.4

# Instantiate wave function
wf = vmc.ASHONIB(N, dim, 1.0)

# Set intial positions
initial_positions = np.random.rand(N, dim)

# Instantiate sampler
sampler = vmc.experimental.RWMEvolution(wf)
dfs = []
times = []
#sampler = vmc.samplers.LMH(wf)
for i in range(10):

    start = time.time()
    results = sampler.sample(nsamples,
                            initial_positions,
                            initial_alpha,
                            scale=1.0,
                            nchains=nchains,
                            seed=None,
                            tune=True,
                            tune_iter=20000,
                            tune_interval=1000,
                            retune=False,
                            retune_iter=5000,
                            retune_interval=250,
                            tol_tune=1e-5,
                            optimize=True,
                            max_iter=50000,
                            batch_size=1000,
                            gradient_method='adam',
                            eta=0.01,
                            tol_optim=1e-5,
                            early_stop=False,
                            warm=True,
                            warmup_iter=5000,
                            log=True,
                            logger_level="INFO",
                            )

    end = time.time()
    print("Sampler elapsed time:", end - start)
    times.append(end-start)
    print(results)
    df_iter = sampler.results_all
    dfs.append(df_iter)
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_filename, mode='a', index=False,
          header=not os.path.exists(output_filename))

print(times)
print(np.mean(times))
print(np.std(times))

"""
energies = sampler.energy_samples
df_cycles = sampler.results_all[[
    "total_cycles", "warmup_cycles", "tuning_cycles", "optimize_cycles", "nsamples"]]
print(df_cycles)
exact_energy = vmc.utils.exact_energy(N, dim)

fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
if nchains == 1:
    ax.plot(energies, label=f'Chain 1')
else:
    for i in range(nchains):
        ax.plot(energies[i], label=f'Chain {i+1}')
ax.axhline(exact_energy, ls='--', color='k')
ax.set(xlabel="Iteration",
       ylabel="Energy")
ax.legend()
plt.show()
"""
