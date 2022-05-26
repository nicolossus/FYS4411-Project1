#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

import jax
import numpy as np
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

# Import code from src
sys.path.insert(0, '../src/')
import vmc  # noqa

"""
Sampling with RWM and ASHONIB
"""

output_filename = "../data/sample_ashonib_rwm.csv"

# Remove file if it exists
if os.path.exists(output_filename):
    os.remove(output_filename)


def non_interact_initial_positions(wf, alpha, N, dim):
    rng = np.random.default_rng()
    positions = rng.random(size=(N, dim))
    # safe initialization
    wf2 = wf.pdf(positions, alpha)
    while np.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wf.pdf(positions, alpha)
    return positions


# Config
N = 100  # Number of particles
dim = 3  # Dimensionality
nsamples_all = np.logspace(12, 17, 6, base=2, dtype=int)
nchains = 16

# Set initial alpha
alpha0 = 0.4

# Instantiate wave function
wf = vmc.ASHONIB(N, dim)

# Instantiate sampler
sampler = vmc.RWM(wf)

# Set intial positions
initial_positions = [non_interact_initial_positions(wf, alpha0, N, dim)
                     for chain in range(nchains)]

for nsamples in tqdm(nsamples_all):

    _ = sampler.sample(int(nsamples),
                       initial_positions,
                       alpha0,
                       scale=1.0,
                       nchains=nchains,
                       seed=None,
                       tune=True,
                       tune_iter=10000,
                       tune_interval=500,
                       tol_tune=1e-5,
                       optimize=True,
                       max_iter=50000,
                       batch_size=500,
                       gradient_method='adam',
                       eta=0.01,
                       tol_optim=1e-5,
                       early_stop=False,
                       log=False,
                       logger_level="INFO",
                       )
    df = sampler.results_all

    # Save results
    df.to_csv(output_filename, mode='a', index=False,
              header=not os.path.exists(output_filename))
