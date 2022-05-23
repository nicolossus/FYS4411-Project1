#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Import code from src
sys.path.insert(0, '../src/')
import vmc  # noqa

"""
Grid search for optimal alpha with ASHONIB and RWM
"""

output_filename = "../data/grid_search_ashonib_rwm.csv"

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
Ns = [1, 10, 100, 500]  # Number of particles
dim = 3                 # Dimensionality
nsamples = 20000
nchains = 10

# Set grid of alphas
alphas = np.linspace(0.1, 1.0, nchains)

for N in tqdm(Ns):

    # Instantiate wave function
    #wf = vmc.ASHONIB(N, dim)
    wf = vmc.SHONIB()

    # Set intial positions
    initial_positions = [non_interact_initial_positions(wf, alpha, N, dim)
                         for alpha in alphas]

    #initial_positions = np.random.rand(N, dim)

    # Instantiate sampler
    sampler = vmc.RWM(wf)
    _ = sampler.sample(nsamples,
                       initial_positions,
                       alphas,
                       scale=1.0,
                       nchains=nchains,
                       seed=42,
                       warm=True,
                       warmup_iter=20000,
                       rewarm=True,
                       rewarm_iter=5000,
                       tune=True,
                       tune_iter=10000,
                       tune_interval=500,
                       tol_tune=1e-5,
                       optimize=False,
                       log=False
                       )
    df = sampler.results_all

    # Save results
    df.to_csv(output_filename, mode='a', index=False,
              header=not os.path.exists(output_filename))
