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
Grid search for optimal alpha with SHONIB and RWM
"""

output_filename = "../data/grid_search_shonib_rwm.csv"

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
nchains = 16

# Set grid of alphas
alphas = np.linspace(0.1, 1.0, 10)

# initializing progress bar objects
outer_loop = tqdm(range(len(Ns)))
inner_loop = tqdm(range(len(alphas)))

for i in range(len(outer_loop)):

    inner_loop.refresh()  # force print final state
    inner_loop.reset()  # reuse bar

    # Instantiate wave function
    wf = vmc.SHONIB()

    # Instantiate sampler
    sampler = vmc.RWM(wf)
    #sampler = vmc.LMH(wf)

    for j in range(len(inner_loop)):

        # Set intial positions
        initial_positions = [non_interact_initial_positions(wf, alphas[j], Ns[i], dim)
                             for chain in range(nchains)]

        # Instantiate sampler
        sampler = vmc.RWM(wf)
        #sampler = vmc.LMH(wf)
        _ = sampler.sample(nsamples,
                           initial_positions,
                           alphas[j],
                           scale=1.0,  # RWM
                           # dt=0.5,     # LMH
                           nchains=nchains,
                           seed=None,
                           warm=True,
                           warmup_iter=20000,
                           rewarm=True,
                           rewarm_iter=20000,
                           tune=True,
                           tune_iter=50000,
                           tune_interval=2000,
                           tol_tune=1e-5,
                           optimize=False,
                           log=False
                           )
        df = sampler.results_all

        # Save results
        df.to_csv(output_filename, mode='a', index=False,
                  header=not os.path.exists(output_filename))

        inner_loop.update()  # update inner tqdm

    outer_loop.update()  # update outer tqdm
