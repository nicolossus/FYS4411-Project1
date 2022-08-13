#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import code from src
sys.path.insert(0, '../src/')
import vmc # noqa

"""
Grid search for optimal alpha with ASHONIB and RWM
"""

output_filename0 = "../data/aehoib_LMH.csv"
output_filename1 = "../data/aehonib_LMH.csv"
output_filename2 = "../data/aehonib_RMW.csv"
output_filename3 = "../data/aehoib_RMW.csv"

# Remove file if it exists
if os.path.exists(output_filename0):
    os.remove(output_filename0)
if os.path.exists(output_filename1):
    os.remove(output_filename1)
if os.path.exists(output_filename2):
    os.remove(output_filename2)
if os.path.exists(output_filename3):
    os.remove(output_filename3)


def non_interact_initial_positions(wf, alpha, N, dim):
    rng = np.random.default_rng()
    positions = rng.random(size=(N, dim))
    # safe initialization
    wf2 = wf.pdf(positions, alpha)
    while np.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wf.pdf(positions, alpha)
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



# Config
Ns = [10, 50, 100]  # Number of particles
dim = 3                 # Dimensionality
nsamples = 2**15
nchains = 4
omega=1.0

# Set grid of alphas
alpha = 0.5
dfs = []
for N in tqdm(Ns):

    # Instantiate wave function
    #wf = vmc.ASHONIB(N, dim)
    wf = vmc.AEHOIB(N, dim)
    # Set intial positions
    #initial_positions = non_interact_initial_positions(wf, alpha, N, dim)

    #initial_positions = np.random.rand(N, dim)
    for i in range(4):
        # Set intial positions
        initial_positions, scale = interact_initial_positions(wf, alpha, N, dim)
        # Instantiate sampler
        sampler = vmc.LMH(wf)
        _ = sampler.sample(nsamples,
                       initial_positions,
                       alpha,
                       dt=0.1,
                       nchains=nchains,
                       seed=None,
                       tune=True,
                       tune_iter=20000,
                       tune_interval=1000,
                       tol_tune=1e-5,
                       optimize=True,
                       max_iter=50000,
                       batch_size=1000,
                       gradient_method='adam',
                       eta=0.01,
                       tol_optim=1e-5,
                       early_stop=True,
                       warm=False,
                       warmup_iter=5000,
                       log=True,
                       logger_level="INFO",
                       )
        df = sampler.results_all
        dfs.append(df)
# Save results
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_filename0, mode='a', index=False,
        header=not os.path.exists(output_filename0))

for N in tqdm(Ns):

    # Instantiate wave function
    #wf = vmc.ASHONIB(N, dim)
    wf = vmc.AEHONIB(N, dim)
    # Set intial positions
    #initial_positions = non_interact_initial_positions(wf, alpha, N, dim)

    #initial_positions = np.random.rand(N, dim)
    for i in range(4):
        # Set intial positions
        initial_positions = non_interact_initial_positions(wf, alpha, N, dim)
        # Instantiate sampler
        sampler = vmc.LMH(wf)
        _ = sampler.sample(nsamples,
                       initial_positions,
                       alpha,
                       dt=0.1,
                       nchains=nchains,
                       seed=None,
                       tune=True,
                       tune_iter=20000,
                       tune_interval=1000,
                       tol_tune=1e-5,
                       optimize=True,
                       max_iter=50000,
                       batch_size=1000,
                       gradient_method='adam',
                       eta=0.01,
                       tol_optim=1e-5,
                       early_stop=True,
                       warm=False,
                       warmup_iter=5000,
                       log=True,
                       logger_level="INFO",
                       )
        df = sampler.results_all
        dfs.append(df)
# Save results
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_filename1, mode='a', index=False,
        header=not os.path.exists(output_filename1))


for N in tqdm(Ns):

    # Instantiate wave function
    #wf = vmc.ASHONIB(N, dim)
    wf = vmc.AEHONIB(N, dim)
    # Set intial positions
    #initial_positions = non_interact_initial_positions(wf, alpha, N, dim)

    #initial_positions = np.random.rand(N, dim)
    for i in range(4):
        # Set intial positions
        initial_positions = non_interact_initial_positions(wf, alpha, N, dim)
        # Instantiate sampler
        sampler = vmc.RWM(wf)
        _ = sampler.sample(nsamples,
                       initial_positions,
                       alpha,
                       scale=1.0,
                       nchains=nchains,
                       seed=None,
                       tune=True,
                       tune_iter=20000,
                       tune_interval=1000,
                       tol_tune=1e-5,
                       optimize=True,
                       max_iter=50000,
                       batch_size=1000,
                       gradient_method='adam',
                       eta=0.01,
                       tol_optim=1e-5,
                       early_stop=True,
                       warm=False,
                       warmup_iter=5000,
                       log=True,
                       logger_level="INFO",
                       )
        df = sampler.results_all
        dfs.append(df)
# Save results
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_filename2, mode='a', index=False,
        header=not os.path.exists(output_filename2))

for N in tqdm(Ns):

    # Instantiate wave function
    #wf = vmc.ASHONIB(N, dim)
    wf = vmc.AEHOIB(N, dim)
    # Set intial positions
    #initial_positions = non_interact_initial_positions(wf, alpha, N, dim)

    #initial_positions = np.random.rand(N, dim)
    for i in range(4):
        # Set intial positions
        initial_positions, scale = interact_initial_positions(wf, alpha, N, dim)
        # Instantiate sampler
        sampler = vmc.RWM(wf)
        _ = sampler.sample(nsamples,
                       initial_positions,
                       alpha,
                       scale=1.0,
                       nchains=nchains,
                       seed=None,
                       tune=True,
                       tune_iter=20000,
                       tune_interval=1000,
                       tol_tune=1e-5,
                       optimize=True,
                       max_iter=50000,
                       batch_size=1000,
                       gradient_method='adam',
                       eta=0.01,
                       tol_optim=1e-5,
                       early_stop=True,
                       warm=False,
                       warmup_iter=5000,
                       log=True,
                       logger_level="INFO",
                       )
        df = sampler.results_all
        dfs.append(df)
# Save results
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_filename3, mode='a', index=False,
        header=not os.path.exists(output_filename3))
