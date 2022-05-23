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


N = 10   # Number of particles
dim = 2     # Dimensionality
omega = 1.   # Oscillator frequency

# Config

nsamples = 1000
initial_alpha = 0.5

x = np.linspace(0, 5, 11)
y = np.linspace(0, 5, 11)
mx, my = np.meshgrid(x, y, indexing="ij")
# NON-INTERACTING

wf = vmc.ASHONIB(N, dim, omega)
#wf = vmc.SHONIB(omega)
# wf = vmc.EHONIB()
#initial_positions = non_interact_initial_positions(wf, initial_alpha, N, dim)

# INTERACTING
#wf = vmc.SHOIB(N, dim, omega)

#wf = vmc.ASHOIB(N, dim, omega)
#wf = vmc.EHOIB()




#wf = vmc.EHOIB()

initial_positions, scale = interact_initial_positions(wf,
                                                      initial_alpha,
                                                      N,
                                                      dim)

# Instantiate sampler
sampler = vmc.OBDMetropolis(wf)
#sampler = vmc.MetropolisHastings(wf)

num_chains = 1
results = {"x": [], "y": [], "pdf": []}
start = time.time()
for i in range(11):
    for j in range(11):
        x_, y_ = mx[i][j], my[i][j]
        results["x"].append(x_)
        results["y"].append(y_)
        particle = np.array([x_, y_])
        if (i==0 and j==0):
            state, pdf = sampler.sample(particle,
                                            nsamples,
                                            initial_positions,
                                            initial_alpha,
                                            scale=1.0,  # METROPOLIS
                                            nchains=num_chains,
                                            warm=True,
                                            warmup_iter=5000,
                                            tune=True,
                                            tune_iter=10000,
                                            tune_interval=500,
                                            )
            results["pdf"].append(pdf)

        else:
            positions, pdf = sampler.sample(particle,
                                            nsamples,
                                            state.positions,
                                            initial_alpha,
                                            scale=1.0,
                                            nchains=num_chains,
                                            warm=True,
                                            warmup_iter=100,
                                            tune=False,
                                            )
            results["pdf"].append(pdf)

end = time.time()
print("Sampler elapsed time:", end - start)
df = pd.DataFrame(data=results)
print(df)

x = df["x"]
y = df["y"]
pdf = df["pdf"]

r = np.sqrt(x**2 + y**2)
plt.plot(r, pdf)
plt.show()