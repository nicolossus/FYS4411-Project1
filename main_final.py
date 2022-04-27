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
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

# Config

nsamples = 10000
initial_alpha = 0.5


# NON-INTERACTING

wf = vmc.ASHONIB(N, dim, omega)
#wf = vmc.SHONIB(omega)
# wf = vmc.EHONIB()
#initial_positions = non_interact_initial_positions(wf, initial_alpha, N, dim)

# INTERACTING
#wf = vmc.SHOIB(omega)

#wf = vmc.ASHOIB(N, dim, omega)
#wf = vmc.EHOIB()




#wf = vmc.EHOIB()

initial_positions, scale = interact_initial_positions(wf,
                                                      initial_alpha,
                                                      N,
                                                      dim)

# Instantiate sampler
sampler = vmc.Metropolis(wf)
#sampler = vmc.MetropolisHastings(wf)

# print(wf.logprob(initial_positions, initial_alpha))
# print(wf.drift_force(initial_positions, initial_alpha))
# print(wf.local_energy(initial_positions, initial_alpha))
num_chains = 8

start = time.time()
results = sampler.sample(nsamples,
                         initial_positions,
                         initial_alpha,
                         scale=1.0,  # METROPOLIS
                         #dt=1e-10,     # METROPOLIS-HASTINGS
                         nchains=num_chains,
                         warm=True,
                         warmup_iter=5000,
                         tune=True,
                         tune_iter=10000,
                         tune_interval=500,
                         tol_tune=1e-5,
                         optimize=False,
                         max_iter=70000,
                         batch_size=500,
                         gradient_method='adam',
                         eta=0.01,
                         tol_optim=1e-5,
                         )

end = time.time()
print("Sampler elapsed time:", end - start)

exact_E = exact_energy(N, dim, omega)
print(f"Exact energy spherical HO NIB: {exact_E}")
print(results)
cwd = os.getcwd()
filename = "/pdfs.csv"
data_path = "/data/interactions"
os.makedirs(cwd+data_path, exist_ok=True)
#print(sampler.energy_samples)
#print(sampler.distance_samples)
#print(sampler.distance_samples[0])
distance_samples = sampler.distance_samples.flatten()
#for i in range(num_chains):
#    distance_samples = np.concatenate(distance_samples, sampler.distance_samples[i])
#print(distance_samples.shape)
pdf_samples = sampler.pdf_samples.flatten()
NCycles, NParticles = distance_samples.shape
OBD_dict = {}

for particle in range(NParticles):
    OBD_dict[f"r_particle_{particle}"] = []
    OBD_dict[f"p_particle_{particle}"] = []
    for cycle in range(NCycles):
        OBD_dict[f"r_particle_{particle}"].append(distance_samples[cycle, particle])
        OBD_dict[f"p_particle_{particle}"].append(pdf_samples[cycle, particle])
OBD_data = pd.DataFrame(data=OBD_dict)
print(OBD_data)

particle_1_r = OBD_data["r_particle_1"]
particle_1_p = OBD_data["p_particle_1"]

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=particle_1_r, bins=50, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
