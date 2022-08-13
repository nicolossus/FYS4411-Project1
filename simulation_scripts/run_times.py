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

N = [1, 10, 50, 100, 500]      # Number of particles
dim = 3      # Dimensionality
#output_filename = "../data/ashonib100.csv"

#if os.path.exists(output_filename):
#    os.remove(output_filename)
# Config
data_A_RWM = {"N": [], "Energy": [], "Time":[]}
data_J_RWM = {"N": [], "Energy": [], "Time":[]}
data_A_LMH = {"N": [], "Energy": [], "Time":[]}
data_J_LMH = {"N": [], "Energy": [], "Time":[]}
nsamples = int(2**15)
nchains = 1
initial_alpha = 0.4
times_RWM_analytical = np.zeros((5, 10))
times_RWM_jax = np.zeros((5,10))
times_LMH_analytical = np.zeros((5, 10))
times_LMH_jax = np.zeros((5, 10))
for i, n in enumerate(N):
    wf_analytical = vmc.ASHONIB(n, dim, 1.0)
    initial_positions = np.random.rand(n, dim)
    sampler_RWM_analytical = vmc.samplers.RWM(wf_analytical)
    sampler_LMH_analytical = vmc.samplers.LMH(wf_analytical)
    wf_jax = vmc.SHONIB(1.0)
    sampler_RWM_jax = vmc.samplers.RWM(wf_jax)
    sampler_LMH_jax = vmc.samplers.LMH(wf_jax)
    for j in range(10):
        data_A_RWM["N"].append(n)
        data_J_RWM["N"].append(n)
        data_A_LMH["N"].append(n)
        data_J_LMH["N"].append(n)

        start = time.time()
        results = sampler_RWM_analytical.sample(nsamples,
                                                initial_positions,
                                                initial_alpha,
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
                                                early_stop=False,
                                                warm=False,
                                                warmup_iter=5000,
                                                log=True,
                                                logger_level="INFO",
                                                )
        end = time.time()
        print(results)
        data_A_RWM["Energy"].append(results["energy"])
        data_A_RWM["Time"].append(end-start)
        start = time.time()
        results = sampler_LMH_analytical.sample(nsamples,
                                                initial_positions,
                                                initial_alpha,
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
                                                early_stop=False,
                                                warm=False,
                                                warmup_iter=5000,
                                                log=True,
                                                logger_level="INFO",
                                                )
        end = time.time()
        print(results)
        data_A_LMH["Energy"].append(results["energy"])
        data_A_LMH["Time"].append(end-start)
        start = time.time()
        results = sampler_RWM_jax.sample(nsamples,
                                         initial_positions,
                                         initial_alpha,
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
                                         early_stop=False,
                                         warm=False,
                                         warmup_iter=5000,
                                         log=True,
                                         logger_level="INFO",
                                         )
        end = time.time()
        print(results)
        data_J_RWM["Energy"].append(results["energy"])
        data_J_RWM["Time"].append(end-start)
        start = time.time()
        results = sampler_LMH_jax.sample(nsamples,
                                         initial_positions,
                                         initial_alpha,
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
                                         early_stop=False,
                                         warm=False,
                                         warmup_iter=5000,
                                         log=True,
                                         logger_level="INFO",
                                         )
        end = time.time()
        print(results)
        data_J_LMH["Energy"].append(results["energy"])
        data_J_LMH["Time"].append(end-start)

data_A_RWM = pd.DataFrame(data=data_A_RWM)
data_J_RWM = pd.DataFrame(data=data_J_RWM)
data_A_LMH = pd.DataFrame(data=data_A_LMH)
data_J_LMH = pd.DataFrame(data=data_J_LMH)

data_A_RWM.to_csv("../data/data_A_RWM.csv", mode='a', index=False,
          header=not os.path.exists("../data/data_A_RWM.csv"))
data_J_RWM.to_csv("../data/data_A_LMH.csv", mode='a', index=False,
          header=not os.path.exists("../data/data_A_LMH.csv"))
data_A_LMH.to_csv("../data/data_J_RWM.csv", mode='a', index=False,
          header=not os.path.exists("../data/data_J_RWM.csv"))
data_J_LMH.to_csv("../data/data_J_LMH.csv", mode='a', index=False,
          header=not os.path.exists("../data/data_J_LMH.csv"))
