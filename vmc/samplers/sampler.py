#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import default_rng
from state import State
from walkers import rw_metropolis_step

#warnings.filterwarnings("ignore", message="divide by zero encountered")


def _init_state(initial_positions, logp_fn, *params):
    """Initial state"""
    initial_logp = logp_fn(initial_positions, *params)
    return State(initial_positions, initial_logp, 0)


class VMC:

    def __init__(self, wavefunction, rng=default_rng()):
        self._wf = wavefunction
        self._rng = rng

        # Retrieve callables from wave function
        self._logp_fn = self._wf.logpdf
        self._locE = self._wf.locE
        #self._drift_force = self._wf._drift_force

    def sample(
        self,
        ncycles,
        initial_positions,
        alpha,
        scale=0.5
    ):

        self._scale = scale
        self._energies = np.zeros(ncycles)
        state = _init_state(initial_positions, self._logp_fn, alpha)

        for i in range(ncycles):
            state = rw_metropolis_step(state,
                                       self._logp_fn,
                                       alpha,
                                       # rng=self._rng,
                                       # scale=self._scale
                                       )
            self._energies[i] = self._locE(state.positions, alpha)

        # print(state.n_accepted)
        self._acc_rate = state.n_accepted / ncycles

        return np.mean(self._energies)

    def sample2(
        self,
        ncycles,
        initial_positions,
        alpha,
        scale=0.5
    ):

        self._scale = scale
        self._energies = np.zeros(ncycles)

        positions = initial_positions
        logp = self._logp_fn(positions, alpha)
        n_accepted = 0

        for i in range(ncycles):
            proposals = self._rng.normal(loc=positions, scale=scale)
            # Sample log uniform rvs
            log_unif = np.log(self._rng.random(size=positions.shape))
            # Compute proposal log density
            logp_proposal = self._logp_fn(proposals, alpha)
            # Metroplis acceptance criterion
            accept = log_unif < logp_proposal - logp
            # Where accept is True, yield proposal, otherwise keep old state
            positions = np.where(accept, proposals, positions)
            logp = np.where(accept, logp_proposal, logp)
            n_accepted += np.sum(accept)

            self._energies[i] = self._locE(positions, alpha)

        # print(state.n_accepted)
        self._acc_rate = n_accepted / ncycles
        return np.mean(self._energies)

    @property
    def accept_rate(self):
        return self._acc_rate

    @property
    def energy_samples(self):
        return self._energies
