#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from abc import abstractmethod

import numpy as np

from .base_sampler import BaseVMC
from .state import State


class Metropolis(BaseVMC):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction, inference_scheme='metropolis', rng=rng)

    def step(self, state, alpha, scale=1.0):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        scale : float
            Scale of proposal distribution. Default: 1.0
        """

        # Sample proposal positions, i.e., move walkers
        proposals = self._rng.normal(loc=state.positions, scale=scale)
        # Sample log uniform rvs
        log_unif = np.log(self._rng.random(size=state.positions.shape))
        # Compute proposal log density
        logp_proposal = self._logp_fn(proposals, alpha)
        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp
        # Where accept is True, yield proposal, otherwise keep old state
        new_positions = np.where(accept, proposals, state.positions)
        new_logp = np.where(accept, logp_proposal, state.logp)
        new_n_accepted = state.n_accepted + np.sum(accept)
        # Create new state
        new_state = State(new_positions, new_logp, new_n_accepted)

        return new_state
