#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from abc import abstractmethod

import numpy as np

from .base_sampler import BaseVMC
from .state import State


class MetropolisHastings(BaseVMC):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction,
                         inference_scheme='metropolis-hastings',
                         rng=rng
                         )

    def step(self, state, alpha, D=0.5, dt=1.0):
        """One step of the ...

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        D : float
            Diffusion constant. Default: 0.5
        dt : float
            Scale of proposal distribution. Default: 1.0
        """

        # Precompute
        Ddt = D * dt
        quarterDdt = 1 / (4 * Ddt)
        sys_size = state.positions.shape
        # Compute drift force at current positions
        F = self._driftF_fn(state.positions, alpha)
        # Sample proposal positions, i.e., move walkers
        proposals = state.positions + F * Ddt + \
            self._rng.normal(loc=0, scale=np.sqrt(dt), size=sys_size)
        # Compute proposal log density
        logp_prop = self._logp_fn(proposals, alpha)
        # Green's function conditioned on proposals
        F_prop = self._driftF_fn(proposals, alpha)
        G_prop = -(state.positions - proposals -
                   Ddt * F_prop)**2 * quarterDdt
        # Green's function conditioned on current positions
        G_cur = -(proposals - state.positions - Ddt * F)**2 * quarterDdt
        # Metroplis-Hastings ratio
        ratio = logp_prop + G_prop - state.logp - G_cur
        # Sample log uniform rvs
        log_unif = np.log(self._rng.random(size=sys_size))
        # Metroplis acceptance criterion
        accept = log_unif < ratio
        # Where accept is True, yield proposal, otherwise keep old state
        new_positions = np.where(accept, proposals, state.positions)
        new_logp = np.where(accept, logp_prop, state.logp)
        new_n_accepted = state.n_accepted + np.sum(accept)
        # Create new state
        new_state = State(new_positions, new_logp, new_n_accepted)

        return new_state
