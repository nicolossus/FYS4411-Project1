#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from abc import abstractmethod

import numpy as np

from .base_sampler import BaseVMC
from .pool_tools import advance_PRNG_state
from .state import State


class Metropolis(BaseVMC):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction, inference_scheme='metropolis', rng=rng)

    def step(self, state, alpha, seed, scale=1.0):
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
        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=scale)

        # Sample log uniform rvs
<<<<<<< HEAD
        log_unif = np.log(rng.random())
        #log_unif = np.log(rng.random(size=state.positions.shape))
=======
        #log_unif = np.log(rng.random(size=state.positions.shape))
        log_unif = np.log(rng.random())

>>>>>>> cc5939f28d3cdd889fa05371d149eeed9df569d7
        # Compute proposal log density
        logp_proposal = self._logp_fn(proposals, alpha)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp
        #print(accept)
        #print(log_unif)
        #print(logp_proposal)
        #print(state.logp)
        # Where accept is True, yield proposal, otherwise keep old state

        new_positions = np.where(accept, proposals, state.positions)
<<<<<<< HEAD

=======
>>>>>>> cc5939f28d3cdd889fa05371d149eeed9df569d7
        new_logp = self._logp_fn(new_positions, alpha)


        """
        if (accept):
            new_positions = proposals
            new_logp = logp_proposal
            n_accepted = state.n_accepted + 1
        else:
            new_positions = state.positions
            new_logp = state.logp
            n_accepted = state.n_accepted
        """
<<<<<<< HEAD

        new_n_accepted = state.n_accepted + np.sum(accept)
        new_delta = state.delta + 1

        if (np.sum(new_logp-state.logp)<np.sum(log_unif)):
            print("New logp: ", np.sum(new_logp))
            print("statelogp: ", np.sum(state.logp))
            new_positions = state.positions
            new_logp = state.logp
            new_n_accepted = state.n_accepted

=======
        new_n_accepted = state.n_accepted + np.sum(accept)
        new_delta = state.delta + 1

>>>>>>> cc5939f28d3cdd889fa05371d149eeed9df569d7
        # Create new state
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state
