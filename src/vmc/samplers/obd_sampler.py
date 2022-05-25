#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import warnings
from abc import abstractmethod
from functools import partial

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathos.pools import ProcessPool

from .pool_tools import check_and_set_jobs, generate_seed_sequence
from .sampler_utils import tune_scale_table
from .state import State

warnings.filterwarnings("ignore", message="divide by zero encountered")


class SamplingNotPerformed(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """
    pass


class OBDVMC:
    """

    Arguments
    ---------
    wavefunction : vmc_numpy.WaveFunction
        Wave function with methods dictated by vmc_numpy.WaveFunction
    inference_scheme : str
        Name of inference scheme. Can be used to condition for algorithm
        specific methods.
    rng : generator
        Random number generator. Default: numpy.random.default_rng
    """

    def __init__(self, wavefunction, rng=None):

        self._wf = wavefunction

        if rng is None:
            rng = default_rng
        self._rng = rng

        # Retrieve callables
        self._wf_fn = self._wf.wf
        self._logp_fn = self._wf.logprob
        self._pdf = self._wf.pdf

    @abstractmethod
    def step(self, particle, state, alpha, **kwargs):
        """Single sampling step.

        To be overwritten by subclass. Signature must be as shown in the
        abstract method, meaning that algorithm specific parameters must be
        set via keyword arguments.
        """

        raise NotImplementedError

    def sample(
        self,
        nsamples,
        initial_positions,
        alpha,
        scale=1.0,
        nchains=1,
        seed=None,
        warm=True,
        warmup_iter=500,
        tune=True,
        tune_iter=5000,
        tune_interval=250,
        normalize=False,
    ):
        """Sampling procedure"""
        # Settings for warm-up
        self._warm = warm
        self._warmup_iter = warmup_iter

        # Settings for tuning
        self._tune = tune
        self._tune_iter = tune_iter
        self._tune_interval = tune_interval


        # Set and run chains
        nchains = check_and_set_jobs(nchains)
        seeds = generate_seed_sequence(seed, nchains)

        if nchains == 1:
            state, positions = self._sample(nsamples,
                                            initial_positions,
                                            alpha,
                                            seeds[0],
                                            scale,
                                            normalize=normalize
                                            )

            #self._results_full = pd.DataFrame([results])

        else:
            # kwargs
            # iterables
            nsamples = (nsamples,) * nchains
            initial_positions = (initial_positions,) * nchains
            alpha = (alpha,) * nchains
            eta = [eta] * nchains
            kwargs = (kwargs,) * nchains

            # nsamples, initial_positions, alpha, eta, **kwargs

            with ProcessPool(nchains) as pool:
                # , self._distances
                state, positions = zip(*pool.map(self._sample,
                                                 nsamples,
                                                 initial_positions,
                                                 alpha,
                                                 seeds,
                                                 scale,
                                                 normalize=normalize
                                                 ))

                #self._results_full = pd.DataFrame(results)

        #print("Shape of pdfs within sample: ", self._pdfs.shape)
        return state, positions

    def _sample(self, nsamples, initial_positions, alpha, seed, scale, normalize=False):
        """To be called by process"""

        # Set some flags and counters
        rewarm = True
        actual_warm_iter = 0
        actual_tune_iter = 0
        actual_optim_iter = 0
        subtract_iter = 0

        # Set initial state
        state = self.initial_state(initial_positions, alpha)

        # Warm-up?
        if self._warm:
            state = self.warmup_chain(state, alpha, seed, scale)
            actual_warm_iter += state.delta
            subtract_iter = actual_warm_iter

        # Tune?
        if self._tune:
            state, scale = self.tune_scale(state, alpha, seed, scale)
            actual_tune_iter += state.delta - subtract_iter
            subtract_iter = actual_tune_iter + actual_warm_iter

        if rewarm:
            state = self.warmup_chain(state, alpha, seed, scale)
            actual_warm_iter += state.delta
            subtract_iter = actual_warm_iter

        # Sample pdfs given one particle position
        state, positions = self.sample_positions(nsamples,
                                                 state,
                                                 alpha,
                                                 seed,
                                                 scale,
                                                 normalize=normalize
                                                 )

        mean_avg_distance, acc_rate = self._accumulate_results(state,
                                                       positions,
                                                       nsamples,
                                                       alpha
                                                       )
        print(f"Acceptance rate={acc_rate}")
        print(f"Avg radial distance={mean_avg_distance}")
        return state, positions

    def _accumulate_results(
        self,
        state,
        positions,
        nsamples,
        alpha,
    ):
        """
        Gather results
        """

        total_moves = nsamples

        acc_rate = state.n_accepted / total_moves
        radial_distances = np.linalg.norm(positions, axis=2)
        mean_radial_distances = np.mean(radial_distances, axis=1)
        average_radial_distance = np.mean(mean_radial_distances)

        return average_radial_distance, acc_rate

    def initial_state(self, initial_positions, alpha):
        state = State(initial_positions,
                      self._logp_fn(initial_positions, alpha),
                      0,
                      0
                      )
        return state

    def warmup_chain(self, state, alpha, seed, scale):
        """Warm-up the chain for warmup_iter cycles.

        Arguments
        ---------
        warmup_iter : int
            Number of cycles to warm-up the chain
        State : vmc_numpy.State
            Current state of the system
        alpha : float
            Variational parameter
        **kwargs
            Arbitrary keyword arguments are passed to the step method

        Returns
        -------
        State
            The state after warm-up
        """
        for i in range(self._warmup_iter):
            state = self.step(state, alpha, seed, scale=scale)
        return state


    def tune_scale(self, state, alpha, seed, scale):
        """For samplers with scale parameter."""

        steps_before_tune = self._tune_interval
        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)
        total_moves = self._tune_interval
        count = 0
        for i in range(self._tune_iter):
            state = self.step(state, alpha, seed, scale=scale)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                old_scale = scale
                accept_rate = state.n_accepted / total_moves
                scale = tune_scale_table(old_scale, accept_rate)
                if scale == old_scale:
                    count += 1
                else:
                    count = 0

                # Reset
                steps_before_tune = self._tune_interval
                state = State(state.positions, state.logp, 0, state.delta)

                # Early stopping? If the same scale has appeared
                # three times in a row, break.
                if count>2:
                    break

        return state, scale

    def sample_positions(self, nsamples, state, alpha, seed, scale, normalize=False):

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)
        N, d = state.positions.shape
        positions_array = np.zeros((nsamples, N, d))

        for i in range(nsamples):
            state = self.step(state, alpha, seed, scale, normalize=normalize)
            positions_array[i, :, :] = state.positions

        return state, positions_array


    @property
    def final_state(self):
        try:
            return self._final_state
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def alpha(self):
        return self.results["alpha"]

    @property
    def accept_rate(self):
        return self.results["accept_rate"]
