#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from abc import abstractmethod

import numpy as np
from numpy.random import default_rng

from .sampler_utils import early_stopping, tune_dt_table, tune_scale_table
from .state import State

warnings.filterwarnings("ignore", message="divide by zero encountered")


class BaseVMC:
    """

    Arguments
    ---------
    wavefunction : vmc_numpy.WaveFunction
        Wave function with methods dictated by vmc_numpy.WaveFunction
    inference_scheme : str
        Name of inference scheme. Can be used to condition for algorithm
        specific methods.
    rng : generator
        Random number generator. Default: numpy.random.default_rng()
    """

    def __init__(self, wavefunction, inference_scheme=None, rng=None):

        self._check_inference_scheme(inference_scheme)

        self._wf = wavefunction
        self._inference_scheme = inference_scheme

        if rng is None:
            rng = default_rng()
        self._rng = rng

        # Retrieve callables
        self._wf_scalar_fn = self._wf.wf_scalar
        self._logp_fn = self._wf.logprob
        self._locE_fn = self._wf.local_energy
        self._driftF_fn = self._wf.drift_force
        self._grad_alpha_fn = self._wf.grad_alpha

    def _check_inference_scheme(self, inference_scheme):
        if inference_scheme is None:
            msg = 'inference_scheme must be passed to the base vmc constructor'
            raise ValueError(msg)

        if not isinstance(inference_scheme, str):
            msg = 'inference_scheme must be passed as str'
            raise TypeError(msg)

    @abstractmethod
    def step(self, state, alpha, **kwargs):
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
        n_chains=1,
        burn=500,
        warm=True,
        warmup_iter=500,
        tune=True,
        tune_iter=5000,
        tune_interval=250,
        tol_tune=1e-5,
        optimize=True,
        max_iter=10000,
        batch_size=500,
        gradient_method='adam',
        eta=0.01,
        tol_optim=1e-5,
        **kwargs
    ):
        """Sampling procedure"""

        # Set some flags
        retune = False

        # Set initial state
        state = self.initial_state(initial_positions, alpha)

        # Warm-up?
        if warm:
            state = self.warmup_chain(warmup_iter, state, alpha, **kwargs)

        # Tune?
        if tune:
            state, kwargs = self.tune_selector(tune_iter,
                                               tune_interval,
                                               tol_tune,
                                               state,
                                               alpha,
                                               **kwargs)

        # Optimize?
        if optimize:
            self.gd_selector(gradient_method, eta)
            state, alpha = self.optimizer(max_iter,
                                          batch_size,
                                          tol_optim,
                                          state,
                                          alpha,
                                          **kwargs)
            retune = True

        # Retune for good measure
        if retune:
            state, kwargs = self.tune_selector(tune_iter,
                                               tune_interval,
                                               tol_tune,
                                               state,
                                               alpha,
                                               **kwargs)

        # Sample energy
        state, energies = self.sample_energy(nsamples, state, alpha, **kwargs)

        # make method for computing final results like acc_rate, mean_energy, etc
        N, d = state.positions.shape
        total_moves = nsamples * N * d
        self._acc_rate = state.n_accepted / total_moves

        #print("alpha:", alpha)

        return np.mean(energies)

    def _sample(self):
        """To be called by process"""
        pass

    def initial_state(self, initial_positions, alpha):
        state = State(initial_positions,
                      self._logp_fn(initial_positions, alpha),
                      0)
        return state

    def warmup_chain(self, warmup_iter, state, alpha, **kwargs):
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

        for _ in range(warmup_iter):
            state = self.step(state, alpha, **kwargs)

        return state

    def tune_selector(
        self,
        tune_iter,
        tune_interval,
        tol_tune,
        state,
        alpha,
        **kwargs
    ):
        """Select appropriate tuning procedure"""

        if self._inference_scheme == "metropolis":
            scale = kwargs.pop("scale")
            state, new_scale = self.tune_scale(tune_iter,
                                               tune_interval,
                                               tol_tune,
                                               state,
                                               alpha,
                                               scale,
                                               **kwargs)
            kwargs = dict(kwargs, scale=new_scale)
        elif self._inference_scheme == "metropolis-hastings":
            dt = kwargs.pop("dt")

            state, new_dt = self.tune_dt(tune_iter,
                                         tune_interval,
                                         tol_tune,
                                         state,
                                         alpha,
                                         dt,
                                         **kwargs)
            kwargs = dict(kwargs, dt=new_dt)
        else:
            msg = (f"Tuning of {self._inference_scheme} currently not "
                   "available, set tune=False")
            raise ValueError(msg)

        return state, kwargs

    def gd_selector(self, gradient_method, eta):
        self._eta = eta

        if gradient_method == 'gd':
            self._gradient_method = self._gradient_descent
        elif gradient_method == 'adam':
            self._beta1 = 0.9
            self._beta2 = 0.999
            self._epsilon = 1e-8
            self._m = 0
            self._v = 0
            self._t = 0
            self._gradient_method = self._adam
        else:
            msg = ("Gradient method not available, use 'gd' or 'adam'")
            raise ValueError(msg)

    def tune_scale(
        self,
        tune_iter,
        tune_interval,
        tol_tune,
        state,
        alpha,
        scale,
        **kwargs
    ):
        """For samplers with scale parameter."""

        steps_before_tune = tune_interval
        # Reset n_accepted
        state = State(state.positions, state.logp, 0)
        N, d = state.positions.shape
        total_moves = tune_interval * N * d

        for i in range(tune_iter):
            state = self.step(state, alpha, scale=scale, **kwargs)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                old_scale = scale
                accept_rate = state.n_accepted / total_moves
                scale = tune_scale_table(old_scale, accept_rate)

                # Reset
                steps_before_tune = tune_interval
                state = State(state.positions, state.logp, 0)

                # Early stopping?
                if early_stopping(scale, old_scale, tolerance=tol_tune):
                    #print(f"Tune early stopping at iter {i+1}/{tune_iter}")
                    break

        return state, scale

    def tune_dt(
        self,
        tune_iter,
        tune_interval,
        tol_tune,
        state,
        alpha,
        dt,
        **kwargs
    ):
        """For samplers with dt parameter."""

        steps_before_tune = tune_interval
        # Reset n_accepted
        state = State(state.positions, state.logp, 0)
        N, d = state.positions.shape
        total_moves = tune_interval * N * d

        for i in range(tune_iter):
            state = self.step(state, alpha, dt=dt, **kwargs)
            steps_before_tune -= 1

            if steps_before_tune == 0:
                old_dt = dt
                accept_rate = state.n_accepted / total_moves
                dt = tune_dt_table(old_dt, accept_rate)

                # Reset
                steps_before_tune = tune_interval
                state = State(state.positions, state.logp, 0)

                # Early stopping?
                if early_stopping(dt, old_dt, tolerance=tol_tune):
                    #print(f"Tune early stopping at iter {i+1}/{tune_iter}")
                    break

        return state, dt

    def sample_energy(self, nsamples, state, alpha, **kwargs):

        # Reset n_accepted
        state = State(state.positions, state.logp, 0)

        energies = np.zeros(nsamples)

        for i in range(nsamples):
            state = self.step(state, alpha, **kwargs)
            energies[i] = self._locE_fn(state.positions, alpha)

        return state, energies

    def optimizer(self, max_iter, batch_size, tol_optim, state, alpha, **kwargs):
        """Optimize alpha -> consider moving grad alpha to logspace to get
        rid of wf eval
        """

        steps_before_optimize = batch_size

        #wf_evals = []
        energies = []
        grad_alpha = []

        for i in range(max_iter):
            state = self.step(state, alpha, **kwargs)
            #wf_evals.append(self._wf_scalar_fn(state.positions, alpha))
            energies.append(self._locE_fn(state.positions, alpha))
            grad_alpha.append(self._grad_alpha_fn(state.positions, alpha))
            steps_before_optimize -= 1

            if steps_before_optimize == 0:
                old_alpha = alpha

                # Expectation values
                energies = np.array(energies)
                grad_alpha = np.array(grad_alpha)
                expect1 = np.mean(grad_alpha * energies)
                expect2 = np.mean(grad_alpha)
                expect3 = np.mean(energies)
                gradE = 2 * (expect1 - expect2 * expect3)

                # Gradient descent
                alpha = self._gradient_method(alpha, gradE)

                # Reset
                energies = []
                grad_alpha = []
                steps_before_optimize = batch_size

                # Early stopping?
                if early_stopping(alpha, old_alpha, tolerance=tol_optim):
                    break

        return state, alpha

    def _gradient_descent(self, alpha, gradient):
        alpha -= self._eta * gradient
        return alpha

    def _adam(self, alpha, gradient):
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * gradient
        self._v = self._beta2 * self._v + (1 - self._beta2) * gradient**2
        m_hat = self._m / (1 - self._beta1**self._t)
        v_hat = self._v / (1 - self._beta2**self._t)
        alpha -= self._eta * m_hat / (np.sqrt(v_hat) - self._epsilon)
        return alpha

    @property
    def accept_rate(self):
        return self._acc_rate
