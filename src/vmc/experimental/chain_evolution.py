#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import warnings
from abc import abstractmethod
from functools import partial
from multiprocessing import Lock, RLock

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathos.pools import ProcessPool
from tqdm.auto import tqdm

from ..utils import (advance_PRNG_state, block, check_and_set_nchains,
                     early_stopping, generate_seed_sequence, setup_logger,
                     tune_dt_table, tune_scale_table)
from .state import State

warnings.filterwarnings("ignore", message="divide by zero encountered")


class SamplingNotPerformed(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """
    pass


class ChainEvolution:
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

    def __init__(self, wavefunction, inference_scheme=None, rng=None):

        self._check_inference_scheme(inference_scheme)

        self._wf = wavefunction
        self._inference_scheme = inference_scheme

        if rng is None:
            rng = default_rng
        self._rng = rng

        # Retrieve callables
        self._wf_fn = self._wf.wf
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

    def _check_and_set_iterable(self, item, nchains, item_name):
        if isinstance(item, (int, float)):
            return (item,) * nchains
        elif isinstance(item, (list, tuple, np.ndarray)):
            if not len(item) == nchains:
                msg = (f"{item_name} must be an iterable with length nchains "
                       "or just a scalar value")
                raise ValueError(msg)
        return item

    def _check_initial_positions(self, init_pos, nchains):
        msg = (f"initial_positions must be an iterable with length "
               "nchains or just an array with all the particles "
               "initial configuration")
        if isinstance(init_pos, np.ndarray):
            # check if init_pos is an array with multiple initial positions
            if init_pos.ndim == 3:
                if not len(init_pos) == nchains:
                    raise ValueError(msg)
            else:
                return (init_pos,) * nchains
        elif isinstance(init_pos, (list, tuple)):
            if not len(init_pos) == nchains:
                raise ValueError(msg)
        else:
            raise TypeError("The initial_positions dtype is not supported")

        return init_pos

    def _check_logger(self, log, logger_level):
        if not isinstance(log, bool):
            raise TypeError("'log' must be True or False")

        if not isinstance(logger_level, str):
            raise TypeError("'logger_level' must be passed as str")

    def _initialize_logger(self, log, logger_level, nchains):
        self._check_logger(log, logger_level)
        self._log = log

        if self._log:
            self.logger = setup_logger(self.__class__.__name__,
                                       level=logger_level)
            str_nchains = "chain" if nchains == 1 else "chains"
            msg = (f"Initialize {self._inference_scheme} sampler with "
                   f"{nchains} {str_nchains}")
            self.logger.info(msg)
        else:
            self.logger = None

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
        nchains=1,
        seed=None,
        tune=True,
        tune_iter=10000,
        tune_interval=500,
        retune=True,
        retune_iter=5000,
        retune_interval=250,
        tol_tune=1e-5,
        optimize=True,
        max_iter=50000,
        batch_size=500,
        gradient_method='adam',
        eta=0.01,
        tol_optim=1e-5,
        early_stop=True,
        warm=True,
        warmup_iter=5000,
        log=True,
        logger_level="INFO",
        **kwargs
    ):
        """Sampling procedure

        Parameters
        ----------
        n_samples : int
            Number of energy samples to obtain
        initial_positions : array_like
            Initial points in configuration space
        alpha : float or array_like
            Variational parameter.
        nchains: int
            Number of Markov chains
        seed : int, optional
            Random number generator seed.
        warm : bool, optional
            Whether to run warm-up cycles or not. Default: True
        warmup_iter : int, optional
            The number of warm-up iterations. Deafult: 1000
        tune : bool, optional
            Whether to tune the proposal scale or not. Default: True
        tune_iter : int, optional
            The maximum number of tuning cycles, Default: 5000
        tune_interval : int, optional
            The number of cycles between each tune update. Default: 250
        tol_tune : float, optional
            The tolerance level to decide whether to early stop the tuning or
            not. Default: 1e-5
        optimize : bool, optional
            Whether to optimize the variational parameter or not. Default: True
        max_iter : int, optional
            The maximum number of optimize cycles, Default: 50000
        batch_size : int, optional
            The number of cycles in a batch used in optimization. Default: 500
        gradient_method : str, optional
            The gradient descent optimization method, either 'gd' or 'adam'.
            Default: 'adam'
        eta : float, optional
            The gradient descent optimizer's learning rate. Default: 0.01
        tol_optim : float, optional
            The tolerance level to decide whether to early stop the optimization
            or not. Default: 1e-5
        log : bool, optional
            Whether to show logger or not. Default: True
        logger_level : str, optional
            Logging level. One of ["DEBUG", "INFO", "WARNING", "ERROR",
            "CRITICAL"]. Default: "INFO"
        **kwargs
            Arbitrary keyword arguments are passed to step method of the
            sampling algorithm

        Returns
        -------
        pandas.DataFrame
            A dataframe with sampling results.
        """

        # Logger
        self._initialize_logger(log, logger_level, nchains)

        self._reject_batch = False

        # Settings for tuning
        self._tune = tune
        self._tune_iter = tune_iter
        self._tune_interval = tune_interval
        self._tol_tune = tol_tune

        self._retune = retune
        self._retune_iter = retune_iter
        self._retune_interval = retune_interval

        # Settings for optimize
        self._optimize = optimize
        self._max_iter = max_iter
        self._batch_size = batch_size
        self._gradient_method = gradient_method
        self._tol_optim = tol_optim

        # Flag for early stopping
        self._early_stop = early_stop

        # Settings for warm-up before sampling
        self._warm = warm
        self._warmup_iter = warmup_iter

        # Set and run chains
        nchains = check_and_set_nchains(nchains, self.logger)
        seeds = generate_seed_sequence(seed, nchains)

        if nchains == 1:
            chain_id = 0
            self._final_state, results, self._energies = self._sample(nsamples,
                                                                      initial_positions,
                                                                      alpha,
                                                                      eta,
                                                                      seeds[0],
                                                                      chain_id,
                                                                      **kwargs
                                                                      )

            self._results_full = pd.DataFrame([results])

        else:

            if self._log:
                # for managing output contention
                initializer = tqdm.set_lock(Lock(),)
                initargs = (tqdm.get_lock(),)
            else:
                initializer = None
                initargs = None

            # Handle iterables
            nsamples = self._check_and_set_iterable(nsamples,
                                                    nchains,
                                                    "nsamples")
            initial_positions = self._check_initial_positions(initial_positions,
                                                              nchains)
            alpha = self._check_and_set_iterable(alpha,
                                                 nchains,
                                                 "alpha")
            eta = self._check_and_set_iterable(eta,
                                               nchains,
                                               "eta")
            kwargs = (kwargs,) * nchains
            chain_ids = range(nchains)

            with ProcessPool(nchains) as pool:
                r = zip(*pool.map(self._sample,
                                  nsamples,
                                  initial_positions,
                                  alpha,
                                  eta,
                                  seeds,
                                  chain_ids,
                                  kwargs,
                                  initializer=initializer,
                                  initargs=initargs
                                  ))

            self._final_state, results, self._energies = r
            self._results_full = pd.DataFrame(results)

        if self._log:
            self.logger.info("Sampling done")

        return self.results

    def _sample(self, *args, **kwargs):
        """To be called by process"""

        # Some trickery to get this to work with multiprocessing
        if not kwargs:
            nsamples, initial_positions, alpha, eta, seed, chain_id, kwargs = args
        else:
            nsamples, initial_positions, alpha, eta, seed, chain_id = args

        # Set some flags and counters
        #retune = False
        #rewarm = True
        actual_tune_iter = 0
        actual_optim_iter = 0
        actual_warm_iter = 0
        subtract_iter = 0

        energies_all = []

        # Set initial state
        state = self.initial_state(initial_positions, alpha)

        '''
        # Warm-up?
        if self._warm:
            state, energies = self.warmup_chain(state,
                                                alpha,
                                                seed,
                                                chain_id,
                                                **kwargs
                                                )
            actual_warm_iter += state.delta
            subtract_iter = actual_warm_iter
            energies_all.append(energies)
        '''

        # Tune?
        if self._tune:
            state, energies, kwargs = self.tune_selector(state,
                                                         alpha,
                                                         seed,
                                                         self._tune_iter,
                                                         self._tune_interval,
                                                         chain_id,
                                                         **kwargs
                                                         )
            actual_tune_iter += state.delta  # - subtract_iter
            subtract_iter = actual_tune_iter  # + actual_warm_iter
            energies_all.append(energies)

        '''
        # Rewarm? (Only if both rewarm and tune are True)
        if self._rewarm and self._tune:
            state, energies = self.rewarm_chain(state,
                                                alpha,
                                                seed,
                                                chain_id,
                                                **kwargs
                                                )
            actual_warm_iter += state.delta
            subtract_iter = actual_warm_iter
            energies_all.append(energies)
        '''

        # Optimize?
        if self._optimize:
            state, energies, alpha = self.optimizer(state,
                                                    alpha,
                                                    eta,
                                                    seed,
                                                    chain_id,
                                                    **kwargs
                                                    )
            actual_optim_iter += state.delta - subtract_iter
            subtract_iter = actual_optim_iter + actual_tune_iter + actual_warm_iter
            energies_all.append(energies)
            # retune = True

        '''
        # Retune for good measure, only if optimize and retune are True
        if retune:
            state, kwargs = self.tune_selector(state, alpha, seed, **kwargs)
            actual_tune_iter += state.delta - subtract_iter


        if rewarm:
            state = self.warmup_chain(state, alpha, seed, **kwargs)
            actual_warm_iter += state.delta
            subtract_iter = actual_warm_iter

        '''

        # Warm-up?
        if self._warm:
            state, energies = self.warmup_chain(state,
                                                alpha,
                                                seed,
                                                chain_id,
                                                **kwargs
                                                )
            actual_warm_iter += self._warmup_iter
            #subtract_iter = actual_warm_iter
            energies_all.append(energies)

        # Sample energy
        state, energies = self.sample_energy(nsamples,
                                             state,
                                             alpha,
                                             seed,
                                             chain_id,
                                             **kwargs
                                             )
        energies_all.append(energies)

        # energies_all = np.array(energies_all, dtype=object).flatten()
        energies_all = np.concatenate(energies_all).ravel()
        # print(energies_all.shape)

        results = self._accumulate_results(state,
                                           energies,
                                           nsamples,
                                           alpha,
                                           eta,
                                           actual_warm_iter,
                                           actual_tune_iter,
                                           actual_optim_iter,
                                           chain_id,
                                           **kwargs
                                           )

        return state, results, energies_all

    def _accumulate_results(
        self,
        state,
        energies,
        nsamples,
        alpha,
        eta,
        warm_cycles,
        tune_cycles,
        optim_cycles,
        chain_id,
        **kwargs
    ):
        """
        Gather results
        """

        N, d = state.positions.shape

        total_moves = nsamples
        acc_rate = state.n_accepted / total_moves

        energy = np.mean(energies)
        error = block(energies)
        variance = np.mean(energies**2) - energy**2

        # scaled_energies = energies / N
        # scaled_energy = np.mean(scaled_energies)
        # scaled_error = block(scaled_energies)
        # scaled_variance = np.mean(scaled_energies**2) - scaled_energy**2

        scaled_energy = energy / N
        scaled_error = error / N
        scaled_variance = variance / N

        if self._inference_scheme == "Random Walk Metropolis":
            scale_name = "scale"
            scale_val = kwargs[scale_name]
        elif self._inference_scheme == "Langevin Metropolis-Hastings":
            scale_name = "dt"
            scale_val = kwargs[scale_name]

        results = {"chain_id": chain_id + 1,
                   "nparticles": N,
                   "dim": d,
                   scale_name: scale_val,
                   "eta": eta,
                   "alpha": alpha,
                   "energy": energy,
                   "std_error": error,
                   "variance": variance,
                   "scaled_energy": scaled_energy,
                   "scaled_std_error": error,
                   "scaled_variance": variance,
                   "accept_rate": acc_rate,
                   "nsamples": nsamples,
                   "total_cycles": state.delta,
                   "warmup_cycles": warm_cycles,
                   "tuning_cycles": tune_cycles,
                   "optimize_cycles": optim_cycles
                   }

        return results

    def initial_state(self, initial_positions, alpha):
        state = State(initial_positions,
                      self._logp_fn(initial_positions, alpha),
                      0,
                      0
                      )
        return state

    def warmup_chain(self, state, alpha, seed, chain_id, **kwargs):
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

        if self._log:
            t_range = tqdm(range(self._warmup_iter),
                           desc=f"[Warm-up progress] Chain {chain_id+1}",
                           position=chain_id,
                           leave=True,
                           colour='green')
        else:
            t_range = range(self._warmup_iter)

        energies_tmp = []

        for _ in t_range:
            state = self.step(state, alpha, seed, **kwargs)
            energies_tmp.append(self._locE_fn(state.positions, alpha))

        if self._log:
            t_range.close()

        return state, np.array(energies_tmp)

    def rewarm_chain(self, state, alpha, seed, chain_id, **kwargs):
        """Rewarm the chain for after tuning for rewarm_iter cycles.

        Arguments
        ---------
        State : vmc.State
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

        if self._log:
            t_range = tqdm(range(self._rewarm_iter),
                           desc=f"[Rewarm progress] Chain {chain_id+1}",
                           position=chain_id,
                           leave=True,
                           colour='green')
        else:
            t_range = range(self._rewarm_iter)

        energies_tmp = []

        for _ in t_range:
            state = self.step(state, alpha, seed, **kwargs)
            energies_tmp.append(self._locE_fn(state.positions, alpha))

        if self._log:
            t_range.close()

        return state, np.array(energies_tmp)

    def tune_selector(self, state, alpha, seed, niter, interval, chain_id, retuning=False, **kwargs):
        """Select appropriate tuning procedure"""

        if self._inference_scheme == "Random Walk Metropolis":
            scale = kwargs.pop("scale")
            state, energies, new_scale = self.tune_scale(state,
                                                         alpha,
                                                         seed,
                                                         scale,
                                                         niter,
                                                         interval,
                                                         chain_id,
                                                         retuning,
                                                         **kwargs)
            kwargs = dict(kwargs, scale=new_scale)
        elif self._inference_scheme == "Langevin Metropolis-Hastings":
            dt = kwargs.pop("dt")

            state, energies, new_dt = self.tune_dt(state,
                                                   alpha,
                                                   seed,
                                                   dt,
                                                   niter,
                                                   interval,
                                                   chain_id,
                                                   retuning,
                                                   **kwargs)
            kwargs = dict(kwargs, dt=new_dt)
        else:
            msg = (f"Tuning of {self._inference_scheme} currently not "
                   "available, set tune=False")
            raise ValueError(msg)

        return state, energies, kwargs

    def tune_scale(self, state, alpha, seed, scale, niter, interval, chain_id, retuning, **kwargs):
        """For samplers with scale parameter."""

        if self._log and not retuning:
            t_range = tqdm(range(niter),
                           desc=f"[Tuning progress] Chain {chain_id+1}",
                           position=chain_id,
                           leave=True,
                           colour='green')
        else:
            t_range = range(niter)

        steps_before_tune = interval
        early_stop_counter = 0

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)

        energies_tmp = []

        for i in t_range:
            state = self.step(state, alpha, seed, scale=scale, **kwargs)
            energies_tmp.append(self._locE_fn(state.positions, alpha))
            steps_before_tune -= 1

            if steps_before_tune == 0:
                old_scale = scale
                accept_rate = state.n_accepted / self._tune_interval
                scale = tune_scale_table(old_scale, accept_rate)

                # Reset
                steps_before_tune = interval
                state = State(state.positions, state.logp, 0, state.delta)

                # Early stopping?
                if self._early_stop:
                    if early_stopping(scale, old_scale, tolerance=self._tol_tune):
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0

                    if early_stop_counter > 2:
                        break

        if self._log and not retuning:
            t_range.close()

        return state, np.array(energies_tmp), scale

    def tune_dt(self, state, alpha, seed, dt, niter, interval, chain_id, retuning, **kwargs):
        """For samplers with dt parameter."""

        if self._log and not retuning:
            t_range = tqdm(range(niter),
                           desc=f"[Tuning progress] Chain {chain_id+1}",
                           position=chain_id,
                           leave=True,
                           colour='green')
        else:
            t_range = range(niter)

        steps_before_tune = interval
        early_stop_counter = 0

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)

        energies_tmp = []

        for i in t_range:
            state = self.step(state, alpha, seed, dt=dt, **kwargs)
            energies_tmp.append(self._locE_fn(state.positions, alpha))
            steps_before_tune -= 1

            if steps_before_tune == 0:
                old_dt = dt
                accept_rate = state.n_accepted / self._tune_interval
                dt = tune_dt_table(old_dt, accept_rate)

                # Reset
                steps_before_tune = interval
                state = State(state.positions, state.logp, 0, state.delta)

                # Early stopping?
                if self._early_stop:
                    if early_stopping(dt, old_dt, tolerance=self._tol_tune):
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0

                    if early_stop_counter > 2:
                        break

        if self._log and not retuning:
            t_range.close()

        return state, np.array(energies_tmp), dt

    def optimizer(self, state, alpha, eta, seed, chain_id, **kwargs):
        """Optimize alpha
        """

        if self._log:
            t_range = tqdm(range(self._max_iter),
                           desc=f"[Optimization progress] Chain {chain_id+1}",
                           position=chain_id,
                           leave=True,
                           colour='green')
        else:
            t_range = range(self._max_iter)

        # Set hyperparameters for Adam
        if self._gradient_method == "adam":
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m = 0
            v = 0
            t = 0

        # Set initial config
        steps_before_optimize = self._batch_size
        energies = []
        grad_alpha = []

        # Used for storing all energy samples
        energies_tmp = []

        for i in t_range:
            state = self.step(state, alpha, seed, **kwargs)
            E = self._locE_fn(state.positions, alpha)
            energies.append(E)
            energies_tmp.append(E)
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
                gradient = 2 * (expect1 - expect2 * expect3)

                # Gradient descent
                if self._gradient_method == "gd":
                    alpha -= eta * gradient
                elif self._gradient_method == "adam":
                    t += 1
                    m = beta1 * m + (1 - beta1) * gradient
                    v = beta2 * v + (1 - beta2) * gradient**2
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    alpha -= eta * m_hat / (np.sqrt(v_hat) - epsilon)

                # Reset
                energies = []
                grad_alpha = []
                steps_before_optimize = self._batch_size

                # Early stopping?
                if self._early_stop:
                    if early_stopping(alpha, old_alpha, tolerance=self._tol_optim):
                        break

                # Retune?
                if self._retune:
                    state, energies_tune, kwargs = self.tune_selector(state,
                                                                      alpha,
                                                                      seed,
                                                                      self._retune_iter,
                                                                      self._retune_interval,
                                                                      chain_id,
                                                                      retuning=True,
                                                                      **kwargs
                                                                      )
                    print(energies_tune.shape)
                    energies_tmp.append(energies_tune)

        if self._log:
            t_range.close()

        #energies_out = np.concatenate(energies_tmp).ravel()

        return state, np.array(energies_tmp), alpha

    def sample_energy(self, nsamples, state, alpha, seed, chain_id, **kwargs):

        if self._log:
            t_range = tqdm(range(nsamples),
                           desc=f"[Sampling progress] Chain {chain_id+1}",
                           position=chain_id,
                           leave=True,
                           colour='green')
        else:
            t_range = range(nsamples)

        # Reset n_accepted
        state = State(state.positions, state.logp, 0, state.delta)
        energies = np.zeros(nsamples)

        for i in t_range:
            state = self.step(state, alpha, seed, **kwargs)
            energies[i] = self._locE_fn(state.positions, alpha)

        if self._log:
            t_range.close()

        return state, energies

    def to_csv(self, filename):
        """Write (full) results dataframe to csv.

        Parameters
        ----------
        filename : str
            Output filename
        """
        self.results_all.to_csv(filename, index=False)

    @property
    def results_all(self):
        try:
            return self._results_full
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

    @property
    def results(self):
        df = self.results_all[["nparticles",
                               "dim",
                               "alpha",
                               "energy",
                               "std_error",
                               "variance",
                               "accept_rate"]]
        return df

    @property
    def energy_samples(self):
        try:
            return self._energies
        except AttributeError:
            msg = "Unavailable, a call to sample must be made first"
            raise SamplingNotPerformed(msg)

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
    def energy(self):
        return self.results["energy"]

    @property
    def variance(self):
        return self.results["variance"]

    @property
    def std_error(self):
        return self.results["std_error"]

    @property
    def accept_rate(self):
        return self.results["accept_rate"]


class RWMEvolution(ChainEvolution):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction,
                         inference_scheme='Random Walk Metropolis',
                         rng=rng)

    def step(self, state, alpha, seed, scale=1.0):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : vmc.State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        scale : float
            Scale of proposal distribution. Default: 1.0

        Returns
        -------
        new_state : vmc.State
            The updated state of the system.
        """

        N, dim = state.positions.shape

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=scale)

        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density
        logp_proposal = self._logp_fn(proposals, alpha)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = self._logp_fn(new_positions, alpha)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state


class LMHEvolution(ChainEvolution):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction,
                         inference_scheme='Langevin Metropolis-Hastings',
                         rng=rng
                         )

    def step(self, state, alpha, seed, D=0.5, dt=1.0):
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

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Compute drift force at current positions
        F = self._driftF_fn(state.positions, alpha)

        # Sample proposal positions, i.e., move walkers
        proposals = state.positions + F * Ddt + \
            rng.normal(loc=0, scale=np.sqrt(dt), size=sys_size)

        # Compute proposal log density
        logp_prop = self._logp_fn(proposals, alpha)

        # Green's function conditioned on proposals
        F_prop = self._driftF_fn(proposals, alpha)
        G_prop = -(state.positions - proposals - Ddt * F_prop)**2 * quarterDdt

        # Green's function conditioned on current positions
        G_cur = -(proposals - state.positions - Ddt * F)**2 * quarterDdt

        # Metroplis-Hastings ratio
        # ratio = logp_prop + G_prop - state.logp - G_cur
        ratio = logp_prop + np.sum(G_prop) - state.logp - np.sum(G_cur)

        # Sample log uniform rvs
        # log_unif = np.log(rng.random(size=sys_size))
        log_unif = np.log(rng.random())

        # Metroplis acceptance criterion
        accept = log_unif < ratio

        # Where accept is True, yield proposal, otherwise keep old state
        new_positions = np.where(accept, proposals, state.positions)
        # new_logp = np.where(accept, logp_prop, state.logp)
        new_logp = self._logp_fn(new_positions, alpha)
        new_n_accepted = state.n_accepted + np.sum(accept)
        new_delta = state.delta + 1

        # Create new state
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state
