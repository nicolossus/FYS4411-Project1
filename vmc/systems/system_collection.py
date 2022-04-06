#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import System, WaveFunction


class NIBWF(WaveFunction):
    """Single particle wave function with Gaussian kernel for a Non-Interacting
    Boson (NIB) system in a spherical harmonic oscillator.

    Parameters
    ----------
    N : int
        Number of particles in system
    dim : int
        Dimensionality of system
    omega : float
        Harmonic oscillator frequency
    """

    def __init__(self, N, dim, omega):
        super().__init__(N, dim)
        self._omega = omega

        # precompute
        self._Nd = N * dim
        self._halfomega2 = 0.5 * omega * omega

    def __call__(self, r, alpha):
        """Evaluate the trial wave function.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        array_like
            Evaluated trial wave function
        """

        return np.exp(-alpha * r * r)

    def wf_scalar(self, r, alpha):
        """Scalar evaluation of the trial wave function"""

        return np.exp(-alpha * np.sum(r * r))

    def local_energy(self, r, alpha):
        """Compute the local energy.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        float
            Computed local energy
        """

        locE = self._Nd * alpha + \
            (self._halfomega2 - 2 * alpha * alpha) * np.sum(r * r)
        return locE

    def drift_force(self, r, alpha):
        """Drift force"""

        return -4 * alpha * r

    def grad_alpha(self, r, alpha):
        """Gradient of wave function w.r.t. variational parameter alpha"""

        return -np.sum(r * r)


class ANIB:
    """Analytical Non-Interacting Boson (ANIB) system.

    Trial wave function:
                psi = exp(-alpha * r**2)
    """

    def __init__(self, N, dim, omega):
        self._N = N
        self._d = dim
        self._omega = omega

        # precompute
        self._Nd = self._N * self._d
        self._halfomega2 = 0.5 * omega * omega

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, r, alpha):
        return jnp.exp(-alpha * r * r)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, r, alpha):
        wf = self(r, alpha)
        return wf * wf

    @partial(jax.jit, static_argnums=(0,))
    def logprob(self, r, alpha):
        wf = self(r, alpha)
        return jnp.log(wf * wf)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, r, alpha):
        locE = self._Nd * alpha + \
            (self._halfomega2 - 2 * alpha * alpha) * jnp.sum(r * r)
        return locE

    @partial(jax.jit, static_argnums=(0,))
    def drift_force(self, r, alpha):
        return -4 * alpha * r

    @partial(jax.jit, static_argnums=(0,))
    def grad_alpha(self, r, alpha):
        return -jnp.sum(r * r)

    @property
    def N(self):
        return self._N

    @property
    def dim(self):
        return self._d


class LogNIB(System):
    """
    Non-Interacting Boson (NIB) system in log domain.

    Trial wave function:
                psi = -alpha * r**2
    """

    def __init__(self, omega):
        super().__init__()
        self._omega2 = omega * omega

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        return -alpha * r * r

    '''
    @partial(jax.jit, static_argnums=(0,))
    def wf_scalar(self, r, alpha):
        return -alpha * jnp.sum(r * r)
    '''

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * r * r
