#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import jax
import jax.numpy as jnp

from . import System


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
