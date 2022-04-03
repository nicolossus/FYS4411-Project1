#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from abc import ABCMeta, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from metropolis_jax import vmc_jax


def wf(r, alpha):
    return -alpha * r * r


def drift_force_exact(r, alpha):
    # return -4 * alpha * jnp.sum(r, axis=0)
    return -4 * alpha * r


def drift_force(r, alpha):
    def wf_closure(r): return wf(r, alpha)
    primals, grad_f = jax.linearize(wf_closure, r)
    return 2 * grad_f(r) / r


def drift_force2(r, alpha):
    def wf_closure(r): return wf(r, alpha)
    _, tangent = jax.jvp(wf_closure, (r,), (r,))
    return 2 * tangent / r


alpha = 0.5
#r = jnp.array([0.5])
#r = jnp.array([[0.5], [0.5]])
r = jnp.array([[0.5, 0.4, 0.2], [0.7, 0.6, 0.4]])

F_exact = drift_force_exact(r, alpha)
print(F_exact)
print("")


F = drift_force(r, alpha)
print(F)
print("")


F = drift_force2(r, alpha)
print(F)
