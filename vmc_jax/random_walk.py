#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial
from typing import Any, Iterable, Mapping, NamedTuple, Union

import jax.numpy as jnp
import jax.scipy as scipy
import numpy as onp
from jax import config, grad, jit, lax, pmap, random, vmap

config.update("jax_enable_x64", True)

Array = Union[onp.ndarray, jnp.ndarray]
PyTree = Union[Array, Iterable[Array], Mapping[Any, Array]]


class State(NamedTuple):
    position: PyTree


def step(state, rng_key):
    move = random.normal(rng_key, shape=state.position.shape)
    new_position = state.position + move
    new_state = State(new_position)
    return new_state, state


def sampler(rng_key, n_samples, initial_state):
    rng_keys = random.split(rng_key, n_samples)
    _, states = lax.scan(step, initial_state, rng_keys)
    return states


def run(seed, n_samples, nchains, initial_positions):
    rng_seed = random.PRNGKey(seed)
    base_keys = random.split(rng_seed, n_chains)

    run_sampler = vmap(sampler, in_axes=(0, None, None), out_axes=0)

    initial_state = State(initial_positions)
    states = run_sampler(base_keys, n_samples, initial_state)
    samples = jnp.stack(states.position).reshape(n_chains, n_samples)

    return samples


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    seed = 42
    n_samples = int(1e4)
    n_chains = 4

    initial_position = jnp.zeros(1)
    start = time.time()
    positions = run(seed, n_samples, n_chains, initial_position)
    end = time.time()
    print("Elapsed time:", end - start)

    for i in range(n_chains):
        plt.plot(positions[i, :], label=f'chain {i}')
    plt.legend(loc='upper left')
    plt.xlabel("Iteration")
    plt.ylabel("Position")
    plt.show()
