from functools import partial
from . import WaveFunction
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.config import config
from abc import ABCMeta, abstractmethod
config.update("jax_enable_x64", True)

class JaxOptimizer:
