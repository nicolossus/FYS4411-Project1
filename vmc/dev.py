from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pathos as pa
from jax import lax
#from jax import grad, jit, lax, pmap, random, vmap
from jax.config import config

config.update("jax_enable_x64", True)


class HOWF:
    def __init__(self, N, dim):
        self._N = N
        self._d = dim

    def __call__(self, x, alpha):
        return np.exp(-0.5 * alpha**2 * x**2)

    @staticmethod
    def evaluate(x, alpha):
        return np.exp(-0.5 * alpha**2 * x**2)

    def pdf(self, x, alpha):
        wf = self(x, alpha)
        return wf * wf

    def logpdf(self, x, alpha):
        wf = self(x, alpha)
        return np.log(wf * wf)

    def locE(self, x, alpha):
        E_L = 0.5 * (alpha**2 + x**2 * (1 - alpha**4))
        return E_L

    def drift_force(self, x, alpha):
        return - 2 * alpha**2 * x

    def grad_local_energy(self, r, alpha):
        return 0.5 * alpha - 1 / (2 * alpha**3)


'''
class JaxWF:

    def __init__(self, wf, potential):
        self._wf = wf
        self._potential = dim

    @partial(jit, static_argnums=(0,))
    def evaluate(self, r, alpha):
        return jnp.exp(-alpha * jnp.sum(r**2))

    @partial(jit, static_argnums=(0,))
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
        E_L = self._N * self._d * alpha + \
            (0.5 * self._omega**2 - 2 * alpha**2) * jnp.sum(r**2)
        return E_L

    @partial(jit, static_argnums=(0,))
    def drift_force_analytical(self, r, alpha):
        return -4 * alpha * jnp.sum(r, axis=0)

    @partial(jit, static_argnums=(0,))
    def drift_force_jax(self, r, alpha):
        grad_wf = grad(self.evaluate)
        F = 2 * jnp.sum(grad_wf(r, alpha), axis=0) / self.evaluate(r, alpha)
        return F
'''

if __name__ == "__main__":
    N = 1
    dim = 1
    alpha = 1
    x = jnp.array([0.5])
    print(x.shape)

    wf = HOWF(N, dim)
    val = wf(x, alpha)
    evalu = wf.evaluate(x, alpha)
    pdf = wf.pdf(x, alpha)
    logpdf = wf.logpdf(x, alpha)
    locE = wf.locE(x, alpha)
    F = wf.drift_force(x, alpha)
    gradE = wf.grad_local_energy(x, alpha)

    print("NumPy")
    # print(f"{val=}")
    # print(f"{evalu=}")
    # print(f"{pdf=}")
    # print(f"{logpdf=}")
    print(f"{locE=}")
    print(f"{F=}")
    # print(f"{gradE=}")

    print("")
    print("JAX")

    def wavef(r, alpha):
        """log wf"""
        return -0.5 * alpha * alpha * jnp.sum(r * r)

    def potential(r):
        """pot = 0.5 * omega**2 * x**2, here omega = 1"""
        return 0.5 * r * r

    def laplace(wf, r, alpha):
        """
        Evaluates the local kinetic energy,
        -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
        """
        n = r.shape[0]
        eye = jnp.eye(n)
        grad_wf = jax.grad(wf, argnums=0)
        def grad_wf_closure(r): return grad_wf(r, alpha)
        primal, dgrad_f = jax.linearize(grad_wf_closure, r)

        # Method 1
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)
        result = -0.5 * jnp.sum(diagonal)

        '''
        # Method 2
        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + dgrad_f(eye[i])[i], 0.0)
        '''
        return result - 0.5 * jnp.sum(primal ** 2)
        # return result

    def hamiltonian(wf, potential, r, alpha):
        ke = laplace(wf, r, alpha)
        pe = potential(r)
        return ke + pe

    def local_energy(wf, potential, r, alpha):
        Hwf = hamiltonian(wf, potential, r, alpha)
        wf_eval = wf(r, alpha)
        loc_E = 1 / wf_eval * Hwf
        return Hwf

    def drift(wf, r, alpha):
        grad_wf = jax.grad(wf, argnums=0)
        grad_wf_val = grad_wf(r, alpha)
        wf_val = wf(r, alpha)
        F = 1 / wf_val * 2 * grad_wf_val
        return F

    locEjax = local_energy(wavef, potential, x, alpha)
    print(locEjax)

    Fjax = drift(wavef, x, alpha)
    print(Fjax)
