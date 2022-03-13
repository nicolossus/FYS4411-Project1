from abc import ABCMeta, abstractmethod
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, hessian
import jax as jax
from jax.config import config

config.update("jax_enable_x64", True)






class BaseJaxWF:

    def __init__(self, wavefunction, potential, n_particles, dim):
        """Base class for computing properties of trial wave functon,
        using jax and autodifferentiation.

        Parameters
        ----------
        wavefunction    : function
            Trial wavefunction
        potential       : function
            Potential used in Hamiltonian of system
        n_particles     : int
            Number of particles in system
        dim             : int
            Dimensionality of system
        """
        # Error handling
        if not isinstance(n_particles, int):
            msg = "The number of particles in the system must be passed as int"
            raise TypeError(msg)
        if not isinstance(dim, int):
            msg = "The dimensionality of the system must be passed as int"
            raise TypeError(msg)
        if not n_particles > 0:
            msg = "The number of particles must be > 0"
            raise ValueError(msg)
        if not 1 <= dim <= 3:
            msg = "Dimensionality must be between 1D, 2D or 3D"
            raise ValueError(msg)

        # Initialise system
        self._N = n_particles
        self._d = dim

        self.wf = wavefunction
        self.potential = potential

        # Finding functions needed to evaluate properties of system.
        self.grad_wf = grad(self.wf, argnums=0, holomorphic=False)
        self.hessian = hessian(self.wf, argnums=0, holomorphic=False)
        if dim == 1:
            self.lap = self.laplacian
        elif dim==2:
            self.lap = self.polar_lap
        else:
            self.lap = self.spherical_laplacian

        self.loc_energy = self.local_energy

    def convert_to_distance(self, r):
        distance_squared = jnp.sum(r**2, axis=1)
        distance = jnp.sqrt(distance_squared)
        return distance

    def spherical_laplacian(self, r, alpha):
        grad2_wf = jax.jacfwd(self.sl_middle, argnums=0)
        laplacian = jnp.sum(jnp.diag(grad2_wf(r, alpha)/(r*r)))
        return laplacian

    def sl_middle(self, r, alpha):
        return r*r*self.grad_wf(r, alpha)

    def polar_lap(self, r, alpha):
        grad2_wf = jax.jacfwd(self.polar_middle, argnums=0)
        laplacian = jnp.sum(jnp.diag(grad2_wf(r, alpha)/r))
        return laplacian

    def polar_middle(self, r, alpha):
        return r*self.grad_wf(r, alpha)

    def hamiltonian(self, r, alpha):
        #kinetic = -0.5*jnp.sum(jnp.diag(jnp.sum(jnp.sum(self.hessian(r, alpha), axis=1), axis=-1)))
        r = self.convert_to_distance(r)
        kinetic = -0.5*self.lap(r, alpha)
        return kinetic + self.potential(r, self._omega)*self.wf(r, alpha)

    def density(self, r, alpha):
        return self.wf(r, alpha)*self.wf(r, alpha)

    def gradient_wf(self, r, alpha):
        return jnp.sum(self.grad_wf(r, alpha), axis=0)

    def logdensity(self, r, alpha):
        return jnp.log(self.wf(r, alpha)*self.wf(r, alpha))

    def reduced_hessian(self, r, alpha):
        return jnp.sum(jnp.sum(self.hessian(r, alpha), axis=1), axis=-1)

    def laplacian_vector(self, r, alpha):
        return jnp.diag(self.reduced_hessian(r, alpha))

    def laplacian(self, r, alpha):
        return jnp.sum(self.lap_vector(r, alpha))

    def local_energy(self, r, alpha):
        H = self.hamiltonian(r, alpha)
        return H/self.wf(r, alpha)

    def drift_force(self, r, alpha):
        #grad_wf = grad(self.evaluate)
        F = 2 *self.gradient_wf(r, alpha) / self.wf(r, alpha)
        return F

    @property
    def dim(self):
        return self._d

    @property
    def n_particles(self):
        return self._N


class SG(BaseJaxWF):
    """Single particle wave function with Gaussian kernel.

    Parameters
    ----------
    n_particles : int
        Number of particles in system
    dim : int
        Dimensionality of system
    omega : float
        Harmonic oscillator frequency
    """

    def __init__(self, wavefunction, potential, _particles, dim, omega):
        super().__init__(wavefunction, potential, _particles, dim)
        self._omega = omega


def wavefunction(r, alpha):
    return jnp.exp(-alpha*jnp.sum(r**2))

def potential(r, _omega):
    return 0.5*_omega*_omega*jnp.sum(r**2)

def drift_force_analytical(r, alpha):
    return -4 * alpha * jnp.sum(r, axis=0)

def local_energy(N, d, omega, r, alpha):
    E_L =N * d * alpha + \
        (0.5 * omega**2 - 2 * alpha**2) * jnp.sum(r**2)
    return E_L

def laplacian(N, d, r, alpha):
    grad2 = (-2*N * d * alpha + 4 *
             alpha**2 * np.sum(r**2)) * wavefunction(r, alpha)
    return grad2


if __name__ == "__main__":
    from jax import random
    N = 10
    d = 3
    alpha = 0.5
    omega = 1

    key = random.PRNGKey(0)
    r = random.normal(key, (N, d))
    print(r.shape)

    # print(r)
    assert np.any(np.array(r) < 0)

    r_np = np.array(r)

    wf = SG(wavefunction, potential, N, d, omega)
    spherical_r = wf.convert_to_distance(r)
    print("Grad_wf(spherical): ", wf.grad_wf(spherical_r, alpha))
    print("Grad_wf(r): ", jnp.sum(wf.grad_wf(r, alpha), axis=1))
    print("Spherical laplacian: ", wf.spherical_laplacian(spherical_r, alpha))
    print("Local energy jax: ", wf.loc_energy(r, alpha))
    print("Analytical energy: ", local_energy(N, d, omega, r, alpha))
    print("-gradient/2 + potential: ", (-0.5*wf.laplacian(r, alpha)+wf.potential(r, omega)))
    print("Drift F analytical:", drift_force_analytical(r, alpha))
    print("Drift F Jax:", wf.drift_force(r, alpha))
    print("Hessian F shape Jax: ", wf.hessian(r, alpha).shape)
    print("Laplacian vector: ", wf.lap_vector(r, alpha))
    print("Jax Laplacian: ", wf.laplacian(r, alpha))
    print("Analytical laplacian: ", laplacian(N, d, r, alpha))
    print("Polar laplacian: ", wf.polar_lap(spherical_r, alpha))
    print("Spherical laplacian: ", wf.spherical_laplacian(spherical_r, alpha))
    print("Potential: ", wf.potential(r, alpha))
    print("N*d*alpha: ", N*d*alpha)
