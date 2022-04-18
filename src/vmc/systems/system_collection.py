#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import distance

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


class ENIBWF(WaveFunction):
    """Single particle wave function with Gaussian kernel for a Non-Interacting
    Boson (NIB) system in an elliptical harmonic oscillator.

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


class AIB(WaveFunction):
    """
    Analytical Interacting Boson (AIB) system.

    Trial wave function:
            psi = exp(-alpha * r**2) * exp(sum(u(rij)))
    """

    def __init__(self, N, dim, omega):
        super().__init__(N, dim)
        self._omega2 = omega * omega
        self._Nd = N * dim
        self._triu_indices = np.triu_indices(N, 1)

    def prepare_handy_values(self, r):
        self.dist_mat = self.distance_matrix(r)
        dist_vec = []
        for i, j in zip(*self._triu_indices):
            dist_vec.append(self.dist_mat[i, j])
        self.dist_vec = np.array(dist_vec)

    def __call__(self, r, alpha):
        """Finds value of wave function.
        Parameters
        ----------
        r           :   np.ndarray, shape=(n_particles, dim)
        alpha       :   float

        Returns
        ---------
        array_like  :   np.ndarray, shape=(n_particles, dim, n_particles)
        """
        u_vector = self.u(r)
        return np.exp(-alpha * r * r + np.sum(u_vector))

    def wf_scalar(self, r, alpha):
        """Finds scalar value of the wave function.

        """
        distance_vector = self.distance_vector(r)
        u_vector = self.u(distance_vector)
        return np.exp(-alpha * np.sum(r * r) + np.sum(u_vector))

    def distance_matrix(self, r, a=0.0043):
        """Finds distances between particles in n_particles times n_particles matrix.
        Parameters:
        ----------
        r               :   np.ndarray, shape=(n_particles, dim)

        Returns:
        ----------
        distance_matrix : np.ndarray, shape=(n_particles, n_particles)
        """
        distance_matrix = distance.cdist(r, r, metric="Euclidean")
        distance_matrix = np.where(distance_matrix < a, 0, distance_matrix)
        return distance_matrix

    def distance_vector(self, r):
        distance_matrix = self.distance_matrix(r)
        distance_vector = []
        for i, j in zip(*self._triu_indices):
            distance_vector.append(distance_matrix[i, j])
        return np.array(distance_vector)

    def unit_matrix(self, r):
        """Orders distances between particles in a 3-dimensional matrix.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)

        Returns:
        ----------
        unit_matrix : np.ndarray, shape=(n_particles, n_particles, dim)
            Matrix filled with the unit vectors between all particles
        """
        N = self._N
        d = self._d
        unit_matrix = np.zeros((N, N, d))
        for i, j in zip(*self._triu_indices):
            rij = np.linalg.norm(r[i] - r[j])
            upper_unit_vector = (r[i] - r[j]) / rij
            unit_matrix[i, j, :] = upper_unit_vector
            unit_matrix[j, i, :] = -upper_unit_vector
        return unit_matrix

    def u(self, r, a=0.0043):
        """Vector of exponent values.
        Parameters:
        -----------
        distance_vector         :   shape = (len(self._triu_indices[0]),)
        a                       :   float
                    hard sphere diameter
        Returns:
        -----------
        u_vec                   :   shape = (len(self._triu_indices[0]),)
                    vector containing values of u
        """
        distance_vector = self.distance_vector(r)
        u = np.where(distance_vector < a, -1e20,
                     np.log(1 - a / distance_vector))
        return u

    def dudr(self, r, a=0.0043):
        """

        Returns:
        ---------
        dudr        : np.ndarray, shape=(n_particles, dim)
        """
        N = self._N
        d = self._d
        distance_matrix = self.distance_matrix(r)
        scaler = np.zeros((N, N, 1))
        for i, j in zip(*self._triu_indices):
            rij = distance_matrix[i, j]
            scaler[i, j] = a / (rij * rij - a * rij)
            scaler[j, i] = scaler[i, j]
        unit_matrix = self.unit_matrix(r)
        dudr = unit_matrix * scaler
        dudr = np.sum(dudr, axis=0)
        #print("Shape dudr: ", dudr.shape)
        return dudr

    def d2udr2(self, r, a=0.0043):
        """

        Returns:
        ---------
        d2udr2      : np.ndarray, shape=(n_particles, n_particles)
        """
        N = self._N
        distance_matrix = self.distance_matrix(r)
        d2udr2 = np.zeros((N, N))
        for i, j in zip(*self._triu_indices):
            rij = distance_matrix[i, j]
            d2udr2[i, j] = a * (a - 2 * rij) / \
                (rij * rij * (a - rij) * (a - rij))
        return d2udr2

    def local_energy(self, r, alpha, a=0.0043):
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
        N = self._N
        d = self._d
        unit_matrix = self.unit_matrix(r)
        distance_matrix = self.distance_matrix(r)
        dudr = self.dudr(r)

        non_interacting_part = self._Nd * alpha + \
            (0.5 * self._omega2 - 2 * alpha * alpha) * np.sum(r * r)
        """
        #print("Shape first term: ", non_interacting_part.shape)
        second_term = -4*alpha*np.sum(np.diag(np.inner(r, dudr)))
        #print("Shape second term: ", second_term.shape)
        scaler = np.zeros((N, N, 1))
        value_fourth_term = np.zeros((N, N))
        for i, j in zip(*self._triu_indices):
            rij = distance_matrix[i,j]
            scaler[i,j,0] = a/(rij*rij-a*rij)
            scaler[j,i,0] = scaler[i,j,0]
            value_fourth_term[i,j] = scaler[i,j,0]*2/rij
            value_fourth_term[j,i] = value_fourth_term[i,j]

        scaled_unit_matrix = scaler*unit_matrix
        #print("Shape scaled unit mat: ", scaled_unit_matrix.shape)
        third_term = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    third_term += np.dot(scaled_unit_matrix[i, j, :],scaled_unit_matrix[i, k, :])

        #print("Shape third term: ", third_term.shape)
        d2udr2 = self.d2udr2(r)
        fourth_term_matrix = d2udr2 + value_fourth_term
        fourth_term = np.sum(np.sum(fourth_term_matrix, axis=0))

        d2udr2 = self.d2udr2(r)
        scaler = np.zeros((N, N, 1))
        value_fourth_term = np.zeros((N, N))
        for i, j in zip(*self._triu_indices):
            rij = distance_matrix[i,j]
            scaler[i,j,0] = a/(rij*rij-a*rij)
            scaler[j,i,0] = scaler[i,j,0]
            value_fourth_term[i,j] = scaler[i,j,0]*2/rij
            value_fourth_term[j,i] = value_fourth_term[i,j]

        scaled_unit_matrix = scaler*unit_matrix
        reduced_scaled_unit_matrix = np.sum(scaled_unit_matrix, axis=)
        second_term = 0
        third_term = 0
        fourth_term = 0
        for k in range(N):
            for j in range(N):

                fourth_term += d2udr2[k, j] + value_fourth_term[k, j]
                for i in range(N):
                    third_term += np.dot(scaled_unit_matrix[k, j, :], scaled_unit_matrix[k, i, :])
        local_energy = non_interacting_part + second_term + third_term + fourth_term
        """
        return local_energy

    def drift_force(self, r, alpha):
        dudr = self.dudr(r)
        drift_force = -4 * alpha * r + dudr
        return drift_force

    def grad_alpha(self, r, alpha):
        return -np.sum(r * r)


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
        # return -alpha * r * r
        return -alpha * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def wf_scalar(self, r, alpha):
        return -alpha * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * r * r


class ElipLogNIB(System):
    """
    Non-Interacting Boson (NIB) system in log domain.

    Trial wave function:
                psi = -alpha * r**2
    """

    def __init__(self):
        super().__init__()

        #self._scaling = 1 / np.sqrt(1 / omega)
        self._beta = 2.82843
        self._gamma2 = self._beta * self._beta

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        r2 = jnp.square(r)

        if r.shape[1] > 2:
            #r[:, 2] *= self._beta
            r = r.at[:, 2].multiply(self._beta)

        return -alpha * r2

    @partial(jax.jit, static_argnums=(0,))
    def wf_scalar(self, r, alpha):

        r2 = jnp.square(r)

        # if r.shape[1] > 2:
        #r[:, 2] *= self._beta
        r2 = r2.at[-1].multiply(self._beta)

        return -alpha * jnp.sum(r)

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):

        r2 = jnp.square(r)
        # if r.shape[1] > 2:
        #r2[:, 2] *= self._gamma2
        #r2 = r2.at[:, 2].multiply(self._gamma2)
        r2 = r2.at[-1].multiply(self._gamma2)
        return 0.5 * r2


class LogIB(System):
    """
    Interacting Boson (IB) system in log domain.

    Trial wave function:
                psi = -alpha * r**2 +np.sum(u)
    """

    def __init__(self, omega, a=0.00433):
        super().__init__()
        self._omega2 = omega * omega
        self._a = a

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        return jnp.sum(self._single(r, alpha)) + jnp.sum(self._correlation(r))

    @partial(jax.jit, static_argnums=(0,))
    def _single(self, r, alpha):
        return -alpha * r * r

    @partial(jax.jit, static_argnums=(0,))
    def _correlation(self, r):
        N = r.shape[0]
        i, j = jnp.triu_indices(N, 1)
        axis = r.ndim - 1
        rij = jnp.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)
        return jnp.log(f)

    @partial(jax.jit, static_argnums=(0,))
    def wf_scalar(self, r, alpha):
        return jnp.sum(self._single(r, alpha)) + jnp.sum(self._correlation(r))

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * r * r
