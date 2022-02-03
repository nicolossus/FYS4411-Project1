#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class WaveFunction:

    def __init__(self, omega, n_particles, dim):
        self._omega = omega
        self._N = n_particles
        self._d = dim

    def __call__(self, r, alpha):
        """Evaluate trial wave function.

        Note that
            psi = np.prod(np.exp(-alpha * r**2))
        is equivalent to
            psi = np.exp(-alpha * np.sum(r**2))

        As np.prod is much slower than np.sum, we use the latter.

        In benchmark with num_particles=1e6 and dim=3:
        Numpy prod: ~11.4985 sec
        Numpy sum: ~0.3397 sec
        (Pure Python with double loop summation of exponent: ~102.65 sec)
        """
        return np.exp(-alpha * np.sum(r**2))

    def squared(self, r, alpha):
        return self(r, alpha)**2

    def gradient(self, r, alpha):
        return -4 * alpha * np.sum(r) * self(r, alpha)

    def laplacian(self, r, alpha):
        grad2 = (-2 * self._N * self._d * alpha + 4 *
                 alpha**2 * np.sum(r**2)) * self(r, alpha)
        return grad2

    def local_energy(self, r, alpha):
        E_L = self._N * self._d * alpha + \
            (0.5 * self._omega**2 - 2 * alpha**2) * np.sum(r**2)
        return E_L

    def drift_force(self, r, alpha):
        return -4 * alpha * np.sum(r)

    def dim(self):
        return self._d

    @property
    def dim(self):
        return self._d

    @property
    def n_particles(self):
        return self._N


if __name__ == "__main__":

    np.random.seed(42)
    N = int(1e1)
    d = 3
    alpha = 0.5
    omega = 1
    #print(np.prod(np.array([1, 2**2, 3**2])))
    #print(np.sum(np.array([1, 2**2, 3**2])))
    r = np.ones((N, d), dtype=np.double) * np.random.random(size=(N, d))
    # print(r)
    psi = WaveFunction(omega, N, d)
    print(psi(r, alpha))
    print(psi.squared(r, alpha))
    print(psi.gradient(r, alpha))
    print(psi.laplacian(r, alpha))
    print(psi.local_energy(r, alpha))
