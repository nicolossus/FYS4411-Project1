#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class VMC:

    def __init__(self, wavefunction):
        self._psi = wavefunction
        self._n_particles = self._psi.n_particles
        self._dim = self._psi.dim
        self._rng = np.random.default_rng()

    def sample(self, ncycles, alphas, initial_state=None, burn=100, proposal_dist='normal', scale=0.5):
        self._ncycles = ncycles
        self._scale = scale

        if proposal_dist == 'uniform':
            self._propsal_dist = self._draw_proposal_uniform
        elif proposal_dist == 'normal':
            self._propsal_dist = self._draw_proposal_gaussian

        energies = np.zeros(len(alphas))
        variances = np.zeros(len(alphas))
        # errors = np.zeros(len(alphas))

        for i, alpha in enumerate(alphas):
            results = self._metropolis(alpha, initial_state)
            energies[i] = results[0]
            variances[i] = results[1]
            # errors = results[2]

        return energies, variances

    def _metropolis(self, alpha, initial_state):
        if initial_state is None:
            positions = self._initial_positions()
        else:
            positions = initial_state

        wf2 = self._psi.squared(positions, alpha)
        energy = 0
        energy2 = 0

        for _ in range(self._ncycles):
            trial_positions = self._propsal_dist(positions)
            trial_wf2 = self._psi.squared(trial_positions, alpha)

            # Metropolis acceptance criterion
            if self._rng.random() <= trial_wf2 / wf2:
                positions = trial_positions
                wf2 = trial_wf2

            local_energy = self._psi.local_energy(positions, alpha)
            energy += local_energy
            energy2 += local_energy**2

        # Calculate mean, variance, error
        energy /= self._ncycles
        energy2 /= self._ncycles
        variance = energy - energy2
        # error = np.sqrt(variance / self._ncycles)

        return energy, variance

    def _tuning(self, initial_positions):
        # ...
        return positions, wf2

    def _initial_positions(self):
        '''
        initial_state = self._rng.normal(
            loc=np.zeros((self._n_particles, self._dim)),
            scale=self._scale
        )
        '''
        initial_state = self._rng.random(size=(self._n_particles, self._dim))
        return initial_state

    def _draw_proposal_gaussian(self, old_positions):
        return self._rng.normal(loc=old_positions, scale=self._scale)

    def _draw_proposal_uniform(self, old_positions):
        return old_positions + (self._rng.random(size=(self._n_particles, self._dim)) - 0.5)

    def _tune_scale_table(self):
        """Proposal scale lookup table.

        10000 ncycles
        1000 tuning
        100 tune interval
        acceptance rate = n_accepted / tune_interval

        Function retrieved from PyMC3 source code.

        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:

        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        """
        if acc_rate < 0.001:
            # reduce by 90 percent
            self._scale *= 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            return scale * 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            return scale * 0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            return scale * 10.0
        elif acc_rate > 0.75:
            # increase by double
            return scale * 2.0
        elif acc_rate > 0.5:
            # increase by ten percent
            return scale * 1.1

        return scale


if __name__ == "__main__":
    from system import WaveFunction

    # System
    N = int(1e0)
    d = 3
    omega = 1
    psi = WaveFunction(omega, N, d)

    # Sampler
    vmc_sampler = VMC(psi)
    ncycles = 10000
    alphas = [0.25, 0.5]

    vmc_sampler.sample(ncycles, alphas)