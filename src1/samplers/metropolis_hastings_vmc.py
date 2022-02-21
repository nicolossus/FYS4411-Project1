#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class MetropolisHastingsVMC:

    def __init__(self, wavefunction, seed=None):
        self._wf = wavefunction
        self._seed = seed

        self._N = self._wf.n_particles
        self._dim = self._wf.dim
        self._rng = np.random.default_rng(seed=self._seed)

    def sample(
        self,
        ncycles,
        alphas,
        dt=0.01):
        self._ncycles = ncycles
        self._alphas = alphas
        self._dt = dt

        initial_state = None
        energies = np.zeros(len(self._alphas))
        variances = np.zeros(len(self._alphas))

        self._propsal_dist = self._draw_proposal_gaussian

        for i, alpha in enumerate(alphas):
            initial_state = self._safe_initialization(alpha)

            results = self._metropolis_hastings(alpha, initial_state)
            energies[i] = results[0]
            variances[i] = results[1]

        return energies, variances

    def _metropolis_hastings(self, alpha, initial_state):
        if initial_state is None:
            positions = self._initial_positions()
        else:
            positions = initial_state
        dt = self._dt/alpha
        u = self._rng.random(size=self._ncycles)
        wf2 = self._wf.density(positions, alpha)
        qforceOLD = self._wf.drift_force(positions, alpha)
        D = 0.5

        n_accepted = 0
        energy = 0
        energy2 = 0

        for i in range(self._ncycles):
            trial_positions = positions + self._rng.normal(loc=0.0, scale=np.sqrt(dt)) + qforceOLD*dt*D
            trial_wf2 = self._wf.density(trial_positions, alpha)
            qforceNEW = self._wf.drift_force(trial_positions, alpha)
            Greens = 0.5*(qforceOLD+qforceNEW)*(positions-trial_positions+0.5*D*dt*(qforceOLD-qforceNEW))
            Greens = np.sum(Greens)
            Greens = np.exp(Greens)
            # Metropolis acceptance criterion
            if u[i] <= Greens*trial_wf2 / wf2:
                positions = trial_positions
                wf2 = trial_wf2
                n_accepted += 1
                qforceOLD = qforceNEW

            local_energy = self._wf.local_energy(positions, alpha)
            energy += local_energy
            energy2 += local_energy**2

        # acceptance rate
        acc_rate = n_accepted / self._ncycles
        print(f"{alpha=:.2f}, {acc_rate=:.2f}")
        # Calculate mean, variance, error
        energy /= self._ncycles
        energy2 /= self._ncycles
        variance = energy - energy2
        # error = np.sqrt(variance / self._ncycles)

        return energy, variance

    def _initial_positions(self):
        '''
        initial_state = self._rng.normal(
            loc=np.zeros((self._n_particles, self._dim)),
            scale=self._scale
        )
        '''
        initial_state = self._rng.random(size=(self._N, self._dim))
        #initial_state = np.zeros(shape=(self._N, self._dim))
        return initial_state

    def _safe_initialization(self, alpha):

        positions = self._initial_positions()
        wf2 = self._wf.density(positions, alpha)

        while wf2 <= 1e-14:
            positions /= 2
            wf2 = self._wf.density(positions, alpha)

        return positions

    def _draw_proposal_gaussian(self, old_positions):
        return self._rng.normal(loc=old_positions, scale=self._scale)

    def _draw_proposal_uniform(self, old_positions):
        return old_positions + (self._rng.random(size=(self._N, self._dim)) - 0.5)
