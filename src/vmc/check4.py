
from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpy.random import default_rng

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
# np.random.seed(42)

Array = Union[np.ndarray, jnp.ndarray]
PyTree = Union[Array, Iterable[Array], Mapping[Any, Array]]


class State(NamedTuple):
    positions: PyTree
    logp: Union[float, PyTree]
    n_accepted: int
    delta: int


def generate_seed_sequence(user_seed=None, pool_size=None):
    seed_sequence = np.random.SeedSequence(user_seed)
    seeds = seed_sequence.spawn(pool_size)
    return seeds


def advance_PRNG_state(seed, delta):
    return np.random.PCG64(seed).advance(delta)


class System:
    def __init__(self):
        pass

    @abstractmethod
    def wf(self, r, alpha):
        raise NotImplementedError

    @abstractmethod
    def potential(self):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, r, alpha):
        return jnp.exp(self.logprob(r, alpha))

    @partial(jax.jit, static_argnums=(0,))
    def logprob(self, r, alpha):
        # return 2. * self.wf(r, alpha)
        wf = self.wf(r, alpha)
        return wf * wf

    @partial(jax.jit, static_argnums=(0,))
    def _local_kinetic_energy(self, r, alpha):
        n = r.shape[0]
        eye = jnp.eye(n)

        grad_wf = jax.grad(self.wf, argnums=0)
        def grad_wf_closure(r): return grad_wf(r, alpha)
        primal, dgrad_f = jax.linearize(grad_wf_closure, r)

        _, diagonal = lax.scan(
            lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)

        return -0.5 * diagonal - 0.5 * primal**2

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, r, alpha):
        def ke_closure(r): return self._local_kinetic_energy(r, alpha)
        ke = jnp.sum(jax.vmap(ke_closure)(r))
        pe = self.potential(r)

        return ke + pe

    @partial(jax.jit, static_argnums=(0,))
    def drift_force(self, r, alpha):
        grad_wf = jax.grad(self.wf, argnums=0)
        F = 2 * grad_wf(r, alpha)
        return F

    @partial(jax.jit, static_argnums=(0,))
    def grad_alpha(self, r, alpha):
        grad_wf_alpha = jax.grad(self.wf, argnums=1)
        return grad_wf_alpha(r, alpha)


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
        return -alpha * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * jnp.sum(r * r)


class LogIB(System):

    def __init__(self, omega, a=0.00433):
        super().__init__()
        self._omega2 = omega * omega
        self._a = a

    @partial(jax.jit, static_argnums=(0,))
    def wf(self, r, alpha):
        return self._single(r, alpha) + self._correlation(r)

    @partial(jax.jit, static_argnums=(0,))
    def _single(self, r, alpha):
        return -alpha * jnp.sum(r * r)

    @partial(jax.jit, static_argnums=(0,))
    def _correlation(self, r):
        N = r.shape[0]
        i, j = jnp.triu_indices(N, 1)
        axis = r.ndim - 1
        rij = jnp.linalg.norm(r[i] - r[j], ord=2, axis=axis)
        f = 1 - self._a / rij * (rij > self._a)
        return jnp.sum(jnp.log(f))

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        return 0.5 * self._omega2 * jnp.sum(r * r)


class VMC:
    """

    Arguments
    ---------
    wavefunction : vmc_numpy.WaveFunction
        Wave function with methods dictated by vmc_numpy.WaveFunction
    inference_scheme : str
        Name of inference scheme. Can be used to condition for algorithm
        specific methods.
    rng : generator
        Random number generator. Default: numpy.random.default_rng
    """

    def __init__(self, wavefunction,  rng=None):

        self._wf = wavefunction

        if rng is None:
            rng = default_rng
        self._rng = rng

        # Retrieve callables
        # self._wf_scalar_fn = self._wf.wf_scalar
        self._wf_fn = self._wf.wf
        self._logp_fn = self._wf.logprob
        self._locE_fn = self._wf.local_energy
        self._driftF_fn = self._wf.drift_force
        self._grad_alpha_fn = self._wf.grad_alpha

    @abstractmethod
    def step(self, state, alpha, **kwargs):
        raise NotImplementedError

    def initial_state(self, initial_positions, alpha):
        state = State(initial_positions,
                      self._logp_fn(initial_positions, alpha),
                      0,
                      0
                      )
        return state

    def sample(
        self,
        nsamples,
        initial_positions,
        alpha,
        nchains=1,
        seed=None,
        **kwargs
    ):
        """Sampling procedure"""

        seeds = generate_seed_sequence(seed, nchains)
        state = self.initial_state(initial_positions, alpha)

        energies = np.zeros(nsamples)

        for i in range(nsamples):
            state = self.step(state, alpha, seed, **kwargs)
            energies[i] = self._locE_fn(state.positions, alpha)

        return energies, state


class Metropolis(VMC):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction, rng=rng)

    def step(self, state, alpha, seed, scale=0.03):
        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)
        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=scale)
        # Sample log uniform rvs
        #log_unif = np.log(rng.random(size=state.positions.shape))
        log_unif = np.log(rng.random())
        # Compute proposal log density
        logp_proposal = self._logp_fn(proposals, alpha)
        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp
        # Where accept is True, yield proposal, otherwise keep old state
        new_positions = np.where(accept, proposals, state.positions)
        #new_logp = np.where(accept, logp_proposal, state.logp)
        new_logp = self._logp_fn(new_positions, alpha)
        new_n_accepted = state.n_accepted + np.sum(accept)
        new_delta = state.delta + 1
        # Create new state
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state


class MetropolisHastings(VMC):

    def __init__(self, wavefunction, rng=None):
        super().__init__(wavefunction, rng=rng)

    def step(self, state, alpha, seed, D=0.5, dt=1.0):
        """One step of the ...

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        D : float
            Diffusion constant. Default: 0.5
        dt : float
            Scale of proposal distribution. Default: 1.0
        """

        # Precompute
        Ddt = D * dt
        quarterDdt = 1 / (4 * Ddt)
        sys_size = state.positions.shape
        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)
        # Compute drift force at current positions
        F = self._driftF_fn(state.positions, alpha)
        # Sample proposal positions, i.e., move walkers
        proposals = state.positions + F * Ddt + \
            rng.normal(loc=0, scale=np.sqrt(dt), size=sys_size)
        # Compute proposal log density
        logp_prop = self._logp_fn(proposals, alpha)
        # Green's function conditioned on proposals
        F_prop = self._driftF_fn(proposals, alpha)
        G_prop = -(state.positions - proposals -
                   Ddt * F_prop)**2 * quarterDdt
        # Green's function conditioned on current positions
        G_cur = -(proposals - state.positions - Ddt * F)**2 * quarterDdt
        # Metroplis-Hastings ratio
        #ratio = logp_prop + G_prop - state.logp - G_cur
        ratio = logp_prop + np.sum(G_prop) - state.logp - np.sum(G_cur)
        # Sample log uniform rvs
        #log_unif = np.log(rng.random(size=sys_size))
        log_unif = np.log(rng.random())
        # Metroplis acceptance criterion
        accept = log_unif < ratio
        # Where accept is True, yield proposal, otherwise keep old state
        new_positions = np.where(accept, proposals, state.positions)
        #new_logp = np.where(accept, logp_prop, state.logp)
        new_logp = self._logp_fn(new_positions, alpha)
        new_n_accepted = state.n_accepted + np.sum(accept)
        new_delta = state.delta + 1
        # Create new state
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state


if __name__ == "__main__":

    def exact_energy(N, dim, omega):
        return (omega * dim * N) / 2

    def interacting_initial_positions(wf, alpha, N, dim, a=0.00433, max_count=50):

        def corr_factor(r1, r2):
            rij = np.linalg.norm(r1 - r2)
            if rij <= a:
                return 0.
            else:
                return 1 - (a / rij)

        std = 2

        r = np.random.randn(N, dim) * std
        N = r.shape[0]

        counter = 0
        corr = 0.
        rerun = True
        while rerun:
            rerun = False
            for i in range(N):
                for j in range(i + 1, N):
                    corr = corr_factor(r[i, :], r[j, :])
                    if corr == 0.:
                        print("corr=0 encountered")
                        rerun = True
                        r[i, :] = np.random.randn() * std
                        r[j, :] = np.random.randn() * std
            std *= 1.5

        return r

    N = 100        # Number of particles
    dim = 3        # Dimensionality
    omega = 1.     # Oscillator frequency
    a = 0.00433
    alpha = 0.5

    #wf = LogNIB(omega)
    wf = LogIB(omega)
    sampler = Metropolis(wf)
    #sampler = MetropolisHastings(wf)
    r = interacting_initial_positions(wf, alpha, N, dim)

    print(wf.logprob(r, alpha))

    # r = np.random.rand(N, dim) * 3.0
    # r = safe_initial_positions(wf, alpha, N, dim)
    nsamples = 10000

    energies, state = sampler.sample(nsamples, r, alpha,)
    print(np.mean(energies[1000:]))
    print("exact=", exact_energy(N, dim, omega))
    print("accepted:", state.n_accepted)
