

class BaseJaxWF:

    def __init__(self):
        pass

    def precompute(self):
        self.grad_wf = grad(self.evaluate, arg=1)

    @abstractmethod
    def wavefunction(self, r, alpha):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self):
        raise NotImplementedError

    @abstractmethod
    def potential(self):
        raise NotImplementedError

    def local_energy(self):
        return (H * psi) / psi

    @partial(jit, static_argnums=(0,))
    def drift_force_jax(self, r, alpha):
        #grad_wf = grad(self.evaluate)
        F = (2 * self.grad_wf(r, alpha)) / self.evaluate(r, alpha)
        return F


class SG(BaseJaxWF):

    def __init__(self):
        pass

    def wf(self):
        return np.exp(r)

    def V(self):
        pass

    def slater(self):
        pass

    def jastrow(self):
        pass
