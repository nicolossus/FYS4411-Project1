import numpy as np
from src import vmc

N = 4       # Number of particles
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

alpha = 0.5
r = np.random.rand(N, dim) * 2.0

# Analytical
wf_a = vmc.ASHOIB(N, dim, omega)
# Numerical
wf_n = vmc.SHOIB(omega)

print("ASHOIB drift_force=", wf_a.drift_force(r, alpha))
#print("SHOIB drift_force=", wf_n.drift_force(r, alpha))

print("ASHOIB local_energy=", wf_a.local_energy(r, alpha))
print("SHOIB local_energy=", wf_n.local_energy(r, alpha))
