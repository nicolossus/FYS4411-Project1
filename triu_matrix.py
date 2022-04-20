import numpy as np
"""
# indices to calculate values in the upper triangle matrix without the diagonal
ind = np.triu_indices(len(spiketrains), 1)
sttcs = np.array([spike_time_tiling_coefficient(spiketrains[i],
                                                spiketrains[j],
                                                dt=bin_size)
                 for i, j in zip(*ind)]
                 )

#
"""

N = 5  # Number of particles
dim = 3
a = 0.5



i, j = np.triu_indices(N, 1)
axis = r.ndim - 1
q = r[i] - r[j]
rij = jnp.linalg.norm(q, ord=2, axis=axis)
f = 1 - a / rij * (rij > a)

print(f)
