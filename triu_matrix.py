import numpy as np
# indices to calculate values in the upper triangle matrix without the diagonal
ind = np.triu_indices(len(spiketrains), 1)
sttcs = np.array([spike_time_tiling_coefficient(spiketrains[i],
                                                spiketrains[j],
                                                dt=bin_size)
                 for i, j in zip(*ind)]
                 )

#


N = 5  # Number of particles
dim = 3
a = 0.5

f = 1.0
# interaction loop
for i in range(N):
    for j in range(i, N - 1):
        # for j in range(i, N):

        # pairwise distances
        dist_tmp = 0.0
        for k in range(dim):
            rij = np.abs(pos[i, k] - pos[j, k])
            # rij = np.abs(pos[i, k] - pos[j+1, k]) # ?
            dist_tmp += rij**2

        dist = np.sqrt(dist_tmp)

        if dist > a:
            f *= (1 - a / dist)
        else:
            f *= 0.0
