
# indices to calculate values in the upper triangle matrix without the diagonal
ind = np.triu_indices(len(spiketrains), 1)
sttcs = np.array([spike_time_tiling_coefficient(spiketrains[i],
                                                spiketrains[j],
                                                dt=bin_size)
                 for i, j in zip(*ind)]
                 )
