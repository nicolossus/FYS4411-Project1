import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large',
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('text', usetex=True)

rng = default_rng()

N = 50
data = []
data2 = []
for _ in range(20):
    data.append(rng.uniform(0, 1, N))
    data2.append(rng.normal(0, 1, N))

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

mean2 = np.mean(data2, axis=0)
std2 = np.std(data2, axis=0)

x = range(N)

fig, axes = plt.subplots(nrows=2,
                         ncols=2,
                         figsize=(6, 4),
                         tight_layout=True,
                         dpi=120,
                         sharex=True,
                         sharey=True)

# Blue
# Fill between
axes[0, 0].plot(mean, color="C0")
axes[0, 0].fill_between(range(N),
                        mean - std, mean + std,
                        alpha=0.5,
                        facecolor='lightblue')
# Errorbar
axes[0, 1].errorbar(x,
                    mean,
                    yerr=std,
                    fmt=':o',
                    color='C0',
                    ecolor='k',
                    markersize=3.5,
                    lw=1.0,
                    elinewidth=1.5,
                    capsize=3,
                    )

# Orange
# Fill between
axes[1, 0].plot(mean2, color="C1")
axes[1, 0].fill_between(x,
                        mean2 - std2, mean2 + std2,
                        alpha=0.5,
                        facecolor='wheat')
# Errorbar
axes[1, 1].errorbar(x,
                    mean,
                    yerr=std,
                    fmt=':o',
                    color='C1',
                    ecolor='k',
                    markersize=3.5,
                    lw=1.0,
                    elinewidth=1.5,
                    capsize=3,
                    )

# Set axis labels
axes[0, 0].set(ylabel="Data")
axes[1, 0].set(xlabel="Iterations", ylabel="Data")
axes[1, 1].set(xlabel="Iterations")

plt.show()
