import numpy as np
from src import vmc


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

def exact_energy(N, dim, omega):
    return (omega * dim * N) / 2

N = 10     # Number of particles
dim = 3      # Dimensionality
omega = 1.   # Oscillator frequency

alpha = 0.5
r = np.random.rand(N, dim) * 2.0

# Analyticalhttps://msu.zoom.us/j/99553919750?pwd=NDFsYno2OVRxdU1iYUdHeThXTEp4UT09



wf_a = vmc.ASHOIB(N, dim, omega)
wf_e = vmc.AEHOIB(N, dim, omega, a=0)
wf_e_NI = vmc.AEHONIB(N, dim, omega)
wf_AI = vmc.AIB(N, dim, omega)
# Numerical
wf_n = vmc.SHOIB(omega)
#wf_AI.test_terms_in_lap(r, wf_AI.dudr_faster(r), alpha)
print("ASHONIB drift_force=", wf_a.drift_force(r, alpha))
print("SHOIB drift_force=", wf_n.drift_force(r, alpha))

print("ASHOIB local_energy=", wf_a.local_energy(r, alpha))
print("SHOIB local_energy=", wf_n.local_energy(r, alpha))
print("AIB local energy=", wf_AI.local_energy(r, alpha))

print("AEHOIB local energy=", wf_e.local_energy(r, alpha))
print("AEHONIB local energy=", wf_e_NI.local_energy(r, alpha))
