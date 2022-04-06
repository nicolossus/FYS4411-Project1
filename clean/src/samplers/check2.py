import numpy as np


def density(r, alpha):
    return np.exp(-alpha * r**2)**2


def logp(r, alpha):
    return np.log(density(r, alpha))


rng = np.random.default_rng(seed=42)

# System
N = 2
dim = 3
alpha = 0.45
scale = 0.5

# Initialize
positions = rng.random(size=(N, dim)) * 1e4
'''
wf2 = density(positions, alpha)
while np.sum(wf2) <= 1e-14:
    positions *= 0.5
    wf2 = density(positions, alpha)
'''
# Metropolis step
logp_current = logp(positions, alpha)
log_unif = np.log(rng.random(size=positions.shape))
proposals = rng.normal(loc=positions, scale=scale)
logp_proposal = logp(proposals, alpha)
delp = logp_proposal - logp_current
print(np.isfinite(delp))
accept = log_unif < delp
# print(accept)
'''
delp = logp(y) - logp(x)
    if np.isfinite(delp) and np.log(np.random.uniform()) < delp:
'''
