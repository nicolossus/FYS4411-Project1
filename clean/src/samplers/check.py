import numpy as np

seed = 42
rng = np.random.default_rng(seed=seed)

scale = 0.5
ncycles = 10
nchains = 3
positions = np.array([[0.2, 0.3], [0.1, 0.5]])
r = rng.normal(loc=positions, scale=scale)
# equivalent
# r = positions + rng.normal(loc=0, scale=1) * scale
# print(r)
# 0.15235853987721568
# 0.15235853987721568
alpha = 0.5
a = np.exp(-alpha * np.sum(r**2))
b = np.exp(-alpha * r**2)
print(a)
print(np.prod(b))

'''
[[ 0.25235854 -0.36999205]
 [ 0.4252256   0.72028236]]

[[ 0.35235854 -0.21999205]
 [ 0.4752256   0.97028236]]


[[ 0.35235854 -0.21999205]
 [ 0.4752256   0.97028236]]

 ---
 [[ 0.15235854 -0.51999205]
 [ 0.3752256   0.47028236]]
'''
