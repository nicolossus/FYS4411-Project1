import vmc

# Initialize a system; here we use the class for a spherical
# harmonic oscillator with non-interacting bosons (SHONIB)
system = vmc.SHONIB()

# Set MCMC method; either RWM or LMH. Both take the system
# object as constructor argument
rwm = vmc.RWM(system)

# Perform sampling
res = rwm.sample(
    nsamples,                # no. of energy samples to obtain
    initial_positions,       # initial spatial configuration
    alpha,                   # (initial) variational parameter
    nchains=1,               # no. of Markov chains
    seed=None,               # seed can be set for reproducibility
    tune=True,               # whether to tune scale parameters
    tune_iter=10000,         # maximum tuning cycles
    tune_interval=500,       # cycles between each tune update
    tol_tune=1e-5,           # tolerance level used to stop tune early
    optimize=True,           # whether to optimize alpha
    max_iter=50000,          # maximum optimize cycles
    batch_size=500,          # cycles in a batch used in optimization
    gradient_method='adam',  # specify optimization method
    eta=0.01,                # optimizer's learning rate
    tol_optim=1e-5,          # tolerance used to stop optimizer early
    early_stop=True,         # whether to use early stopping or not
    log=True,                # whether to show logger and progress bar
    logger_level="INFO",     # the level of logger
    **kwargs                 # kwargs depends on the MCMC method;
)                            # set scale for RWM and dt for LMH

# The results are returned in a pandas.DataFrame. They can
# be saved directly by using the sampler's `.to_csv` method
rwm.to_csv("filename.csv")
