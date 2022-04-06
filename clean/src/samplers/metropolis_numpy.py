import numpy as np


def metropolis_hastings_vec(log_prob, proposal_cov, iters, chains, init):
    """Vectorized Metropolis-Hastings.

    Allows pretty ridiculous scaling across chains:
    Runs 1,000 chains of 1,000 iterations each on a
    correlated 100D normal in ~5 seconds.
    """
    proposal_cov = np.atleast_2d(proposal_cov)
    dim = proposal_cov.shape[0]
    # Initialize with a single point, or an array of shape (chains, dim)
    if init.shape == (dim,):
        init = np.tile(init, (chains, 1))

    samples = np.empty((iters, chains, dim))
    samples[0] = init
    current_log_prob = log_prob(init)

    proposals = np.random.multivariate_normal(np.zeros(dim), proposal_cov,
                                              size=(iters - 1, chains))
    log_unifs = np.log(np.random.rand(iters - 1, chains))
    for idx, (sample, log_unif) in enumerate(zip(proposals, log_unifs), 1):
        proposal = sample + samples[idx - 1]
        proposal_log_prob = log_prob(proposal)
        accept = (log_unif < proposal_log_prob - current_log_prob)

        # copy previous row, update accepted indexes
        samples[idx] = samples[idx - 1]
        samples[idx][accept] = proposal[accept]

        # update log probability
        current_log_prob[accept] = proposal_log_prob[accept]
    return samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.stats as st

    dim = 10
    cov_mat = 0.1 * np.eye(dim) + 0.9 * np.ones((dim, dim))

    # Correlated Gaussian
    log_prob = st.multivariate_normal(np.zeros(dim),  cov_mat).logpdf

    proposal_cov = np.eye(dim)
    iters = 2_000
    chains = 1_024
    init = np.zeros(dim)

    samples = metropolis_hastings_vec(log_prob,
                                      proposal_cov,
                                      iters,
                                      chains,
                                      init)

    print(samples.shape)

    # plt.hist(samples)
    # plt.show()
