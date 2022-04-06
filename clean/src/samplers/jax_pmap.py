from jax import jit, lax, local_device_count, pmap, random, vmap


def do_mcmc(rng_key, n_vectorized=8):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        progress_bar=False,
        num_chains=n_vectorized,
        chain_method='vectorized'
    )
    mcmc.run(
        rng_key,
        extra_fields=("potential_energy",),
    )
    return {**mcmc.get_samples(), **mcmc.get_extra_fields()}


# Number of devices to pmap over
n_parallel = jax.local_device_count()
rng_keys = jax.random.split(PRNGKey(rng_seed), n_parallel)
traces = pmap(do_mcmc)(rng_keys)
# concatenate traces along pmap'ed axis
trace = {k: np.concatenate(v) for k, v in traces.items()}

'''
'''
partial_map_fn = partial(
    self._single_chain_mcmc,
    args=args,
    kwargs=kwargs,
    collect_fields=collect_fields,
)
map_args = (rng_key, init_state, init_params)
if self.num_chains == 1:
    states_flat, last_state = partial_map_fn(map_args)
    states = tree_map(lambda x: x[jnp.newaxis, ...], states_flat)
else:
    if self.chain_method == "sequential":
        states, last_state = _laxmap(partial_map_fn, map_args)
    elif self.chain_method == "parallel":
        states, last_state = pmap(partial_map_fn)(map_args)
