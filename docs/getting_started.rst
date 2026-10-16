Getting started
===============

Installation
------------

Install from PyPI::

    pip install sequince

``sequince`` targets Python 3.11+. Its core dependencies are installed
automatically; integrations such as emcee, BlackJAX, and HDF5 checkpointing are
available as optional extras.

Quick start
-----------

Below, we show a short example of how to use `sequince`.
For more details, see the [user guide](https://sequince.readthedocs.io/user_guide), and
see [`sequince-examples`](https://github.com/sequince-dev/sequince-examples) for more advanced usage.
Here, we fit a straight line, `y = slope * x + intercept`, to synthetic data with
independent Gaussian measurement errors and uniform priors on both parameters.
The log-prior and log-likelihood callbacks accept particles shaped
`(n_particles, n_parameters)` and return one log density per particle.

```python
import numpy as np
from sequince import ParticleState, make_standard_bayesian_smc


def log_prior(theta, context):
    """Uniform log-prior"""
    # log-prior and -likelihood can optionally accept context to access shared data.
    lower, upper = context.shared["bounds"]
    inside = np.all((theta >= lower) & (theta <= upper), axis=-1)
    log_density = -theta.shape[-1] * np.log(upper - lower)
    return np.where(inside, log_density, -np.inf)


def log_likelihood(theta, context):
    """Gaussian log-likelihood"""
    x = context.shared["x"]
    y = context.shared["y"]
    sigma = context.shared["sigma"]
    slope = theta[..., 0, None]
    intercept = theta[..., 1, None]
    residual = (y - (slope * x + intercept)) / sigma
    return -0.5 * np.sum(
        residual**2 + np.log(2 * np.pi * sigma**2), axis=-1
    )


# Construct the runner
# `smc` is the main runner
# `posterior` contains definitions of the tempered posterior
# `metadata` contains metadata passed between the SMC steps
smc, posterior, metadata = make_standard_bayesian_smc(
    log_prior,
    log_likelihood,
)

rng = np.random.default_rng(42)

# Generate observations with true slope 2 and intercept 1.
x = np.linspace(-1, 1, 20)
sigma = 0.5
y = 2 * x + 1 + rng.normal(scale=sigma, size=x.shape)

bounds = [-5, 5]

# Draw initial (slope, intercept) particles from the prior.
n = 1000
initial_particles = rng.uniform(*bounds, size=(n, 2))
# Construct the initial state
initial_state = ParticleState(particles=initial_particles)

# Construct the shared context dictionary that will be exposed to the log_prior
# and log_likelihood
shared_context = dict(bounds=bounds, x=x, y=y, sigma=sigma)

# Run the SMC
results = smc.run(
    initial_state,
    rng=rng,
    shared_context=shared_context,
    metadata=metadata,
)

# Obtain equal-weight posterior samples and their weights
samples, sample_weights = results.posterior_samples(
    rng=rng, method="rejection"
)

# Compare the posterior means with the true slope 2 and intercept 1.
mean_slope, mean_intercept = np.average(
    samples, weights=sample_weights, axis=0
)
print(f"Posterior mean slope: {mean_slope:.2f}")
print(f"Posterior mean intercept: {mean_intercept:.2f}")
```

Next steps
----------

See :doc:`evidence_uncertainty` for interpreting the evidence estimate and its
uncertainty. To customise the run, continue with :doc:`smc_configuration` and
the sampling choices in :doc:`resampling_options` and :doc:`mutation_options`.
The :doc:`step_context` guide explains the distinction between shared inputs
and evolving metadata. For sampling without SMC weighting and resampling, see
:doc:`mcmc_sampler`.
