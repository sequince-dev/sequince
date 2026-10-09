# sequince

Sequince is a modular Sequential Monte Carlo (SMC) library for Bayesian inference,
particle filtering, and related sequential methods.

## Installation

`sequince` targets Python 3.11+ and can be installed directly from PyPI:

```bash
pip install sequince
```

## Features

- Array API support for NumPy, JAX, PyTorch, and CuPy backends.
- Compose SMC pipelines from reusable weighting, resampling, mutation, and control steps.
- Prebuilt pipelines with `make_standard_bayesian_smc` and other pre-built recipes.
- Analyse runs via `SMCResult`, including posterior resampling and diagnostics helpers.
- Estimate Bayesian evidence alongside posterior sampling.
- Save checkpoints and resume sampling runs.

## Quick start

Fit a straight line, `y = slope * x + intercept`, to synthetic data with
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

See the [documentation](<https://sequince.readthedocs.io>) for details on the
different components (shared context, metadata, results), composing custom pipelines and
implementing additional steps.

## Documentation

Full documentation is maintained in the `docs/` directory and published to Read
the Docs: <https://sequince.readthedocs.io>

## Examples

The [`sequince-examples`](https://github.com/sequince-dev/sequince-examples)
repository and corresponding documentation contain examples highlighting
different features in `sequince`.

## Contributing

Bug reports, feature requests, and pull requests are welcome via the GitHub
repository. Please see the [contributing guidelines](https://github.com/sequince-dev/sequince/blob/main/CONTRIBUTING.md)
and [AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).


## Related repositories

The following repositories are part of the `sequince` ecosystem:

- [`sequince-density`](https://github.com/sequince-dev/sequince-density): provides density estimators for `sequince`, supporting preconditioned SMC and ASPIRE-style analyses
- [`sequince-bilby`](https://github.com/sequince-dev/sequince-bilby): provides an interface for using prebuilt `sequince` samplers with `bilby`
- [`orng`](https://github.com/sequince-dev/orng): provides cross-backend random number generation
- [`acai`](https://github.com/sequince-dev/acai): provides array-API compatible control flow and iteration


## License

Licensed under the [MIT License](LICENSE).
