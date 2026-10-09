# sequince

Sequince is module Sequential Monte Carlo (SMC) library for Bayesian inference,
particle filtering, and related sequential methods.

## Features

- Compose SMC pipelines from reusable weighting, resampling, mutation, and control steps.
- Prebuilt pipelines with `make_standard_bayesian_smc` and other pre-built recipes.
- Analyse runs via `SMCResults`, including posterior resampling and diagnostics helpers.

## Installation

`sequince` targets Python 3.11+ and can be installed directly from PyPI:

```bash
pip install sequince
```

## Quick start

```python
import numpy as np
from sequince import ParticleState, make_standard_bayesian_smc


def log_prior(theta):
    ...


def log_likelihood(theta):
    ...


smc, posterior, shared_context = make_standard_bayesian_smc(
    log_prior,
    log_likelihood,
)

n = 2_048
initial_particles = np.random.normal(size=(n, 2))
initial_state = ParticleState(particles=initial_particles)

rng = np.random.default_rng()
results = smc.run(initial_state, rng=rng, shared_context=shared_context)
samples = results.posterior_samples(1_000, rng=rng)
```

See the [user guide](docs/user_guide.rst) for details on composing custom
pipelines and implementing additional steps.

## Documentation

Full documentation is maintained in the `docs/` directory and published to Read
the Docs: <https://sequince.readthedocs.io>

## Examples

The [`sequince-examples`](https://github.com/sequince-dev/sequince-examples)
repository and corresponding documentation contains various examples highlight
different features in `sequince`.

## Contributing

Bug reports, feature requests, and pull requests are welcome via the GitHub
repository. Please see the [contributing guidelines](https://github.com/sequince-dev/sequince/blob/main/CONTRIBUTING.md)
and [AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).


## License

Licensed under the [MIT License](LICENSE).
