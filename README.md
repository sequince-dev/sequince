# sequince

Modular Sequential Monte Carlo (SMC) building blocks for Bayesian inference,
particle filtering, and related sequential methods.

## Features

- Compose SMC pipelines from reusable weighting, resampling, mutation, and control steps.
- Bootstrap workflows quickly with `make_standard_bayesian_smc` and other pre-built recipes.
- Analyse runs via `SMCResults`, including posterior resampling and diagnostics helpers.

## Installation

`sequince` targets Python 3.10+ and can be installed directly from the main branch:

```bash
pip install git+https://github.com/sequince-dev/sequince.git
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
initial_state = ParticleState(
    particles=initial_particles,
    log_weights=np.full(n, -np.log(n)),
    metadata={"temperature": 0.0},
)

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

- `examples/adaptive_schedule.py` – Adaptive tempering starting from the prior distribution.
- `examples/adaptive_schedule_proposal.py` – Tempering with an informative proposal distribution.
- `examples/gmm_transdimensional.py` – Adaptive annealing with on-the-fly component expansion for a Gaussian mixture model.

Run a script with `python examples/<name>.py` inside an environment where
`sequince` and its dependencies are installed.

## Contributing

Bug reports, feature requests, and pull requests are welcome via the GitHub
repository. Run the test suite with:

```bash
pytest
```

## License

Licensed under the [MIT License](LICENSE).
