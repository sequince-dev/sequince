"""Run in a fresh process with BILBY_ARRAY_API=1 SCIPY_ARRAY_API=1.

Requires the Bilby v3 development checkout, Torch, and sequince[pyro].
The likelihood can be replaced without changing the mutation API.
"""

import torch
from array_api_compat import is_torch_namespace
from bilby import Likelihood
from bilby.compat.utils import array_module
from bilby.core.prior import Gaussian, PriorDict

from sequince import ParticleState, TemperedPosterior
from sequince.steps import PyroNUTSMutation, StepContext


class TorchLikelihood(Likelihood):
    """Cheap differentiable model with explicit Bilby parameter dictionaries."""

    def log_likelihood(
        self, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # A differentiable conversion can derive waveform parameters here.
        prediction = 2 * parameters["x"]
        return -0.5 * (prediction - 1).square()


def make_target() -> TemperedPosterior:
    """Connect physical particle coordinates using existing target callbacks."""
    priors = PriorDict({"x": Gaussian(mu=0, sigma=2)})
    likelihood = TorchLikelihood()

    def parameters(x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not is_torch_namespace(array_module(x)):
            raise TypeError(
                "Bilby Torch backend required: set BILBY_ARRAY_API=1 and "
                "SCIPY_ARRAY_API=1 before importing Bilby."
            )
        return {"x": x[..., 0]}

    return TemperedPosterior(
        log_prior=lambda x: priors.ln_prob(parameters(x), axis=0),
        log_likelihood=lambda x: likelihood.log_likelihood(parameters(x)),
    )


def main() -> None:
    target = make_target()
    state = ParticleState(
        torch.linspace(-1, 1, 8, dtype=torch.float64)[:, None]
    )
    context = StepContext(t=0, metadata={"beta": 0.5})
    mutation = PyroNUTSMutation(target, n_steps=2, step_size=0.2)
    state, _, diagnostics = mutation.apply(state, context, rng=42)
    print(state.particles)
    print(diagnostics)


if __name__ == "__main__":
    main()
