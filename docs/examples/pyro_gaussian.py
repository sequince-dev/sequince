"""A minimal tempered Gaussian mutation; run with sequince[pyro]."""

import torch

from sequince import ParticleState, TemperedPosterior
from sequince.steps import PyroNUTSMutation, StepContext


def main() -> None:
    target = TemperedPosterior(
        log_prior=lambda x: -0.5 * x.square().sum(-1),
        log_likelihood=lambda x: -0.5 * (x - 1).square().sum(-1),
    )
    state = ParticleState(
        torch.linspace(-1, 1, 8, dtype=torch.float64)[:, None]
    )
    context = StepContext(t=0, metadata={"beta": 0.5})
    mutation = PyroNUTSMutation(target, n_steps=2, step_size=0.2)
    state, context, diagnostics = mutation.apply(state, context, rng=42)
    print(state.particles)
    print(diagnostics)


if __name__ == "__main__":
    main()
