"""Compare fixed, length-adaptive and jointly adaptive mutation on N(0, I).

Run: pixi run --frozen python examples/mixture_mutation.py
Each mode uses 8,000 particles, eight outer iterations, and seed 731.
"""

from __future__ import annotations

import numpy as np
from orng import RandomGenerator

from sequince._typing import Array
from sequince.state import ParticleState
from sequince.steps import (
    AdaptMoveWeights,
    AdaptMutationSteps,
    JumpDistancePerEvaluation,
    MixtureMutation,
    StepContext,
)
from sequince.steps.base import SMCStep
from sequince.steps.mutation import (
    CallableMutationKernel,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)


def log_target(x: Array) -> Array:
    """Return standard normal log density, omitting its normalizer."""
    return -0.5 * np.sum(x**2, axis=-1)


def make_mutation(mode: str) -> SMCStep:
    """Build a fresh mutation with symmetric and asymmetric proposals."""
    independent = CallableMutationKernel(
        lambda x, rng: rng.normal(size=x.shape) + 0.7,
        log_transition_ratio_fn=lambda x, y: -0.5
        * np.sum((x - 0.7) ** 2 - (y - 0.7) ** 2, axis=-1),
    )
    mixture = MixtureMutation(
        {
            "walk": MetropolisHastingsMutation(
                log_target, GaussianRandomWalk(scale=0.4)
            ),
            "independent": MetropolisHastingsMutation(log_target, independent),
        },
        n_steps=8,
    )
    child: MixtureMutation | AdaptMoveWeights = mixture
    if mode == "joint":
        child = AdaptMoveWeights(mixture, JumpDistancePerEvaluation())
    if mode != "fixed":
        return AdaptMutationSteps(
            child,
            AcceptanceRatePolicy(4, min_steps=2, max_steps=20),
            initial_steps=8,
            min_steps=2,
            max_steps=20,
        )
    return child


def main() -> None:
    """Report accuracy and actual evaluation work for each mode."""
    print(
        "mode       mean       variance   target evals   final probabilities"
    )
    for mode in ("fixed", "length", "joint"):
        mutation = make_mutation(mode)
        state = ParticleState(
            particles=np.full((8000, 1), 4.0), log_weights=np.zeros(8000)
        )
        rng = RandomGenerator(seed=731)
        cost = 0
        for iteration in range(8):
            state, _, output = mutation(state, StepContext(iteration), rng)
            cost += int(output.evaluation_counts.log_target_evaluations)
        print(
            f"{mode:8s} {np.mean(state.particles): .5f}    "
            f"{np.var(state.particles):.5f}    {cost:10d}     "
            f"{tuple(round(p, 3) for p in output.weights)}"
        )


if __name__ == "__main__":
    main()
