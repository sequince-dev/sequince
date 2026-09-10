from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...target import Target
from ...transforms import ParticleTransform
from ..base import SMCStep
from .blackjax import BlackJAXMutation
from .emcee import EmceeMutation
from .eryn import ErynMutation
from .kernels import get_complementary_ensemble_kernel, get_mutation_kernel
from .metropolis import (
    DifferentialEvolutionMutation,
    MetropolisHastingsMutation,
    RedBlueMetropolisHastingsMutation,
)
from .minipcn import MiniPCNMutation


def get_mutation_step(
    mutation: str | SMCStep | None,
    target: "Target | Callable[..., Any]",
    transform: ParticleTransform | None = None,
    mutation_n_steps: int = 10,
    mutation_kwargs: dict[str, Any] | None = None,
) -> SMCStep | None:
    """Construct a configured mutation step."""
    kwargs = dict(mutation_kwargs or {})
    if transform is not None:
        kwargs["transform"] = transform
    if isinstance(mutation, SMCStep):
        mutation_step = mutation
    elif mutation == "blackjax":
        mutation_step = BlackJAXMutation(
            target,
            n_steps=mutation_n_steps,
            **kwargs,
        )
    elif mutation == "emcee":
        run_kwargs = kwargs.pop("run_kwargs", {})
        mutation_step = EmceeMutation(
            target,
            n_steps=mutation_n_steps,
            run_kwargs=run_kwargs,
            **kwargs,
        )
    elif mutation == "eryn":
        mutation_step = ErynMutation(
            target, n_steps=mutation_n_steps, **kwargs
        )
    elif mutation == "minipcn":
        mutation_step = MiniPCNMutation(
            target,
            n_steps=mutation_n_steps,
            **kwargs,
        )
    elif mutation == "metropolis-hastings":
        try:
            kernel = kwargs.pop("kernel")
        except KeyError as exc:
            raise ValueError(
                "metropolis-hastings mutation requires a 'kernel' option."
            ) from exc
        kernel_kwargs = dict(kwargs.pop("kernel_kwargs", {}) or {})
        if isinstance(kernel, str):
            kernel = get_mutation_kernel(kernel, **kernel_kwargs)
        elif kernel_kwargs:
            raise ValueError(
                "kernel_kwargs can only be used with a named mutation kernel."
            )
        mutation_step = MetropolisHastingsMutation(
            target,
            kernel=kernel,
            n_steps=mutation_n_steps,
            **kwargs,
        )
    elif mutation in {
        "red-blue-metropolis-hastings",
        "red-blue-mh",
        "red-blue",
    }:
        try:
            kernel = kwargs.pop("kernel")
        except KeyError as exc:
            raise ValueError(
                "red-blue-metropolis-hastings mutation requires a 'kernel' option."
            ) from exc
        kernel_kwargs = dict(kwargs.pop("kernel_kwargs", {}) or {})
        if isinstance(kernel, str):
            kernel = get_complementary_ensemble_kernel(
                kernel,
                **kernel_kwargs,
            )
        elif kernel_kwargs:
            raise ValueError(
                "kernel_kwargs can only be used with a named mutation kernel."
            )
        mutation_step = RedBlueMetropolisHastingsMutation(
            target,
            kernel=kernel,
            n_steps=mutation_n_steps,
            **kwargs,
        )
    elif mutation == "differential-evolution":
        mutation_step = DifferentialEvolutionMutation(
            target,
            n_steps=mutation_n_steps,
            **kwargs,
        )
    elif mutation == "none" or mutation is None:
        mutation_step = None
    else:
        raise ValueError(
            "mutation must be 'blackjax', 'emcee', 'eryn', 'metropolis-hastings', "
            "'minipcn', 'differential-evolution', "
            "'red-blue-metropolis-hastings', 'red-blue-mh', 'red-blue', "
            "'none', or an SMCStep instance."
        )
    return mutation_step
