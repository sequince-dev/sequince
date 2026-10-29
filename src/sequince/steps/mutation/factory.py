from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import Any

from ...target import Target
from ...transforms import ParticleTransform
from ..base import SMCStep
from .adaptive.policies.acceptance import AcceptanceRatePolicy
from .adaptive.steps import AdaptMutationSteps
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


def _acceptance_policy(
    config: Mapping[str, object] | None, initial_steps: int
) -> AcceptanceRatePolicy | None:
    """Validate configuration even when adaptation is disabled."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("adaptive_mutation must be a mapping or None.")
    unknown = config.keys() - {
        "enabled",
        "min_steps",
        "max_steps",
        "target_accepted_moves",
    }
    if unknown:
        raise ValueError(f"Unknown adaptive_mutation keys: {list(unknown)!r}")
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("adaptive_mutation enabled must be a boolean.")
    bounds = []
    for name, default in (("min_steps", 0), ("max_steps", 1000)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"adaptive_mutation {name} must be an integer.")
        if value < 0:
            raise ValueError(f"adaptive_mutation {name} must be >= 0.")
        bounds.append(int(value))
    minimum, maximum = bounds
    if maximum < minimum:
        raise ValueError("adaptive_mutation max_steps must be >= min_steps.")
    target = config.get("target_accepted_moves", 5.0)
    if isinstance(target, bool) or not isinstance(target, Real):
        raise TypeError("target_accepted_moves must be a real number.")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_accepted_moves must be finite and positive.")
    if not enabled:
        return None
    if isinstance(initial_steps, bool) or not isinstance(
        initial_steps, Integral
    ):
        raise TypeError("mutation_n_steps must be an integer.")
    if not minimum <= initial_steps <= maximum:
        raise ValueError(
            "mutation_n_steps must be within [min_steps, max_steps]."
        )
    return AcceptanceRatePolicy(
        float(target), min_steps=minimum, max_steps=maximum
    )


def get_mutation_step(
    mutation: str | SMCStep | None,
    target: "Target | Callable[..., Any]",
    transform: ParticleTransform | None = None,
    mutation_n_steps: int = 10,
    mutation_kwargs: dict[str, Any] | None = None,
    adaptive_mutation: Mapping[str, object] | None = None,
) -> SMCStep | None:
    """Construct a configured mutation step.

    Parameters
    ----------
    mutation : str, SMCStep or None
        Named mutation, custom step, or disabled mutation.
    target : Target or callable
        Target passed to the mutation constructor.
    transform : ParticleTransform or None, optional
        Transform passed to the mutation constructor.
    mutation_n_steps : int, optional
        Fixed total budget, or initial budget when adaptation is enabled.
    mutation_kwargs : dict or None, optional
        Additional mutation constructor options.
    adaptive_mutation : Mapping[str, object] or None, optional
        Named miniPCN and differential-evolution mutations support ``enabled``
        (default True), ``min_steps`` (0), ``max_steps`` (1000), and
        ``target_accepted_moves`` (5.0). Omission disables adaptation.
        Completed acceptance metrics select the next iteration's bounded
        budget; the current block is frozen before execution. Accepted moves
        need not be independent samples. Proposal parameters are unchanged.

    Returns
    -------
    SMCStep or None
        Configured mutation, optionally wrapped in ``AdaptMutationSteps``.
    """
    policy = _acceptance_policy(adaptive_mutation, mutation_n_steps)
    if policy is not None and (
        not isinstance(mutation, str)
        or mutation not in {"minipcn", "differential-evolution"}
    ):
        raise ValueError(
            "adaptive_mutation requires named minipcn or "
            "differential-evolution mutation."
        )
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
    if policy is not None:
        assert isinstance(
            mutation_step, (MiniPCNMutation, DifferentialEvolutionMutation)
        )
        return AdaptMutationSteps(
            mutation_step,
            policy,
            initial_steps=mutation_n_steps,
            min_steps=policy.min_steps,
            max_steps=policy.max_steps,
        )
    return mutation_step
