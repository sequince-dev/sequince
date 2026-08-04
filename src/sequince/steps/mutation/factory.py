from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...transforms import ParticleTransform
from ..base import SMCStep
from .emcee import EmceeMutation
from .kernels import get_mutation_kernel
from .metropolis import MetropolisHastingsMutation
from .minipcn import MiniPCNMutation

if TYPE_CHECKING:
    from ...target import SequentialTarget


def get_mutation_step(
    mutation: str | SMCStep | None,
    target: "SequentialTarget | Callable[..., Any]",
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
    elif mutation == "emcee":
        run_kwargs = kwargs.pop("run_kwargs", {})
        mutation_step = EmceeMutation(
            target,
            n_steps=mutation_n_steps,
            run_kwargs=run_kwargs,
            **kwargs,
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
    elif mutation == "none" or mutation is None:
        mutation_step = None
    else:
        raise ValueError(
            "mutation must be 'emcee', 'metropolis-hastings', 'minipcn', "
            "'none', or an SMCStep instance."
        )
    return mutation_step
