"""Blocked Metropolis-Hastings mutation kernels and steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..._typing import Array
from ...transforms import ParticleTransform
from ..base import SMCStep
from ..context import StepContext
from .base import TransformRefit
from .kernels import MutationKernel
from .metropolis import MetropolisHastingsMutation, MutationKernelFactory

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...posterior import TemperedPosterior
    from ...state import ParticleState


@dataclass(frozen=True)
class MetropolisBlock:
    """Configuration for one Metropolis-within-Gibbs parameter block."""

    name: str
    kernel: MutationKernel | MutationKernelFactory
    n_steps: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MetropolisBlock.name must not be empty.")
        if self.n_steps < 1:
            raise ValueError("MetropolisBlock.n_steps must be >= 1.")


class BlockedMetropolisHastingsMutation(SMCStep):
    """Apply sequential Metropolis-Hastings updates to parameter blocks.

    Each block targets the complete log probability while its kernel changes
    only a subset of coordinates. The composition is therefore a valid
    Metropolis-within-Gibbs rejuvenation step.

    Coordinate indices refer to the space in which the kernels operate. When a
    transform is supplied, it should preserve the requested block structure
    (for example, a coordinate-wise bounds transform rather than a dense
    whitening transform).

    Example
    -------
    >>> from sequince import steps, transforms
    >>> from sequince.steps.mutation import kernels
    >>> blocks = [
    ...     steps.mutation.MetropolisBlock(
    ...         name="block1",
    ...         kernel=kernels.AdaptiveMaskedGaussianRandomWalk(
    ...             indices=[0, 1],
    ...             scale=0.1,
    ...         ),
    ...         n_steps=2,
    ...     ),
    ...     steps.mutation.MetropolisBlock(
    ...         name="block2",
    ...         kernel=kernels.AdaptiveMaskedGaussianRandomWalk(
    ...             indices=[2, 3],
    ...             scale=0.2,
    ...         ),
    ...         n_steps=3,
    ...     ),
    ... ]
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "TemperedPosterior",
        blocks: tuple[MetropolisBlock, ...] | list[MetropolisBlock],
        *,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
    ) -> None:
        super().__init__()
        self._blocks = tuple(blocks)
        if not self._blocks:
            raise ValueError(
                "BlockedMetropolisHastingsMutation requires at least one block."
            )
        names = [block.name for block in self._blocks]
        if len(set(names)) != len(names):
            raise ValueError("Metropolis block names must be unique.")
        self._steps = tuple(
            MetropolisHastingsMutation(
                log_prob_fn,
                block.kernel,
                n_steps=block.n_steps,
                transform=transform,
                transform_refit=transform_refit,
                store_chains=store_chains,
            )
            for block in self._blocks
        )
        self._store_chains = store_chains

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> ParticleState:
        updated = state
        acceptance: dict[str, float] = {}
        acceptance_by_step: dict[str, list[float]] = {}
        chains: dict[str, Array] = {}

        for block, step in zip(self._blocks, self._steps, strict=True):
            updated = step(updated, context, rng)
            acceptance[block.name] = float(
                updated.metadata["metropolis_hastings_acceptance"]
            )
            acceptance_by_step[block.name] = list(
                updated.metadata["metropolis_hastings_acceptance_by_step"]
            )
            if self._store_chains:
                chains[block.name] = updated.metadata["mutation_chains"]

        metadata = dict(updated.metadata)
        metadata["blocked_metropolis_hastings_acceptance"] = acceptance
        metadata["blocked_metropolis_hastings_acceptance_by_step"] = (
            acceptance_by_step
        )
        metadata["blocked_metropolis_hastings_steps"] = {
            block.name: block.n_steps for block in self._blocks
        }
        if self._store_chains:
            metadata["blocked_metropolis_hastings_chains"] = chains

        context.metadata["blocked_metropolis_hastings_acceptance"] = acceptance
        context.metadata["blocked_metropolis_hastings_acceptance_by_step"] = (
            acceptance_by_step
        )
        return updated.with_updates(metadata=metadata)
