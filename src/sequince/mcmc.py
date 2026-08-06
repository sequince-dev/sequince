from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ._typing import Array
from .array_api import asarray, namespace
from .checkpoint import CheckpointWriter, SMCCheckpoint
from .result import SMCResult
from .smc import SequentialMonteCarlo, SMCRuntime
from .state import ParticleState
from .steps.base import StepContext
from .steps.mutation import MutationStep, get_mutation_step
from .transforms import ParticleTransform

if TYPE_CHECKING:
    import numpy as np
    from orng import RandomGenerator


@dataclass(frozen=True)
class MCMCResult:
    """Result of applying one or more checkpointable MCMC mutation blocks."""

    states: Sequence[ParticleState]
    context: StepContext

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("MCMCResult requires at least one ParticleState.")
        if not any(
            "mutation_chains" in state.metadata for state in self.states
        ):
            raise ValueError(
                "MCMCResult requires mutation steps configured with "
                "store_chains=True."
            )

    @classmethod
    def from_smc_result(cls, result: SMCResult) -> MCMCResult:
        """Build an MCMC result from a mutation-only sequential run."""
        return cls(states=result.states, context=result.context)

    @property
    def final_state(self) -> ParticleState:
        return self.states[-1]

    @property
    def chain(self) -> Array:
        """Return mutation blocks concatenated as ``(steps, chains, dims)``."""
        chains: list[Array] = []
        for state in self.states:
            chain = state.metadata.get("mutation_chains")
            if isinstance(chain, Mapping):
                chains.extend(chain.values())
            elif chain is not None:
                chains.append(chain)
        xp = namespace(*chains)
        arrays = [asarray(chain, xp=xp, like=chains[0]) for chain in chains]
        return xp.concat(arrays, axis=0)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostics recorded on the final mutation state."""
        return {
            key: value
            for key, value in self.final_state.metadata.items()
            if key.startswith(
                ("blocked_", "emcee_", "minipcn_", "metropolis_")
            )
            and key != "mutation_chains"
        }

    def samples(
        self,
        *,
        burn_in: int = 0,
        thin: int = 1,
        flat: bool = True,
    ) -> Array:
        """Return the chain after burn-in and thinning."""
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if thin < 1:
            raise ValueError("thin must be at least one.")
        chain = self.chain[burn_in::thin]
        if not flat:
            return chain
        xp = namespace(chain)
        return xp.reshape(chain, (-1, chain.shape[-1]))


class MarkovChainMonteCarlo:
    """Checkpointable standalone MCMC built from a Sequince mutation step."""

    def __init__(
        self,
        mutation: MutationStep,
        *,
        callbacks: Sequence[Callable[..., None]] | None = None,
        checkpoint_writers: Sequence[CheckpointWriter] | None = None,
    ) -> None:
        if not isinstance(mutation, MutationStep):
            raise TypeError("mutation must be a MutationStep.")
        mutation.store_chains = True
        self.mutation = mutation
        self._runner = SequentialMonteCarlo(
            steps=[mutation],
            callbacks=callbacks,
            checkpoint_writers=checkpoint_writers,
            auto_normalize=False,
        )

    def run(
        self,
        initial_state: ParticleState | None = None,
        *,
        rng: RandomGenerator | np.random.Generator | int | None = None,
        shared_context: Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        runtime: SMCRuntime | None = None,
    ) -> MCMCResult:
        """Run one configured MCMC block."""
        result = self._runner.run(
            initial_state,
            num_steps=1,
            rng=rng,
            shared_context=shared_context,
            metadata=metadata,
            outputs=outputs,
            runtime=runtime,
        )
        return MCMCResult.from_smc_result(result)

    def resume_from(
        self,
        checkpoint: SMCCheckpoint | Mapping[str, Any],
        *,
        rng: RandomGenerator | np.random.Generator | int | None = None,
        shared_context: Mapping[str, Any] | None = None,
    ) -> SMCRuntime:
        """Restore a checkpoint for continuation with :meth:`run`."""
        return self._runner.resume_from(
            checkpoint,
            rng=rng,
            shared_context=shared_context,
        )


def make_mcmc(
    log_prob_fn: Callable[..., Array | float],
    *,
    mutation: str | MutationStep = "minipcn",
    n_steps: int = 100,
    mutation_kwargs: dict[str, Any] | None = None,
    particle_transform: ParticleTransform | None = None,
    callbacks: Sequence[Callable[..., None]] | None = None,
    checkpoint_writers: Sequence[CheckpointWriter] | None = None,
) -> MarkovChainMonteCarlo:
    """Construct standalone MCMC using an existing Sequince mutation backend."""
    kwargs = dict(mutation_kwargs or {})
    kwargs["store_chains"] = True
    mutation_step = get_mutation_step(
        mutation,
        target=log_prob_fn,
        transform=particle_transform,
        mutation_n_steps=n_steps,
        mutation_kwargs=kwargs,
    )
    if not isinstance(mutation_step, MutationStep):
        raise ValueError("Standalone MCMC requires a mutation step.")
    return MarkovChainMonteCarlo(
        mutation_step,
        callbacks=callbacks,
        checkpoint_writers=checkpoint_writers,
    )
