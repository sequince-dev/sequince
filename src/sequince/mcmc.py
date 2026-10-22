from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Self

from .array_api import asarray, namespace
from .checkpoint import CheckpointWriter, SMCCheckpoint
from .result import SMCResult, _decode_result_output, _encode_result_output
from .smc import SequentialMonteCarlo, SMCRuntime
from .state import ParticleState
from .steps.base import StepContext
from .steps.mutation import MutationStep, get_mutation_step
from .transforms import ParticleTransform
from .typing import Array

if TYPE_CHECKING:
    import numpy as np
    from orng import RandomGenerator


@dataclass(frozen=True)
class MCMCResult:
    """Result of applying one or more checkpointable MCMC mutation blocks."""

    states: Sequence[ParticleState]
    context: StepContext
    outputs: Sequence[tuple[Any, ...]]

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("MCMCResult requires at least one ParticleState.")
        if not self._chains():
            raise ValueError(
                "MCMCResult requires mutation steps configured with "
                "store_chains=True."
            )

    @classmethod
    def from_smc_result(cls, result: SMCResult) -> MCMCResult:
        """Build an MCMC result from a mutation-only sequential run."""
        return cls(
            states=result.states,
            context=result.context,
            outputs=result.outputs,
        )

    def encode_for_hdf5(self, *, strict: bool = False) -> dict[str, Any]:
        """Return the logical HDF5 payload for this MCMC result."""
        from .utils.hdf5 import nested, schema_header

        return {
            **schema_header("mcmc_result"),
            "states": [
                nested(state.encode_for_hdf5()) for state in self.states
            ],
            "context": self.context._encode_for_result_hdf5(),
            "outputs": _encode_result_output(
                self.outputs, "outputs", set(), strict=strict
            ),
        }

    @classmethod
    def _decode_from_hdf5(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct an MCMC result from its logical payload."""
        from .utils.hdf5 import validate_schema

        validate_schema(
            payload,
            kinds={"mcmc_result"},
            required={"states", "context", "outputs"},
        )
        states = payload["states"]
        if not isinstance(states, (list, tuple)):
            raise TypeError("MCMCResult states must be a sequence.")
        return cls(
            states=[ParticleState._decode_from_hdf5(item) for item in states],
            context=StepContext._decode_from_result_hdf5(payload["context"]),
            outputs=_decode_result_output(payload["outputs"]),
        )

    def save(
        self,
        h5_file_or_group: Any,
        path: str = "mcmc_result",
        *,
        strict: bool = False,
    ) -> None:
        """Save this MCMC result below ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import recursively_save_to_h5_file

        recursively_save_to_h5_file(
            h5_file_or_group, path, self.encode_for_hdf5(strict=strict)
        )

    @classmethod
    def load(cls, h5_file_or_group: Any, path: str = "mcmc_result") -> Self:
        """Load an MCMC result from ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import load_from_h5_file

        return cls._decode_from_hdf5(load_from_h5_file(h5_file_or_group, path))

    def _chains(self) -> list[Array]:
        chains: list[Array] = []
        for iteration_outputs in self.outputs:
            for output in iteration_outputs:
                if not isinstance(output, Mapping):
                    continue
                chain = output.get("chain")
                if isinstance(chain, Mapping):
                    chains.extend(chain.values())
                elif chain is not None:
                    chains.append(chain)
        return chains

    @property
    def final_state(self) -> ParticleState:
        return self.states[-1]

    @property
    def chain(self) -> Array:
        """Return mutation blocks concatenated as ``(steps, chains, dims)``."""
        chains = self._chains()
        xp = namespace(*chains)
        arrays = [asarray(chain, xp=xp, like=chains[0]) for chain in chains]
        return xp.concat(arrays, axis=0)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the final mutation output."""
        for iteration_outputs in reversed(self.outputs):
            for output in reversed(iteration_outputs):
                if isinstance(output, Mapping) and "chain" in output:
                    return {
                        key: value
                        for key, value in output.items()
                        if key != "chain"
                    }
        return {}

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
        outputs: Sequence[tuple[Any, ...]] | None = None,
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
