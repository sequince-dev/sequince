"""JAX-specific helpers for Sequince's functional SMC interface."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import jax

from .smc import IterationCallable, SequentialMonteCarlo
from .state import ParticleState
from .steps.context import StepContext

logger = logging.getLogger(__name__)


def compile_functional_iteration(
    smc: SequentialMonteCarlo,
    initial_state: ParticleState,
    random_state: Any,
    *,
    observation: Any = None,
    next_observation_index: int = 0,
    shared_context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IterationCallable:
    """Compile initial and steady-state functional iteration signatures.

    Warm-up transitions are evaluated and discarded. Because functional
    iterations explicitly thread all state, this does not consume the caller's
    state or random key. The supplied context structures must match those used
    by the subsequent run.
    """
    initialized_metadata = smc.initialize_functional_metadata(
        initial_state,
        metadata,
    )
    shared = shared_context if shared_context is not None else {}
    context = StepContext(
        t=0,
        next_observation_index=next_observation_index,
        observation=observation,
        shared=shared,
        metadata=initialized_metadata,
    )
    iteration = jax.jit(smc.functional_iteration)
    carry = (initial_state, context, random_state)
    carry_structure = jax.tree_util.tree_structure(carry)
    logger.debug(
        "Compiling functional iteration with initial carry structure: %s",
        carry_structure,
    )
    for _ in range(3):
        state, output_context, key, outputs = iteration(*carry)
        jax.block_until_ready((state, output_context, key, outputs))
        next_context = StepContext(
            t=context.t + 1,
            next_observation_index=next_observation_index,
            observation=observation,
            shared=shared,
            metadata=output_context.metadata,
        )
        next_carry = (state, next_context, key)
        next_structure = jax.tree_util.tree_structure(next_carry)
        if next_structure == carry_structure:  # type: ignore[operator]
            return iteration
        carry = next_carry
        carry_structure = next_structure
        context = next_context

    raise RuntimeError(
        "Functional iteration carry did not reach a stable pytree structure "
        "during compilation."
    )
