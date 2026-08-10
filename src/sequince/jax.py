"""JAX-specific helpers for Sequince's functional SMC interface."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import jax

from .smc import IterationCallable, SequentialMonteCarlo
from .state import ParticleState
from .steps.context import StepContext

logger = logging.getLogger(__name__)


def _carry_signature(tree: Any) -> tuple[Any, tuple[Any, ...]]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    leaf_signatures = []
    for leaf in leaves:
        abstract = jax.typeof(leaf)
        leaf_signatures.append(
            (
                tuple(abstract.shape),
                abstract.dtype,
            )
        )
    return structure, tuple(leaf_signatures)


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
    """Lower and compile one stable functional iteration signature.

    Functional state and context initialization runs before lowering, but
    compilation does not execute or discard an SMC transition. The supplied
    context structures must match those used by the subsequent run.
    """
    shared = shared_context if shared_context is not None else {}
    context = StepContext(
        t=0,
        next_observation_index=next_observation_index,
        observation=observation,
        shared=shared,
        metadata=metadata if metadata is not None else {},
    )
    prepared_state, prepared_context = smc.initialize_functional(
        initial_state,
        context,
    )
    carry = (prepared_state, prepared_context, random_state)
    logger.debug(
        "Lowering functional iteration with carry structure: %s",
        jax.tree_util.tree_structure(carry),
    )
    lowered = jax.jit(smc.functional_iteration).lower(*carry)
    output_state, output_context, output_key, _ = lowered.out_info
    output_carry = (output_state, output_context, output_key)
    if _carry_signature(carry) != _carry_signature(output_carry):
        raise ValueError(
            "Functional iteration changes its carry signature. Initialize "
            "all leaves in initialize_functional()."
        )
    return cast(IterationCallable, lowered.compile())
