"""Optional JAX integration for Sequince's functional SMC interface.

Register the runtime containers before compiling a runner method::

    from sequince.jax import register_pytrees

    register_pytrees()
    iteration = jax.jit(smc.functional_iteration)
    state, context, key, outputs, valid = iteration(state, context, key)

For use in ``jax.lax.scan``, the input and output state, context and step
outputs must retain the same pytree structure.  In particular, initialize
metadata keys and cached target evaluations before entering the scan when a
step would otherwise create them on its first iteration.  Shared/context data
crossing the compiled boundary must itself be a valid JAX pytree.

The initial standard-sampler subset is fixed tempering, unconditional
resampling and BlackJAX mutation without a stateful fitted transform.  Adaptive
tempering, conditional resampling, logging, callbacks, checkpoints, stopping
and observation iteration remain host-side concerns.
"""

from __future__ import annotations

_PYTREES_REGISTERED = False


def register_pytrees() -> None:
    """Register Sequince runtime containers as JAX pytrees.

    Call this once before passing :class:`ParticleState` or
    :class:`StepContext` directly across a JAX transformation boundary.  JAX
    remains optional and is imported only when this function is called.
    """
    global _PYTREES_REGISTERED
    if _PYTREES_REGISTERED:
        return

    import jax

    from .state import ParticleState
    from .steps.context import StepContext
    from .steps.mutation.blackjax import BlackJAXOutput
    from .target import TargetEvaluation

    jax.tree_util.register_dataclass(
        ParticleState,
        data_fields=(
            "particles",
            "log_weights",
            "log_evidence",
            "evaluation",
            "metadata",
        ),
        meta_fields=(),
    )
    jax.tree_util.register_dataclass(
        StepContext,
        data_fields=(
            "t",
            "next_observation_index",
            "observation",
            "shared",
            "metadata",
            "stop_requested",
            "stop_immediately",
        ),
        meta_fields=(),
    )
    jax.tree_util.register_dataclass(
        TargetEvaluation,
        data_fields=("components",),
        meta_fields=("target_id", "evaluation_key"),
    )
    jax.tree_util.register_dataclass(
        BlackJAXOutput,
        data_fields=("acceptance", "divergence", "chain"),
        meta_fields=("algorithm", "steps"),
    )
    _PYTREES_REGISTERED = True
