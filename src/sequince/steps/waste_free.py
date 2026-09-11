"""Waste-free resampling and chain retention around an existing mutation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any

from array_api_compat import device as get_device

from ..array_api import namespace
from .base import SMCStep, StepContext
from .mutation import MutationMetricsOutput, MutationStep

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ..state import ParticleState


class WasteFreeResampleMove(SMCStep):
    """Resample chain starts and retain a wrapped mutation's full chains.

    Implements the resampling and chain-retention portion of Algorithm 2 of
    Dau and Chopin [1]_. With N particles and chain length P, resample M=N/P
    starts using the input weights, run P-1 transitions and keep all P states
    per chain. Starts and repeated states after rejection are retained. No
    further resampling occurs inside this step; the output has N uniform
    weights and preserves the input evidence estimate.

    Parameters
    ----------
    mutation:
        A MutationStep supporting apply_with_n_steps() and stored chains,
        such as MetropolisHastingsMutation, BlackJAXMutation or MiniPCNMutation.
        It owns the target, proposal, transforms and transition logic. One
        complete block of P-1 steps is requested, overriding its default
        n_steps for this call only. Chain storage is temporarily enabled and
        its previous setting restored even if mutation fails.
    chain_length:
        Number P of retained states per chain, INCLUDING the start. Must be a
        positive integer dividing N. P=1 only resamples; mutation is not called.

    Notes
    -----
    The mutation must preserve the current target and particle ordering, and
    return chains in original particle coordinates with shape
    ``(P-1, M, *particle_shape)``. For the paper's independent-chain setting,
    use per-chain transitions; ensemble samplers couple chains and do not
    inherit that independence assumption. Fit kernels/transforms before the
    block and keep them fixed inside it. Put adaptation wrappers OUTSIDE this
    step to adapt after the entire batch, not between retained transitions.

    Output particles are ordered by chain position, then chain index. Reshape
    them to ``(P, M, *particle_shape)`` to recover the chains. ``ancestors``
    maps every state to its input ancestor. ``metrics`` is the mutation's
    completed metrics; acceptance semantics are those of the chosen sampler.
    Its full output, including sampler-specific diagnostics, is under
    ``mutation``. ``acceptance_by_step`` is forwarded when available, otherwise
    None. Uniform weights do not imply independent samples.

    After mutation, the population evaluation cache is cleared: an endpoint
    evaluation cannot describe the entire retained chain. Subsequent target
    preparation (e.g. weighting) evaluates the retained population as needed.
    Pure resampling (P=1) preserves the resampled cache.

    Examples
    --------
    With 1000 input particles, run 20 chains with 50 retained states each:

    >>> mutation = MetropolisHastingsMutation(target, GaussianRandomWalk(0.5))
    >>> step = WasteFreeResampleMove(mutation, chain_length=50)

    References
    ----------
    .. [1] Dau, H.-D. and Chopin, N. (2022). Waste-free Sequential Monte
       Carlo. JRSS B, 84(1), 114-148. https://doi.org/10.1111/rssb.12475
       Algorithm 2: https://arxiv.org/abs/2011.02328
    """

    def __init__(self, mutation: MutationStep, *, chain_length: int):
        if not isinstance(mutation, MutationStep):
            raise TypeError(
                "mutation must be a MutationStep supporting stored chains."
            )
        if isinstance(chain_length, bool) or not isinstance(
            chain_length, Integral
        ):
            raise TypeError("chain_length must be a positive integer.")
        self._chain_length: int = int(chain_length)
        if self._chain_length < 1:
            raise ValueError("chain_length must be >= 1.")
        self.mutation: MutationStep = mutation

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        n = state.particles.shape[0]
        p = self._chain_length
        if n == 0 or n % p:
            raise ValueError(
                "WasteFreeResampleMove requires a nonempty population whose "
                "size is divisible by chain_length."
            )
        m = n // p
        xp = namespace(state.particles, state.log_weights)
        ancestor_indices = rng.choice(n, size=m, replace=True, p=state.weights)
        ancestors = xp.concat((ancestor_indices,) * p, axis=0)
        starts = state.resample(ancestor_indices)
        mutation_output = None
        metrics = MutationMetricsOutput(
            n_steps=xp.asarray(0),
            n_attempted=xp.asarray(0),
            n_accepted=xp.asarray(0),
        )
        acceptance_by_step: Any = ()
        if p == 1:
            result = starts
        else:
            store_chains = self.mutation.store_chains
            try:
                self.mutation.store_chains = True
                _, context, mutation_output = self.mutation.apply_with_n_steps(
                    starts, context, rng, n_steps=p - 1
                )
            finally:
                self.mutation.store_chains = store_chains
            if not isinstance(mutation_output, Mapping):
                raise TypeError(
                    "Mutation output must be a mapping with chain and metrics."
                )
            chain = mutation_output.get("chain")
            expected_shape = (p - 1, m, *state.particles.shape[1:])
            if (
                chain is None
                or getattr(chain, "shape", None) != expected_shape
            ):
                raise ValueError(
                    f"Mutation must return chain with shape {expected_shape} "
                    "in original particle coordinates."
                )
            mutation_metrics = mutation_output.get("metrics")
            if not isinstance(mutation_metrics, MutationMetricsOutput):
                raise TypeError(
                    "Mutation must return completed MutationMetricsOutput."
                )
            metrics = mutation_metrics
            acceptance_by_step = mutation_output.get("acceptance_by_step")
            particles = xp.concat(
                (
                    starts.particles,
                    xp.reshape(
                        chain, ((p - 1) * m, *state.particles.shape[1:])
                    ),
                ),
                axis=0,
            )
            result = state.replace_particles(
                particles,
                evaluation=None,
                log_weights=xp.full(
                    (n,),
                    -math.log(n),
                    dtype=state.log_weights.dtype,
                    device=get_device(state.log_weights),
                ),
                initial_ancestor_indices=state.initial_ancestor_indices[
                    ancestors
                ],
            )
        return (
            result,
            context,
            {
                "metrics": metrics,
                "ancestors": ancestors,
                "chain_length": p,
                "n_chains": m,
                "acceptance_by_step": acceptance_by_step,
                "mutation": mutation_output,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        """Forward the mutation's checkpoint state."""
        return {"mutation": self.mutation.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the wrapped mutation's checkpoint state."""
        self.mutation.load_state_dict(state.get("mutation", {}))
