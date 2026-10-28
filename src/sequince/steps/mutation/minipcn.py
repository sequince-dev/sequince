from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ...array_api import asarray, namespace
from ...target import EvaluationCounts
from ...transforms import ParticleTransform
from ...typing import Array
from ...utils.pytrees import pytree_dataclass
from ..base import validate_n_steps
from ..context import StepContext
from .base import (
    PARTICLE_TRANSFORM_KEY,
    MutationStep,
    TransformRefit,
)
from .metrics import (
    MutationMetric,
    MutationStepState,
)
from .metrics.base import with_metrics

if TYPE_CHECKING:
    from minipcn.step import StepState
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target

logger = logging.getLogger(__name__)

_MINIPCN_BLOCK_KEY = "_minipcn_proposal_block"


@dataclass(frozen=True)
class _MiniPCNBlock:
    """Proposal shared through context for one eager, fixed mutation block."""

    mutation: MiniPCNMutation
    rho: float
    proposal: StepState | None = None
    transform: ParticleTransform | None = None


@pytree_dataclass(
    data_fields=("acceptance_by_step", "chain", "metrics"),
    meta_fields=("step_fn", "steps"),
)
@dataclass(frozen=True)
class MiniPCNOutput(Mapping[str, Any]):
    """MiniPCN results with block acceptance in ``metrics.acceptance``.

    ``acceptance_by_step`` contains the backend's accepted fraction per step.
    """

    acceptance_by_step: Any
    step_fn: str
    steps: int
    chain: Any
    metrics: Any = None

    def __getitem__(self, key: str) -> Any:
        if key not in {
            "acceptance_by_step",
            "step_fn",
            "steps",
            "chain",
            "metrics",
        }:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(
            ("acceptance_by_step", "step_fn", "steps", "chain", "metrics")
        )

    def __len__(self) -> int:
        return 5


class MiniPCNMutation(MutationStep):
    """Run MiniPCN MCMC moves to rejuvenate particles.

    By default uses the 'tpcn' step function and targets an acceptance rate of 0.234.
    Also sets 'verbose' to False by default.

    Wrap with ``AdaptiveMiniPCNMutation`` for tuning between fixed mutation
    blocks. The backend ``adaptive=True`` option instead tunes rho inside
    sampling and cannot be combined with that wrapper.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
        *,
        n_steps: int = 1,
        step_fn: str = "tpcn",
        verbose: bool = False,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        metrics: tuple[MutationMetric, ...] | list[MutationMetric] = (),
        **kwargs,
    ) -> None:
        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
            metrics=metrics,
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self._n_steps = n_steps
        self._step_fn = step_fn
        self._verbose = verbose
        self._kwargs = kwargs
        self._kwargs.setdefault("target_acceptance_rate", 0.234)
        self._kwargs.setdefault("adaptive", False)

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple[ParticleState, StepContext, MiniPCNOutput]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self._n_steps
        )

    def apply_with_mutation_state(
        self,
        state: ParticleState,
        context: StepContext,
        rng: "RandomGenerator",
        *,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[
        ParticleState, StepContext, MiniPCNOutput, MutationStepState | None
    ]:
        n_steps = validate_n_steps(n_steps)
        if n_steps == 0:
            xp = namespace(state.particles)
            metric_particles = state.particles
            if (
                self.metrics
                and mutation_state is not None
                and mutation_state.n_steps > 0
            ):
                # Continue diagnostics in the preceding round's coordinates,
                # without fitting geometry for an empty round.
                block = self._proposal_block(context)
                transform = (
                    block.transform
                    if block is not None
                    else self._prepared_transform
                )
                if transform is not None:
                    context = context.with_metadata(
                        **{PARTICLE_TRANSFORM_KEY: transform}
                    )
                metric_particles, _ = self._forward_transform(
                    state.particles, transform, context, rng=rng
                )
            mutation_state = (
                self.metrics.initialize(
                    metric_particles, state.log_weights, context
                )
                if mutation_state is None
                else self.metrics.prepare(
                    mutation_state, metric_particles, context
                )
            )
            chain = (
                xp.empty(
                    (0, *state.particles.shape), dtype=state.particles.dtype
                )
                if self._store_chains
                else None
            )
            return (
                state,
                context,
                MiniPCNOutput(
                    acceptance_by_step=xp.empty(
                        (0,), dtype=state.particles.dtype
                    ),
                    step_fn=self._step_fn,
                    steps=0,
                    chain=chain,
                    metrics=None,
                ),
                mutation_state,
            )
        (
            sampler,
            latent_particles,
            transform,
            transform_auxiliary,
            context,
        ) = self._prepare_sampler(state, context, rng=rng)

        mutation_state = (
            self.metrics.initialize(
                latent_particles, state.log_weights, context
            )
            if mutation_state is None
            else self.metrics.prepare(
                mutation_state, latent_particles, context
            )
        )
        return_last_only = not (self._store_chains or self.metrics)
        chain, history = sampler.sample(
            latent_particles,
            n_steps=n_steps,
            verbose=self._verbose,
            rng=rng,
            return_last_only=return_last_only,
            **self._proposal_kwargs(context),
        )

        acceptance_by_step = asarray(
            history.acceptance_rate,
            xp=namespace(latent_particles),
            like=latent_particles,
        )
        return self._complete_mutation(
            state,
            context,
            latent_last=chain if return_last_only else chain[-1],
            latent_chain=chain[-n_steps:] if self._store_chains else None,
            diagnostic_chain=None if return_last_only else chain[-n_steps:],
            latent_particles=latent_particles,
            transform=transform,
            transform_auxiliary=transform_auxiliary,
            acceptance_by_step=acceptance_by_step,
            n_steps=n_steps,
            mutation_state=mutation_state,
        )

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, MiniPCNOutput]:
        """Run MiniPCN while explicitly threading backend random state."""
        from orng import RandomGenerator

        transform_rng = RandomGenerator.from_xp(
            namespace(random_state), generator=random_state
        )
        (
            sampler,
            latent_particles,
            transform,
            transform_auxiliary,
            context,
        ) = self._prepare_sampler(state, context, rng=transform_rng)
        random_state = transform_rng.state()
        mutation_state: MutationStepState | None = self.metrics.initialize(
            latent_particles, state.log_weights, context
        )
        return_last_only = not (self._store_chains or self.metrics)
        chain, history, next_random_state = sampler.sample_functional(
            latent_particles,
            n_steps=self._n_steps,
            rng_state=random_state,
            verbose=False,
            return_last_only=return_last_only,
        )
        xp = namespace(latent_particles)
        acceptance_rate = xp.asarray(history.acceptance_rate)
        acceptance_by_step = acceptance_rate
        latent_last = chain if return_last_only else chain[-1]
        latent_chain = None if return_last_only else chain[-self._n_steps :]
        updated_state, context, output, mutation_state = (
            self._complete_mutation(
                state,
                context,
                latent_last=latent_last,
                latent_chain=latent_chain if self._store_chains else None,
                diagnostic_chain=None
                if return_last_only
                else chain[-self._n_steps :],
                latent_particles=latent_particles,
                transform=transform,
                transform_auxiliary=transform_auxiliary,
                acceptance_by_step=acceptance_by_step,
                n_steps=self._n_steps,
                mutation_state=mutation_state,
            )
        )
        output = with_metrics(
            output, self.metrics.result(mutation_state, context)
        )
        return updated_state, context, next_random_state, output

    def _prepare_sampler(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        rng: Any = None,
    ) -> tuple[Any, Array, ParticleTransform | None, Any, StepContext]:
        """Prepare the latent particles and MiniPCN sampler."""
        try:
            from minipcn import Sampler
        except ModuleNotFoundError as exc:
            if exc.name != "minipcn":
                raise
            raise ImportError(
                "MiniPCN mutation requires the 'minipcn' extra; install it "
                "with `pip install 'sequince[minipcn]'`."
            ) from exc

        xp = namespace(state.particles)
        block = self._proposal_block(context)
        if block is None or block.proposal is None:
            if block is not None:
                weights = state.log_weights
                if not bool(xp.all(xp.isfinite(weights))) or not bool(
                    xp.all(weights == weights[0])
                ):
                    raise ValueError(
                        "AdaptiveMiniPCNMutation requires equal, finite "
                        "particle weights. Resample before mutation."
                    )
            transform, context = self._prepare_transform(
                state.particles, context, rng=rng
            )
        else:
            transform = block.transform
            if transform is not None:
                context = context.with_metadata(
                    **{PARTICLE_TRANSFORM_KEY: transform}
                )
        latent_particles, transform_auxiliary = self._forward_transform(
            state.particles, transform, context, rng=rng
        )

        def log_prob_fn(position):
            backend_position = asarray(position, xp=xp, like=latent_particles)
            original, log_det = self._inverse_transform(
                backend_position,
                transform,
                context,
                auxiliary=transform_auxiliary,
            )
            result, _, _ = self._evaluate_target(original, context)
            return self._combine_with_log_det(result, log_det)

        dims = latent_particles.shape[1] if latent_particles.ndim > 1 else 1
        sampler = Sampler(
            log_prob_fn=log_prob_fn,
            step_fn=self._step_fn,
            dims=dims,
            xp=xp,
            **(
                self._kwargs
                if block is None
                else {**self._kwargs, "adaptive": False, "rho": block.rho}
            ),
        )
        if block is not None and block.proposal is None:
            block = replace(
                block,
                proposal=sampler.init_step_state(latent_particles),
                transform=transform,
            )
            context = context.with_metadata(**{_MINIPCN_BLOCK_KEY: block})
        return (
            sampler,
            latent_particles,
            transform,
            transform_auxiliary,
            context,
        )

    def _proposal_block(self, context: StepContext) -> _MiniPCNBlock | None:
        block = context.metadata.get(_MINIPCN_BLOCK_KEY)
        if isinstance(block, _MiniPCNBlock) and block.mutation is self:
            return block
        return None

    def _proposal_kwargs(self, context: StepContext) -> dict[str, Any]:
        block = self._proposal_block(context)
        return {} if block is None else {"step_state": block.proposal}

    def _complete_mutation(
        self,
        state: ParticleState,
        context: StepContext,
        *,
        latent_last: Array,
        latent_chain: Array | None,
        diagnostic_chain: Array | None,
        latent_particles: Array,
        transform: ParticleTransform | None,
        transform_auxiliary: Any,
        acceptance_by_step: Any,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[
        ParticleState, StepContext, MiniPCNOutput, MutationStepState | None
    ]:
        """Map MiniPCN results back to particle space and build metrics."""
        xp = namespace(latent_particles)
        latent_last = asarray(latent_last, xp=xp, like=latent_particles)
        new_particles, _ = self._inverse_transform(
            latent_last,
            transform,
            context,
            auxiliary=transform_auxiliary,
        )
        _, final_evaluation, final_counts = self._evaluate_target(
            new_particles, context
        )
        counts = EvaluationCounts(
            log_target_evaluations=(
                final_counts.log_target_evaluations * (n_steps + 2)
            ),
            log_likelihood_evaluations=(
                final_counts.log_likelihood_evaluations * (n_steps + 2)
            ),
            log_proposal_evaluations=(
                final_counts.log_proposal_evaluations * (n_steps + 2)
            ),
            log_increment_evaluations=(
                final_counts.log_increment_evaluations * (n_steps + 2)
            ),
        )
        context = context.add_evaluation_counts(counts)

        stored_chain = None
        if latent_chain is not None:
            stored_chain = self._inverse_chain(
                latent_chain,
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=latent_particles,
            )

        if mutation_state is None:
            raise RuntimeError("MiniPCN mutation metrics are not initialized.")
        initial_metrics = mutation_state
        if diagnostic_chain is not None:

            def record(carry, positions):
                mutation_state, previous = carry
                accepted = xp.any(
                    xp.reshape(
                        positions != previous, (positions.shape[0], -1)
                    ),
                    axis=1,
                )
                return (
                    self.metrics.update(
                        mutation_state, positions, accepted, context
                    ),
                    positions,
                ), None

            from ...array_api import is_jax_namespace

            if is_jax_namespace(xp):
                import jax

                (mutation_state, _), _ = jax.lax.scan(
                    record,
                    (mutation_state, latent_particles),
                    diagnostic_chain,
                )
            else:
                carry = mutation_state, latent_particles
                for positions in diagnostic_chain:
                    carry, _ = record(carry, positions)
                mutation_state, _ = carry

        # Accepted proposals can round to unchanged coordinates. Use backend
        # acceptance for totals, independently of movement-based diagnostics.
        n_particles = latent_particles.shape[0]
        mutation_state = replace(
            mutation_state,
            n_steps=initial_metrics.n_steps + n_steps,
            n_attempted=initial_metrics.n_attempted + n_steps * n_particles,
            n_accepted=initial_metrics.n_accepted
            + xp.asarray(
                xp.round(xp.sum(acceptance_by_step) * n_particles),
                dtype=initial_metrics.n_accepted.dtype,
            ),
        )

        logger.debug(
            "Iteration %d: MiniPCNMutation completed after %d steps.",
            context.t,
            n_steps,
        )
        output = MiniPCNOutput(
            acceptance_by_step=acceptance_by_step,
            step_fn=self._step_fn,
            steps=n_steps,
            chain=stored_chain,
            metrics=None,
        )
        return (
            state.replace_particles(
                new_particles,
                evaluation=final_evaluation,
            ),
            context,
            output,
            mutation_state,
        )
