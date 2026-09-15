from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..._typing import Array
from ...array_api import asarray, namespace
from ...target import EvaluationCounts
from ...transforms import ParticleTransform
from ...utils.pytrees import pytree_dataclass
from ..base import validate_n_steps
from ..context import StepContext
from .base import (
    MutationStep,
    TransformRefit,
)
from .metrics import (
    MutationMetric,
    MutationStepState,
)
from .metrics.base import with_metrics

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target
    from .metropolis import MetropolisHastingsMutation

logger = logging.getLogger(__name__)


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

    def _validate_mixture_support(self) -> None:
        """Reject options that violate a frozen mixture proposal."""
        if self._transform is not None:
            raise ValueError(
                "MiniPCN mixture children do not support transforms."
            )
        if self._kwargs.get("adaptive", False):
            raise ValueError(
                "MiniPCN mixture children require adaptive=False."
            )
        if not isinstance(self._step_fn, str) or self._step_fn.lower() not in {
            "pcn",
            "tpcn",
        }:
            raise ValueError("MiniPCN mixture children require pcn or tpcn.")

    def _prepare_mixture_mutation(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> MetropolisHastingsMutation:
        """Fit a block-local MiniPCN proposal for interleaved MH transitions.

        Fit from the population at block entry, before any child transitions.
        The returned mutation reuses that fit, while evaluating the latest
        particles on each selection. Nothing is cached on this child between
        blocks, so runner checkpoints need no additional proposal state.
        """
        from minipcn.step import step_factory

        from .kernels import CallableMutationKernel
        from .metropolis import MetropolisHastingsMutation

        self._validate_mixture_support()
        particles = state.particles
        if particles.ndim != 2 or particles.shape[0] < 2:
            raise ValueError(
                "MiniPCN mixtures require particles with shape (n, d), n >= 2."
            )
        rng_backend, _ = rng.to_functional()
        kwargs = dict(self._kwargs)
        # Sampler-level settings do not belong to the frozen proposal. As in
        # standalone execution, the explicitly supplied RNG takes precedence.
        kwargs.pop("target_acceptance_rate", None)
        kwargs.pop("rng", None)
        step = step_factory(
            self._step_fn,
            dims=particles.shape[1],
            xp=namespace(particles),
            rng_backend=rng_backend,
            **kwargs,
        )
        prepared = step.init_state(particles)
        correction: Array | None = None

        def sample(particles: Array, rng: RandomGenerator) -> Array:
            nonlocal correction
            next_random_state, proposals, correction = step.propose(
                prepared, rng.state(), particles
            )
            rng.set_state(next_random_state)
            return proposals

        def log_transition_ratio(start: Array, end: Array) -> Array:
            # MH consumes this correction immediately after its proposal.
            if correction is None:  # pragma: no cover - call order guard
                raise RuntimeError("MiniPCN proposal has not been sampled.")
            return correction

        target = (
            self._target if self._target is not None else self._log_prob_fn
        )
        assert target is not None
        return MetropolisHastingsMutation(
            target,
            CallableMutationKernel(
                sample, log_transition_ratio_fn=log_transition_ratio
            ),
            metrics=list(self.metrics),
        )

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
            mutation_state = (
                self.metrics.initialize(
                    state.particles, state.log_weights, context
                )
                if mutation_state is None
                else self.metrics.prepare(
                    mutation_state, state.particles, context
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
        from minipcn import Sampler

        xp = namespace(state.particles)
        transform, context = self._prepare_transform(
            state.particles, context, rng=rng
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
            **self._kwargs,
        )
        return (
            sampler,
            latent_particles,
            transform,
            transform_auxiliary,
            context,
        )

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

        if diagnostic_chain is not None:
            assert mutation_state is not None

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

        # Last-position-only sampling still exposes actual per-step acceptance.
        if diagnostic_chain is None:
            assert mutation_state is not None
            n_particles = latent_particles.shape[0]
            mutation_state = replace(
                mutation_state,
                n_steps=mutation_state.n_steps + n_steps,
                n_attempted=mutation_state.n_attempted + n_steps * n_particles,
                n_accepted=mutation_state.n_accepted
                + xp.asarray(
                    xp.round(xp.sum(acceptance_by_step) * n_particles),
                    dtype=mutation_state.n_accepted.dtype,
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
