from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ._typing import Array
from .proposals import ProposalDistribution
from .state import ParticleState
from .steps.context import LOG_PROPOSAL_OVERRIDE_KEY, StepContext
from .target import (
    EvaluationCounts,
    TargetComponentFn,
    TargetEvaluation,
    TemperedTarget,
    _evaluate_component,
)
from .transforms import ParticleTransform, get_prebuilt_transform
from .utils.callables import call_with_optional_context
from .utils.copying import array_safe_copy

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ._typing import MutationStrategy
    from .result import SMCResult
    from .steps.context import StepContext

LogProposalLike = TargetComponentFn | ProposalDistribution
JointLogDensityFn = Callable[..., tuple[Array | float, Array | float]]


def _resolve_log_proposal_fn(
    log_proposal: LogProposalLike | None,
) -> TargetComponentFn | None:
    """Normalize proposal inputs to a log-density callable."""
    if log_proposal is None:
        return None
    if callable(log_proposal):
        return log_proposal
    log_prob = getattr(log_proposal, "log_prob", None)
    if callable(log_prob):
        return log_prob
    raise TypeError(
        "log_proposal must be a callable, a ProposalDistribution, or None."
    )


@dataclass(frozen=True)
class TemperedPosterior(TemperedTarget):
    """Wrapper exposing posterior building blocks for SMC workflows.

    Parameters
    ----------
    log_prior:
        Callable returning ``log p(x)`` for each particle.
    log_likelihood:
        Callable returning ``log l(x)`` for each particle. May be omitted when
        ``log_likelihood_and_proposal`` is supplied.
    log_proposal:
        Optional callable or proposal object returning ``log q(x)``. When
        supplied the intermediate targets follow the bridge
        ``q(x)^{1-β} [p(x)L(x)]^{β}``.
    log_proposal_override_key:
        Context metadata key containing a runtime proposal override.
    log_likelihood_and_proposal:
        Alternative to separate likelihood and proposal inputs: a callable
        returning ``(log l(x), log q(x))`` from shared computation. Accepts
        particles and optionally context. Requesting either component evaluates
        and caches both. A runtime proposal override takes precedence over the
        returned proposal; refreshing only that override skips the joint call.
    """

    log_prior: TargetComponentFn
    log_likelihood: TargetComponentFn | None = None
    log_proposal: LogProposalLike | None = None
    log_proposal_override_key: str = LOG_PROPOSAL_OVERRIDE_KEY
    log_likelihood_and_proposal: JointLogDensityFn | None = None

    def __post_init__(self) -> None:
        if self.log_likelihood_and_proposal is None:
            if self.log_likelihood is None:
                raise ValueError(
                    "Supply log_likelihood or log_likelihood_and_proposal."
                )
        elif self.log_likelihood is not None or self.log_proposal is not None:
            raise ValueError(
                "log_likelihood_and_proposal cannot be combined with "
                "log_likelihood or log_proposal."
            )

    def _joint_log_proposal(self, particles: Array, context: StepContext):
        assert self.log_likelihood_and_proposal is not None
        _, log_proposal = call_with_optional_context(
            self.log_likelihood_and_proposal, (particles,), context
        )
        return log_proposal

    def resolve_log_proposal(
        self,
        log_proposal_override: LogProposalLike | None = None,
    ) -> TargetComponentFn | None:
        """Return the effective proposal callable for the current evaluation."""
        if log_proposal_override is not None:
            return _resolve_log_proposal_fn(log_proposal_override)
        if self.log_likelihood_and_proposal is not None:
            return self._joint_log_proposal
        return _resolve_log_proposal_fn(self.log_proposal)

    def log_initial(
        self,
        particles: Array,
        context: "StepContext",
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> float | Any:
        """Return the proposal or prior endpoint log density.

        When ``evaluation`` is omitted, all posterior components are evaluated
        from ``particles``. Supplying a matching evaluation avoids recomputing
        those components.
        """
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        log_proposal_override = context.metadata.get(
            self.log_proposal_override_key
        )
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal is None:
            if particles.ndim == 1:
                return evaluation["log_prior"][0]
            return evaluation["log_prior"]
        if "log_proposal" not in evaluation:
            raise ValueError("Evaluation does not contain log_proposal.")
        if particles.ndim == 1:
            return evaluation["log_proposal"][0]
        return evaluation["log_proposal"]

    def log_terminal(
        self,
        particles: Array,
        context: "StepContext",
        *,
        evaluation: TargetEvaluation | None = None,
    ) -> float | Any:
        """Return the posterior endpoint log density.

        When ``evaluation`` is omitted, all posterior components are evaluated
        from ``particles``. Supplying a matching evaluation avoids recomputing
        those components.
        """
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        posterior = evaluation["log_prior"] + evaluation["log_likelihood"]
        if particles.ndim == 1:
            return posterior[0]
        return posterior

    def evaluate(
        self,
        particles: Array,
        context: "StepContext",
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
    ) -> TargetEvaluation:
        """Evaluate or refresh posterior components for a particle population."""
        evaluation, _ = self._evaluate(
            particles,
            context,
            components=components,
            evaluation=evaluation,
        )
        return evaluation

    def _evaluate(
        self,
        particles: Array,
        context: "StepContext",
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
        refresh: bool = False,
    ) -> tuple[TargetEvaluation, EvaluationCounts]:
        """Evaluate posterior components and report newly incurred work."""
        del refresh
        if evaluation is not None:
            self.validate_evaluation(evaluation, particles, context)
        allowed = {"log_prior", "log_likelihood", "log_proposal"}
        requested = allowed if components is None else set(components)
        unknown = requested.difference(allowed)
        if unknown:
            raise ValueError(
                f"Unknown evaluation components: {sorted(unknown)}"
            )
        override = context.metadata.get(self.log_proposal_override_key)
        values, counts = self._evaluate_components(
            particles,
            context,
            requested,
            {} if evaluation is None else dict(evaluation.components),
            self.resolve_log_proposal(override),
            proposal_overridden=override is not None,
        )
        return (
            self.bind_evaluation(TargetEvaluation(components=values), context),
            counts,
        )

    def _evaluate_components(
        self,
        particles: Array,
        context: StepContext,
        requested: set[str],
        values: dict[str, Array],
        log_proposal: TargetComponentFn | None,
        *,
        proposal_overridden: bool,
    ) -> tuple[dict[str, Array], EvaluationCounts]:
        n_particles = 1 if particles.ndim == 1 else particles.shape[0]
        log_target_evaluations = (
            n_particles
            if requested.intersection({"log_prior", "log_likelihood"})
            else 0
        )
        log_likelihood_evaluations = 0
        log_proposal_evaluations = 0
        if "log_prior" in requested:
            values["log_prior"] = _evaluate_component(
                self.log_prior, particles, context
            )
        joint = self.log_likelihood_and_proposal
        use_joint = joint is not None and (
            "log_likelihood" in requested
            or ("log_proposal" in requested and not proposal_overridden)
        )
        if use_joint:
            assert joint is not None
            likelihood_value, proposal_value = call_with_optional_context(
                joint, (particles,), context
            )
            values["log_likelihood"] = _evaluate_component(
                lambda _: likelihood_value, particles, context
            )
            if not proposal_overridden:
                values["log_proposal"] = _evaluate_component(
                    lambda _: proposal_value, particles, context
                )
            log_target_evaluations = n_particles
            log_likelihood_evaluations = n_particles
            log_proposal_evaluations = n_particles
        elif "log_likelihood" in requested:
            assert self.log_likelihood is not None
            values["log_likelihood"] = _evaluate_component(
                self.log_likelihood, particles, context
            )
            log_likelihood_evaluations = n_particles
        if "log_proposal" in requested and (
            not use_joint or proposal_overridden
        ):
            if log_proposal is None:
                values.pop("log_proposal", None)
            else:
                values["log_proposal"] = _evaluate_component(
                    log_proposal, particles, context
                )
                log_proposal_evaluations += n_particles
        return (
            values,
            EvaluationCounts(
                log_target_evaluations=log_target_evaluations,
                log_likelihood_evaluations=log_likelihood_evaluations,
                log_proposal_evaluations=log_proposal_evaluations,
            ),
        )

    def freeze_log_prob(
        self,
        context: StepContext,
        *,
        log_proposal_override: LogProposalLike | None = None,
    ):
        frozen_context = StepContext(
            t=context.t,
            next_observation_index=context.next_observation_index,
            observation=context.observation,
            shared=context.shared,
            metadata=dict(context.metadata),
            evaluation_counts=context.evaluation_counts,
            stop_requested=context.stop_requested,
            stop_immediately=context.stop_immediately,
        )
        if log_proposal_override is not None:
            frozen_context = frozen_context.with_metadata(
                **{self.log_proposal_override_key: log_proposal_override}
            )
        override = frozen_context.metadata.get(self.log_proposal_override_key)
        log_proposal = self.resolve_log_proposal(override)

        def _frozen_log_prob(particles):
            components, _ = self._evaluate_components(
                particles,
                frozen_context,
                {"log_prior", "log_likelihood", "log_proposal"},
                {},
                log_proposal,
                proposal_overridden=override is not None,
            )
            evaluation = TargetEvaluation(
                components=components,
                target_id=self.evaluation_id,
            )
            return self.log_prob(
                particles,
                frozen_context,
                evaluation=evaluation,
            )

        return _frozen_log_prob


def draw_additional_samples(
    *,
    log_prior: Callable[..., Any] | None = None,
    log_likelihood: Callable[..., Any] | None = None,
    log_proposal: Callable[..., Any] | ProposalDistribution | None = None,
    posterior: TemperedPosterior | None = None,
    result: SMCResult,
    rng: "RandomGenerator",
    n_particles: int,
    mutation: MutationStrategy | None = "minipcn",
    mutation_n_steps: int = 10,
    mutation_kwargs: dict[str, Any] | None = None,
    particle_transform: ParticleTransform | str | None = "affine",
    particle_transform_kwargs: dict[str, Any] | None = None,
) -> tuple[ParticleState, SMCResult]:
    """Draw additional samples from the current particles to fill up to n_particles.

    Parameters
    ----------
    log_prior:
        Callable evaluating the log prior density ``log p(x)``. If `posterior` is
        not provided, both `log_prior` and `log_likelihood` must be given.
    log_likelihood:
        Callable evaluating the log likelihood ``log l(x)``. If `posterior` is
        not provided, both `log_prior` and `log_likelihood` must be given.
    log_proposal:
        Optional callable or proposal object for ``log q(x)`` when importance
        sampling proposals are used.
    posterior:
        TemperedPosterior object defining the target distribution.
    state:
        Current ParticleState containing particles and weights.
    context:
        StepContext providing shared information such as beta.
    rng:
        Random number generator.
    n_particles:
        Desired total number of particles after drawing additional samples.
    mutation:
        Mutation strategy to apply to the new particles.
    mutation_n_steps:
        Number of mutation steps to apply if mutation is used.
    mutation_kwargs:
        Additional keyword arguments to pass to the mutation step.
    particle_transform:
        ParticleTransform instance or string identifier for the transform to use
        during mutation. If `None`, no transform is applied.
    """
    from .result import SMCResult
    from .steps.mutation import get_mutation_step

    state = result.final_state
    context = result.context

    if posterior is None:
        if log_prior is None or log_likelihood is None:
            raise ValueError(
                "Either 'posterior' or both 'log_prior' and 'log_likelihood' must be provided."
            )
        posterior = TemperedPosterior(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            log_proposal=log_proposal,
        )

    # Resample particles to reach the desired number
    current_n_particles = state.particles.shape[0]
    if current_n_particles >= n_particles:
        return state, result

    # Resample particles to reach the desired number using the terminal weights.
    weights = state.weights
    indices = rng.choice(
        current_n_particles, size=n_particles, replace=True, p=weights
    )
    resampled_particles = state.particles[indices]
    resampled_state = ParticleState(
        particles=resampled_particles,
        log_normalizing_constant=state.log_normalizing_constant,
        initial_ancestor_indices=state.initial_ancestor_indices[indices],
        evaluation=(
            None
            if state.evaluation is None
            else state.evaluation.take(indices)
        ),
        metadata=array_safe_copy(state.metadata),
    )

    if isinstance(particle_transform, str):
        particle_transform_kwargs = particle_transform_kwargs or {}
        particle_transform = get_prebuilt_transform(
            particle_transform, **particle_transform_kwargs
        )
    elif (
        isinstance(particle_transform, ParticleTransform)
        or particle_transform is None
    ):
        pass
    else:
        raise ValueError(
            "particle_transform must be a string identifier, "
            "a ParticleTransform instance, or None."
        )

    mutation_step = get_mutation_step(
        mutation=mutation,
        target=posterior,
        mutation_n_steps=mutation_n_steps,
        mutation_kwargs=mutation_kwargs,
        transform=particle_transform,
    )
    if mutation_step is None:
        updated_result = result.append_state(resampled_state)
        return resampled_state, updated_result

    state, context, output = mutation_step.apply(
        resampled_state,
        context,
        rng,
    )
    updated_result = SMCResult(
        states=[*result.states, state],
        context=context,
        initial_state=result.initial_state,
        outputs=[*result.outputs, (output,)],
    )
    return state, updated_result
