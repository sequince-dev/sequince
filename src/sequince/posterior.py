from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ._array_api import asarray, full, namespace
from ._typing import Array
from .proposals import ProposalDistribution
from .state import ParticleState
from .steps.context import StepContext
from .target import AnnealedTarget, TargetEvaluation
from .transforms import ParticleTransform, get_prebuilt_transform
from .utils import call_with_optional_context

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ._typing import MutationStrategy
    from .result import SMCResult
    from .steps.context import StepContext

ArrayLikeFn = Callable[..., Array | float]
LogProposalLike = ArrayLikeFn | ProposalDistribution


def _evaluate(
    fn: ArrayLikeFn,
    particles: Array,
    context: "StepContext",
) -> Array:
    """Call function with optional context and return a 1d float array."""
    values = call_with_optional_context(fn, (particles,), context)
    xp = namespace(particles, values)
    array = asarray(values, xp=xp, like=particles)
    if array.ndim == 0:
        n_particles = 1 if particles.ndim == 1 else particles.shape[0]
        array = full(
            (n_particles,),
            float(array),
            xp=xp,
            dtype=array.dtype,
            like=particles,
        )
    if array.ndim > 1:
        array = xp.reshape(array, (array.shape[0],))
    return array


def _resolve_log_proposal_fn(
    log_proposal: LogProposalLike | None,
) -> ArrayLikeFn | None:
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


def _reject_unknown_options(options: Mapping[str, Any]) -> None:
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"Unknown target options: {names}.")


@dataclass(frozen=True)
class TemperedPosterior(AnnealedTarget):
    """Wrapper exposing posterior building blocks for SMC workflows.

    Parameters
    ----------
    log_prior:
        Callable returning ``log p(x)`` for each particle.
    log_likelihood:
        Callable returning ``log l(x)`` for each particle.
    log_proposal:
        Optional callable or proposal object returning ``log q(x)``. When
        supplied the intermediate targets follow the bridge
        ``q(x)^{1-β} [p(x)L(x)]^{β}``.
    """

    log_prior: ArrayLikeFn
    log_likelihood: ArrayLikeFn
    log_proposal: LogProposalLike | None = None

    def resolve_log_proposal(
        self,
        log_proposal_override: LogProposalLike | None = None,
    ) -> ArrayLikeFn | None:
        """Return the effective proposal callable for the current evaluation."""
        if log_proposal_override is not None:
            return _resolve_log_proposal_fn(log_proposal_override)
        return _resolve_log_proposal_fn(self.log_proposal)

    def log_increment(
        self,
        particles: Array,
        context: "StepContext",
        *,
        evaluation: TargetEvaluation | None = None,
        log_proposal_override: LogProposalLike | None = None,
        **kwargs: Any,
    ) -> float | Any:
        """Return the log-weight increment required by the tempering step.

        When ``evaluation`` is omitted, all posterior components are evaluated
        from ``particles``. Supplying a matching evaluation avoids recomputing
        those components.
        """
        _reject_unknown_options(kwargs)
        if evaluation is None:
            evaluation = self.evaluate(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )
        self.validate_evaluation(evaluation, particles)
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal is None:
            if particles.ndim == 1:
                return float(evaluation["log_likelihood"][0])
            return evaluation["log_likelihood"]
        if "log_proposal" not in evaluation:
            raise ValueError("Evaluation does not contain log_proposal.")
        bridge = (
            evaluation["log_prior"]
            + evaluation["log_likelihood"]
            - evaluation["log_proposal"]
        )
        if particles.ndim == 1:
            return float(bridge[0])
        return bridge

    def log_prob(
        self,
        particles: Array,
        context: "StepContext",
        *,
        evaluation: TargetEvaluation | None = None,
        log_proposal_override: LogProposalLike | None = None,
        **kwargs: Any,
    ) -> float | Any:
        """Return the tempered log posterior evaluated at the current β.

        When ``evaluation`` is omitted, all posterior components are evaluated
        from ``particles``. Supplying a matching evaluation avoids recomputing
        those components.
        """
        _reject_unknown_options(kwargs)
        if evaluation is None:
            evaluation = self.evaluate(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )
        self.validate_evaluation(evaluation, particles)
        beta = float(context.metadata.get("temperature", 0.0))
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal is None:
            posterior = (
                evaluation["log_prior"] + beta * evaluation["log_likelihood"]
            )
        else:
            if "log_proposal" not in evaluation:
                raise ValueError("Evaluation does not contain log_proposal.")
            if beta == 0.0:
                posterior = evaluation["log_proposal"]
            elif beta == 1.0:
                posterior = (
                    evaluation["log_prior"] + evaluation["log_likelihood"]
                )
            else:
                posterior = (1.0 - beta) * evaluation[
                    "log_proposal"
                ] + beta * (
                    evaluation["log_prior"] + evaluation["log_likelihood"]
                )
        if particles.ndim == 1:
            return float(posterior[0])
        return posterior

    def evaluate(
        self,
        particles: Array,
        context: "StepContext",
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
        log_proposal_override: LogProposalLike | None = None,
        log_proposal_values: Array | None = None,
        **kwargs: Any,
    ) -> TargetEvaluation:
        """Evaluate or refresh posterior components for a particle population."""
        _reject_unknown_options(kwargs)
        if evaluation is not None:
            self.validate_evaluation(evaluation, particles)
        allowed = {"log_prior", "log_likelihood", "log_proposal"}
        requested = allowed if components is None else set(components)
        unknown = requested.difference(allowed)
        if unknown:
            raise ValueError(
                f"Unknown evaluation components: {sorted(unknown)}"
            )
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal_values is not None and log_proposal is None:
            raise ValueError(
                "log_proposal_values were supplied without a proposal."
            )
        values = {} if evaluation is None else dict(evaluation.components)
        if "log_prior" in requested:
            values["log_prior"] = _evaluate(self.log_prior, particles, context)
        if "log_likelihood" in requested:
            values["log_likelihood"] = _evaluate(
                self.log_likelihood, particles, context
            )
        if "log_proposal" in requested:
            if log_proposal is None:
                values.pop("log_proposal", None)
            else:
                values["log_proposal"] = (
                    log_proposal_values
                    if log_proposal_values is not None
                    else _evaluate(log_proposal, particles, context)
                )
        return self.bind_evaluation(TargetEvaluation(components=values))

    def freeze_log_prob(
        self,
        context: StepContext,
        *,
        log_proposal_override: LogProposalLike | None = None,
    ):
        frozen_context = StepContext(
            t=context.t,
            observation=context.observation,
            shared=copy.deepcopy(dict(context.shared)),
            metadata=copy.deepcopy(dict(context.metadata)),
        )
        log_prior = self.log_prior
        log_likelihood = self.log_likelihood
        log_proposal = self.resolve_log_proposal(log_proposal_override)

        def _frozen_log_prob(particles):
            components = {
                "log_prior": _evaluate(log_prior, particles, frozen_context),
                "log_likelihood": _evaluate(
                    log_likelihood, particles, frozen_context
                ),
            }
            if log_proposal is not None:
                components["log_proposal"] = _evaluate(
                    log_proposal, particles, frozen_context
                )
            evaluation = TargetEvaluation(
                components=components,
                target_id=self.evaluation_id,
            )
            return self.log_prob(
                particles,
                frozen_context,
                evaluation=evaluation,
                log_proposal_override=log_proposal_override,
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
        StepContext providing shared information such as temperature.
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
        log_evidence=state.log_evidence,
        evaluation=(
            None
            if state.evaluation is None
            else state.evaluation.take(indices)
        ),
        metadata=copy.deepcopy(state.metadata),
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

    state = mutation_step.apply(
        resampled_state,
        context,
        rng,
    )
    updated_result = result.append_state(state)
    return state, updated_result
