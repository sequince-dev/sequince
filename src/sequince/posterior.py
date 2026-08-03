from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ._array_api import asarray, full, namespace
from ._array_api import copy as array_copy
from ._typing import Array
from .proposals import ProposalDistribution
from .state import ParticleState
from .steps.context import StepContext
from .transforms import ParticleTransform, get_prebuilt_transform
from .utils import call_with_optional_context

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ._typing import MutationStrategy
    from .result import SMCResult
    from .steps.context import StepContext

ArrayLikeFn = Callable[..., Array | float]
LogProposalLike = ArrayLikeFn | ProposalDistribution


@dataclass(frozen=True)
class PosteriorEvaluation:
    """Particle-aligned values evaluated by a :class:`TemperedPosterior`."""

    log_prior: Array
    log_likelihood: Array
    log_proposal: Array | None = None
    auxiliary: Mapping[str, Array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = [self.log_prior, self.log_likelihood]
        if self.log_proposal is not None:
            arrays.append(self.log_proposal)
        arrays.extend(self.auxiliary.values())
        if any(array.ndim == 0 for array in arrays):
            raise ValueError(
                "Posterior evaluation arrays must be particle-aligned."
            )
        n_particles = self.log_prior.shape[0]
        if any(array.shape[0] != n_particles for array in arrays):
            raise ValueError(
                "Posterior evaluation arrays must have the same leading dimension."
            )

    def validate_for(self, particles: Array) -> None:
        """Validate that this evaluation has one entry per supplied particle."""
        n_particles = 1 if particles.ndim == 1 else particles.shape[0]
        if self.log_prior.shape[0] != n_particles:
            raise ValueError(
                "Posterior evaluation has "
                f"{self.log_prior.shape[0]} entries but received "
                f"{n_particles} particles."
            )

    def copy(self) -> "PosteriorEvaluation":
        """Return a detached copy preserving array backends."""
        xp = namespace(self.log_prior)
        return PosteriorEvaluation(
            log_prior=array_copy(self.log_prior, xp=xp),
            log_likelihood=array_copy(self.log_likelihood, xp=xp),
            log_proposal=(
                None
                if self.log_proposal is None
                else array_copy(self.log_proposal, xp=xp)
            ),
            auxiliary={
                key: array_copy(value, xp=xp)
                for key, value in self.auxiliary.items()
            },
        )

    def take(self, indices: Array) -> "PosteriorEvaluation":
        """Select particle evaluations using ancestor indices."""
        return PosteriorEvaluation(
            log_prior=self.log_prior[indices],
            log_likelihood=self.log_likelihood[indices],
            log_proposal=(
                None
                if self.log_proposal is None
                else self.log_proposal[indices]
            ),
            auxiliary={
                key: value[indices] for key, value in self.auxiliary.items()
            },
        )

    @classmethod
    def concatenate(
        cls, evaluations: Collection["PosteriorEvaluation"]
    ) -> "PosteriorEvaluation":
        """Concatenate evaluations from multiple particle batches."""
        items = list(evaluations)
        if not items:
            raise ValueError("No posterior evaluations to concatenate.")
        xp = namespace(items[0].log_prior)
        auxiliary_keys = items[0].auxiliary.keys()
        if any(item.auxiliary.keys() != auxiliary_keys for item in items[1:]):
            raise ValueError(
                "Posterior evaluations have different auxiliary fields."
            )
        has_proposal = items[0].log_proposal is not None
        if any(
            (item.log_proposal is not None) != has_proposal for item in items
        ):
            raise ValueError("Posterior evaluations disagree on log_proposal.")
        return cls(
            log_prior=xp.concat(tuple(item.log_prior for item in items)),
            log_likelihood=xp.concat(
                tuple(item.log_likelihood for item in items)
            ),
            log_proposal=(
                None
                if not has_proposal
                else xp.concat(tuple(item.log_proposal for item in items))
            ),
            auxiliary={
                key: xp.concat(tuple(item.auxiliary[key] for item in items))
                for key in auxiliary_keys
            },
        )

    def select(
        self,
        proposed: "PosteriorEvaluation",
        accepted: Array,
    ) -> "PosteriorEvaluation":
        """Select proposed values for accepted particles."""
        if self.auxiliary.keys() != proposed.auxiliary.keys():
            raise ValueError(
                "Posterior evaluations have different auxiliary fields."
            )
        if (self.log_proposal is None) != (proposed.log_proposal is None):
            raise ValueError("Posterior evaluations disagree on log_proposal.")

        def choose(current: Array, candidate: Array) -> Array:
            xp = namespace(current, candidate, accepted)
            mask = accepted
            if current.ndim > 1:
                mask = xp.reshape(
                    accepted,
                    (accepted.shape[0],) + (1,) * (current.ndim - 1),
                )
            return xp.where(mask, candidate, current)

        return PosteriorEvaluation(
            log_prior=choose(self.log_prior, proposed.log_prior),
            log_likelihood=choose(
                self.log_likelihood, proposed.log_likelihood
            ),
            log_proposal=(
                None
                if self.log_proposal is None
                else choose(self.log_proposal, proposed.log_proposal)
            ),
            auxiliary={
                key: choose(value, proposed.auxiliary[key])
                for key, value in self.auxiliary.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_prior": self.log_prior,
            "log_likelihood": self.log_likelihood,
            "log_proposal": self.log_proposal,
            "auxiliary": dict(self.auxiliary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PosteriorEvaluation":
        return cls(
            log_prior=payload["log_prior"],
            log_likelihood=payload["log_likelihood"],
            log_proposal=payload.get("log_proposal"),
            auxiliary=dict(payload.get("auxiliary", {})),
        )


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


@dataclass(frozen=True)
class TemperedPosterior:
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
        evaluation: PosteriorEvaluation | None = None,
        log_proposal_override: LogProposalLike | None = None,
    ) -> float | Any:
        """Return the log-weight increment required by the tempering step.

        When ``evaluation`` is omitted, all posterior components are evaluated
        from ``particles``. Supplying a matching evaluation avoids recomputing
        those components.
        """
        if evaluation is None:
            evaluation = self.evaluate(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )
        evaluation.validate_for(particles)
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal is None:
            if particles.ndim == 1:
                return float(evaluation.log_likelihood[0])
            return evaluation.log_likelihood
        if evaluation.log_proposal is None:
            raise ValueError("Evaluation does not contain log_proposal.")
        bridge = (
            evaluation.log_prior
            + evaluation.log_likelihood
            - evaluation.log_proposal
        )
        if particles.ndim == 1:
            return float(bridge[0])
        return bridge

    def log_prob(
        self,
        particles: Array,
        context: "StepContext",
        *,
        evaluation: PosteriorEvaluation | None = None,
        log_proposal_override: LogProposalLike | None = None,
    ) -> float | Any:
        """Return the tempered log posterior evaluated at the current β.

        When ``evaluation`` is omitted, all posterior components are evaluated
        from ``particles``. Supplying a matching evaluation avoids recomputing
        those components.
        """
        if evaluation is None:
            evaluation = self.evaluate(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )
        evaluation.validate_for(particles)
        beta = float(context.metadata.get("temperature", 0.0))
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal is None:
            posterior = evaluation.log_prior + beta * evaluation.log_likelihood
        else:
            if evaluation.log_proposal is None:
                raise ValueError("Evaluation does not contain log_proposal.")
            if beta == 0.0:
                posterior = evaluation.log_proposal
            elif beta == 1.0:
                posterior = evaluation.log_prior + evaluation.log_likelihood
            else:
                posterior = (1.0 - beta) * evaluation.log_proposal + beta * (
                    evaluation.log_prior + evaluation.log_likelihood
                )
        if particles.ndim == 1:
            return float(posterior[0])
        return posterior

    def evaluate(
        self,
        particles: Array,
        context: "StepContext",
        *,
        log_proposal_override: LogProposalLike | None = None,
        log_proposal_values: Array | None = None,
    ) -> PosteriorEvaluation:
        """Evaluate every posterior component for a particle population."""
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal_values is not None and log_proposal is None:
            raise ValueError(
                "log_proposal_values were supplied without a proposal."
            )
        return PosteriorEvaluation(
            log_prior=_evaluate(self.log_prior, particles, context),
            log_likelihood=_evaluate(self.log_likelihood, particles, context),
            log_proposal=(
                log_proposal_values
                if log_proposal_values is not None
                else (
                    None
                    if log_proposal is None
                    else _evaluate(log_proposal, particles, context)
                )
            ),
        )

    def refresh(
        self,
        particles: Array,
        evaluation: PosteriorEvaluation,
        context: "StepContext",
        *,
        components: Collection[str],
        log_proposal_override: LogProposalLike | None = None,
    ) -> PosteriorEvaluation:
        """Reevaluate selected components for unchanged particles."""
        evaluation.validate_for(particles)
        allowed = {"log_prior", "log_likelihood", "log_proposal"}
        unknown = set(components).difference(allowed)
        if unknown:
            raise ValueError(
                f"Unknown evaluation components: {sorted(unknown)}"
            )
        log_prior = evaluation.log_prior
        log_likelihood = evaluation.log_likelihood
        log_proposal = evaluation.log_proposal
        if "log_prior" in components:
            log_prior = _evaluate(self.log_prior, particles, context)
        if "log_likelihood" in components:
            log_likelihood = _evaluate(self.log_likelihood, particles, context)
        if "log_proposal" in components:
            proposal_fn = self.resolve_log_proposal(log_proposal_override)
            log_proposal = (
                None
                if proposal_fn is None
                else _evaluate(proposal_fn, particles, context)
            )
        return PosteriorEvaluation(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            log_proposal=log_proposal,
            auxiliary=evaluation.auxiliary,
        )

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
            evaluation = PosteriorEvaluation(
                log_prior=_evaluate(log_prior, particles, frozen_context),
                log_likelihood=_evaluate(
                    log_likelihood, particles, frozen_context
                ),
                log_proposal=(
                    None
                    if log_proposal is None
                    else _evaluate(log_proposal, particles, frozen_context)
                ),
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
        posterior=posterior,
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
