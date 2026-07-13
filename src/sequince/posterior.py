from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ._array_api import asarray, full, namespace
from ._typing import Array
from .proposals import ProposalDistribution
from .state import ParticleState
from .steps.context import StepContext
from .transforms import ParticleTransform, get_prebuilt_transform
from .utils import call_with_optional_context

if TYPE_CHECKING:
    from orng import ArrayRNG

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
        array = full(
            (particles.shape[0],),
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
        log_proposal_override: LogProposalLike | None = None,
    ) -> float | Any:
        """Return the log-weight increment required by the tempering step."""
        log_like = _evaluate(self.log_likelihood, particles, context)
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal is None:
            if particles.ndim == 1:
                return float(log_like[0])
            return log_like
        log_prior = _evaluate(self.log_prior, particles, context)
        log_prop = _evaluate(log_proposal, particles, context)
        bridge = log_prior + log_like - log_prop
        if particles.ndim == 1:
            return float(bridge[0])
        return bridge

    def log_prob(
        self,
        particles: Array,
        context: "StepContext",
        *,
        log_proposal_override: LogProposalLike | None = None,
    ) -> float | Any:
        """Return the tempered log posterior evaluated at the current β."""
        beta = float(context.metadata.get("temperature", 0.0))
        log_prior = _evaluate(self.log_prior, particles, context)
        log_like = _evaluate(self.log_likelihood, particles, context)
        log_proposal = self.resolve_log_proposal(log_proposal_override)
        if log_proposal is None:
            posterior = log_prior + beta * log_like
        else:
            log_prop = _evaluate(log_proposal, particles, context)
            if beta == 0.0:
                posterior = log_prop
            elif beta == 1.0:
                posterior = log_prior + log_like
            else:
                posterior = (1.0 - beta) * log_prop + beta * (
                    log_prior + log_like
                )
        if particles.ndim == 1:
            return float(posterior[0])
        return posterior

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
            beta = float(frozen_context.metadata.get("temperature", 0.0))
            frozen_log_prior = _evaluate(log_prior, particles, frozen_context)
            frozen_log_like = _evaluate(
                log_likelihood,
                particles,
                frozen_context,
            )
            if log_proposal is None:
                posterior = frozen_log_prior + beta * frozen_log_like
            else:
                frozen_log_prop = _evaluate(
                    log_proposal,
                    particles,
                    frozen_context,
                )
                if beta == 0.0:
                    posterior = frozen_log_prop
                elif beta == 1.0:
                    posterior = frozen_log_prior + frozen_log_like
                else:
                    posterior = (1.0 - beta) * frozen_log_prop + beta * (
                        frozen_log_prior + frozen_log_like
                    )
            if particles.ndim == 1:
                return float(posterior[0])
            return posterior

        return _frozen_log_prob


def draw_additional_samples(
    *,
    log_prior: Callable[..., Any] | None = None,
    log_likelihood: Callable[..., Any] | None = None,
    log_proposal: Callable[..., Any] | ProposalDistribution | None = None,
    posterior: TemperedPosterior | None = None,
    result: SMCResult,
    rng: "ArrayRNG",
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
