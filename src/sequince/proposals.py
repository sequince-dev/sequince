from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Self,
    runtime_checkable,
)

from array_api_compat import device as get_device

from ._typing import Array
from .array_api import asarray, namespace, to_numpy
from .flows import (
    NormalizingFlow,
    clone_flow,
    flow_asarray,
    flow_state_dict,
    load_flow_state_dict,
)
from .utils import call_with_optional_context, resolve_rng

if TYPE_CHECKING:
    from orng import RandomGenerator

    from .posterior import TemperedPosterior
    from .state import ParticleState
    from .steps.context import StepContext
    from .target import Target

logger = logging.getLogger(__name__)


@runtime_checkable
class ProposalDistribution(Protocol):
    """Protocol for proposal distributions used by bridged SMC targets."""

    def log_prob(
        self,
        particles: Array,
        context: StepContext | None = None,
    ) -> Array:
        """Evaluate the log proposal density in particle space."""

    def sample(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> Array:
        """Draw samples from the proposal distribution."""

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> tuple[Array, Array]:
        """Draw samples and return their log proposal densities."""


@runtime_checkable
class TrainableProposal(ProposalDistribution, Protocol):
    """Proposal distribution that can be fitted to a particle population."""

    def fit(
        self,
        particles: Array,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
    ) -> Self:
        """Fit the proposal and return the fitted proposal."""


class FlowProposal(TrainableProposal):
    """Adapt a normalizing flow to the proposal-distribution interface."""

    def __init__(
        self,
        flow: NormalizingFlow,
        *,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._flow = flow
        self._fit_kwargs = dict(fit_kwargs or {})
        reserved = {"context", "rng"}.intersection(self._fit_kwargs)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"fit_kwargs must not override proposal arguments: {names}."
            )

    @property
    def flow(self) -> NormalizingFlow:
        return self._flow

    def clone(self) -> "FlowProposal":
        return FlowProposal(
            clone_flow(self._flow),
            fit_kwargs=self._fit_kwargs,
        )

    def fit(
        self,
        particles: Array,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
    ) -> "FlowProposal":
        logger.info("Fitting flow proposal to %d particles", len(particles))
        backend_particles = flow_asarray(self._flow, particles)
        fitted_flow = self._flow.fit(
            backend_particles,
            context=context,
            rng=rng,
            **self._fit_kwargs,
        )
        if fitted_flow is not None:
            self._flow = fitted_flow
        return self

    def state_dict(self) -> dict[str, Any]:
        return {"flow": flow_state_dict(self._flow)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        flow_state = state.get("flow", {})
        if not isinstance(flow_state, Mapping):
            raise TypeError("FlowProposal state 'flow' must be a mapping.")
        load_flow_state_dict(self._flow, flow_state)

    def log_prob(
        self,
        particles: Array,
        context: StepContext | None = None,
    ) -> Array:
        xp = namespace(particles)
        backend_particles = flow_asarray(self._flow, particles)
        return asarray(
            self._flow.log_prob(backend_particles, context=context), xp=xp
        )

    def sample(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> Array:
        samples = self._flow.sample(
            n_samples,
            context=context,
            rng=rng,
        )
        if xp is not None:
            samples = asarray(samples, xp=xp)
        return samples

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ) -> tuple[Array, Array]:
        flow_sample_and_log_prob = getattr(
            self._flow, "sample_and_log_prob", None
        )
        if callable(flow_sample_and_log_prob):
            samples, log_prob = flow_sample_and_log_prob(
                n_samples,
                context=context,
                rng=rng,
            )
        else:
            samples = self.sample(
                n_samples,
                context=context,
                rng=rng,
                xp=xp,
            )
            log_prob = self.log_prob(samples, context=context)
        if xp is not None:
            samples = asarray(samples, xp=xp)
            log_prob = asarray(log_prob, xp=xp)
        return samples, log_prob


def _proposal_batch(
    proposal: ProposalDistribution,
    n_particles: int,
    context: StepContext,
    *,
    rng: RandomGenerator | int | None = None,
    xp: Any | None = None,
) -> tuple[Array, Array]:
    """Draw and validate one proposal batch."""
    particles, log_proposal = proposal.sample_and_log_prob(
        n_particles,
        context=context,
        rng=rng,
        xp=xp,
    )
    xp = namespace(particles, log_proposal)
    particles = asarray(particles, xp=xp)
    if particles.ndim == 0 or particles.shape[0] != n_particles:
        raise ValueError(
            "Proposal must return exactly one particle per requested sample."
        )
    if not bool(to_numpy(xp.all(xp.isfinite(particles)))):
        raise ValueError("Proposal returned non-finite particles.")

    log_proposal = _particle_values(
        log_proposal,
        particles,
        name="Proposal log densities",
    )
    if not bool(to_numpy(xp.all(xp.isfinite(log_proposal)))):
        raise ValueError("Proposal returned non-finite log densities.")
    return particles, log_proposal


def _particle_values(values: Any, particles: Array, *, name: str) -> Array:
    """Return one scalar value per particle."""
    xp = namespace(particles, values)
    array = asarray(values, xp=xp, like=particles)
    n_particles = particles.shape[0]
    if array.ndim == 0:
        array = xp.full(
            (n_particles,),
            float(array),
            dtype=array.dtype,
            device=get_device(particles),
        )
    if array.shape != (n_particles,):
        raise ValueError(f"{name} must contain one value per particle.")
    return array


def draw_initial_state(
    proposal: ProposalDistribution,
    n_particles: int,
    *,
    posterior: TemperedPosterior | None = None,
    context: StepContext | None = None,
    rng: RandomGenerator | int | None = None,
    xp: Any | None = None,
) -> "ParticleState":
    """Draw one initial population and retain its target evaluation.

    The proposal must return finite particles and log densities. Non-finite
    posterior densities are retained because discarding them would condition
    the proposal and invalidate evidence estimates. NaN target evaluations are
    rejected as evaluation errors.
    """
    from .state import ParticleState
    from .steps.context import StepContext
    from .target import TargetEvaluation

    if n_particles < 1:
        raise ValueError("n_particles must be positive.")

    evaluation_context = context or StepContext(t=0)
    particles, log_proposal = _proposal_batch(
        proposal,
        n_particles,
        evaluation_context,
        rng=rng,
        xp=xp,
    )
    xp = namespace(particles, log_proposal)

    evaluation = None
    if posterior is not None:
        evaluation = posterior.bind_evaluation(
            TargetEvaluation(components={"log_proposal": log_proposal}),
            evaluation_context,
        )
        evaluation = posterior.evaluate(
            particles,
            evaluation_context,
            components={"log_prior", "log_likelihood"},
            evaluation=evaluation,
        )
        if any(
            bool(to_numpy(xp.any(xp.isnan(values) | (values == float("inf")))))
            for values in evaluation.components.values()
        ):
            raise ValueError(
                "Posterior evaluation returned NaN or +inf values."
            )

    return ParticleState(particles=particles, evaluation=evaluation)


def draw_mcmc_initial_state(
    proposal: ProposalDistribution,
    n_particles: int,
    log_prob_fn: Callable[..., Array | float] | Target,
    *,
    context: StepContext | None = None,
    max_batches: int = 100,
    rng: RandomGenerator | int | None = None,
    xp: Any | None = None,
) -> "ParticleState":
    """Draw finite-target starting particles for standalone MCMC.

    Unlike :func:`draw_initial_state`, this function conditions the proposal by
    rejecting particles whose target log density is ``-inf``. It must not be
    used to initialize an evidence estimator.
    """
    from .state import ParticleState
    from .steps.context import StepContext
    from .target import Target, TargetEvaluation

    if n_particles < 1:
        raise ValueError("n_particles must be positive.")
    if max_batches < 1:
        raise ValueError("max_batches must be positive.")

    evaluation_context = context or StepContext(t=0)
    generator = resolve_rng(rng)
    accepted: list[Array] = []
    accepted_evaluations: list[TargetEvaluation] = []
    n_accepted = 0
    for _ in range(max_batches):
        particles, _ = _proposal_batch(
            proposal,
            n_particles,
            evaluation_context,
            rng=generator,
            xp=xp,
        )
        xp = namespace(particles)
        evaluation = None
        if isinstance(log_prob_fn, Target):
            evaluation = log_prob_fn.prepare_evaluation(
                particles,
                evaluation_context,
            )
            log_prob = log_prob_fn.log_prob(
                particles,
                evaluation_context,
                evaluation=evaluation,
            )
        else:
            log_prob = call_with_optional_context(
                log_prob_fn,
                (particles,),
                evaluation_context,
            )
        log_prob = _particle_values(
            log_prob,
            particles,
            name="Target log densities",
        )
        numerical_error = xp.isnan(log_prob) | (log_prob == float("inf"))
        if bool(to_numpy(xp.any(numerical_error))):
            raise ValueError(
                "Target log densities contain NaN or +inf values."
            )

        valid = xp.isfinite(log_prob)
        if bool(to_numpy(xp.any(valid))):
            indices = xp.nonzero(valid)[0]
            batch = particles[indices]
            accepted.append(batch)
            if evaluation is not None:
                accepted_evaluations.append(evaluation.take(indices))
            n_accepted += batch.shape[0]
        if n_accepted >= n_particles:
            break

    if n_accepted < n_particles:
        raise RuntimeError(
            f"Could only draw {n_accepted} finite-target particles after "
            f"{max_batches} proposal batches."
        )

    xp = namespace(*accepted)
    particles = (
        accepted[0] if len(accepted) == 1 else xp.concat(accepted, axis=0)
    )[:n_particles]
    evaluation = None
    if accepted_evaluations:
        evaluation = TargetEvaluation.concatenate(accepted_evaluations).take(
            xp.arange(n_particles, device=get_device(particles))
        )
    return ParticleState(particles=particles, evaluation=evaluation)
