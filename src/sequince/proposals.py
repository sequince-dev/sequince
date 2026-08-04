from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ._array_api import asarray, full, namespace, to_numpy
from ._typing import Array
from .flows import (
    NormalizingFlow,
    clone_flow,
    flow_asarray,
    flow_state_dict,
    load_flow_state_dict,
)

if TYPE_CHECKING:
    from .posterior import TemperedPosterior
    from .state import ParticleState
    from .steps.context import StepContext

logger = logging.getLogger(__name__)


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
    ) -> Array:
        """Draw samples from the proposal distribution."""

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
    ) -> tuple[Array, Array]:
        """Draw samples and return their log proposal densities."""


class FlowProposal(ProposalDistribution):
    """Adapt a normalizing flow to the proposal-distribution interface."""

    def __init__(self, flow: NormalizingFlow):
        self._flow = flow

    @property
    def flow(self) -> NormalizingFlow:
        return self._flow

    def clone(self) -> "FlowProposal":
        return FlowProposal(clone_flow(self._flow))

    def fit(
        self,
        particles: Array,
        context: StepContext | None = None,
    ) -> "FlowProposal":
        logger.info("Fitting flow proposal to %d particles", len(particles))
        backend_particles = flow_asarray(self._flow, particles)
        self._flow.fit(backend_particles, context=context)
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
    ) -> Array:
        return self._flow.sample(n_samples, context=context)

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
    ) -> tuple[Array, Array]:
        flow_sample_and_log_prob = getattr(
            self._flow, "sample_and_log_prob", None
        )
        if callable(flow_sample_and_log_prob):
            return flow_sample_and_log_prob(
                n_samples,
                context=context,
            )
        samples = self.sample(n_samples, context=context)
        return samples, self.log_prob(samples, context=context)


def draw_initial_state(
    proposal: ProposalDistribution,
    n_particles: int,
    *,
    posterior: TemperedPosterior | None = None,
    context: StepContext | None = None,
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
    particles, log_proposal = proposal.sample_and_log_prob(
        n_particles,
        context=evaluation_context,
    )
    xp = namespace(particles, log_proposal)
    particles = asarray(particles, xp=xp)
    if particles.ndim == 0 or particles.shape[0] != n_particles:
        raise ValueError(
            "Proposal must return exactly one particle per requested sample."
        )
    if not bool(to_numpy(xp.all(xp.isfinite(particles)))):
        raise ValueError("Proposal returned non-finite particles.")

    log_proposal = asarray(log_proposal, xp=xp, like=particles)
    if log_proposal.ndim == 0:
        log_proposal = full(
            (n_particles,),
            float(log_proposal),
            xp=xp,
            dtype=log_proposal.dtype,
            like=particles,
        )
    if log_proposal.shape != (n_particles,):
        raise ValueError(
            "Proposal must return one log density per requested sample."
        )
    if not bool(to_numpy(xp.all(xp.isfinite(log_proposal)))):
        raise ValueError("Proposal returned non-finite log densities.")

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
            bool(to_numpy(xp.any(xp.isnan(values))))
            for values in evaluation.components.values()
        ):
            raise ValueError("Posterior evaluation returned NaN values.")

    return ParticleState(particles=particles, evaluation=evaluation)
