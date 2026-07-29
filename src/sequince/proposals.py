from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Protocol

from ._array_api import asarray, namespace
from ._typing import Array
from .flows import (
    NormalizingFlow,
    clone_flow,
    flow_asarray,
    flow_state_dict,
    load_flow_state_dict,
)
from .state import ParticleState
from .steps.context import StepContext

if TYPE_CHECKING:
    from .posterior import TemperedPosterior

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
    validation_fns: Sequence[Callable[[Array], Array]] = (),
    posterior: TemperedPosterior | None = None,
    context: StepContext | None = None,
    max_batches: int = 100,
) -> ParticleState:
    """Draw a finite initial particle population from a proposal.

    Proposal samples with non-finite proposal density or non-finite values from
    any validation function are discarded.
    """
    if n_particles < 1:
        raise ValueError("n_particles must be positive.")
    if max_batches < 1:
        raise ValueError("max_batches must be positive.")

    accepted: list[Array] = []
    accepted_evaluations = []
    evaluation_context = context or StepContext(t=0)
    n_accepted = 0
    for _ in range(max_batches):
        particles, log_proposal = proposal.sample_and_log_prob(
            n_particles,
            context=context,
        )
        xp = namespace(particles, log_proposal)
        particles = asarray(particles, xp=xp)
        log_proposal = asarray(log_proposal, xp=xp, like=particles)
        valid = xp.isfinite(log_proposal)
        evaluation = None
        if posterior is not None:
            evaluation = posterior.evaluate(
                particles,
                evaluation_context,
                log_proposal_values=log_proposal,
            )
            valid = (
                valid
                & xp.isfinite(evaluation.log_prior)
                & xp.isfinite(evaluation.log_likelihood)
            )
        else:
            for validation_fn in validation_fns:
                values = asarray(
                    validation_fn(particles),
                    xp=xp,
                    like=log_proposal,
                )
                valid = valid & xp.isfinite(values)
        if bool(xp.any(valid)):
            batch = particles[valid]
            accepted.append(batch)
            if evaluation is not None:
                accepted_evaluations.append(
                    evaluation.take(xp.nonzero(valid)[0])
                )
            n_accepted += batch.shape[0]
        if n_accepted >= n_particles:
            break

    if n_accepted < n_particles:
        raise RuntimeError(
            f"Could only draw {n_accepted} finite particles after "
            f"{max_batches} proposal batches."
        )
    xp = namespace(*accepted)
    particles = (
        accepted[0] if len(accepted) == 1 else xp.concat(accepted, axis=0)
    )
    initial_evaluation = None
    if accepted_evaluations:
        from .posterior import PosteriorEvaluation

        initial_evaluation = PosteriorEvaluation.concatenate(
            accepted_evaluations
        ).take(xp.arange(n_particles))
    return ParticleState(
        particles=particles[:n_particles],
        evaluation=initial_evaluation,
    )
