from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ._typing import Array
from .flows import NormalizingFlow, clone_flow, flow_asarray

if TYPE_CHECKING:
    from .steps.context import StepContext


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
        backend_particles = flow_asarray(self._flow, particles)
        self._flow.fit(backend_particles, context=context)
        return self

    def log_prob(
        self,
        particles: Array,
        context: StepContext | None = None,
    ) -> Array:
        backend_particles = flow_asarray(self._flow, particles)
        return self._flow.log_prob(backend_particles, context=context)

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
            return flow_sample_and_log_prob(n_samples, context=context)
        samples = self.sample(n_samples, context=context)
        return samples, self.log_prob(samples, context=context)
