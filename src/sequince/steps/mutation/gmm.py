"""Approximate direct rejuvenation using an injected Gaussian-mixture proposal."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from array_api_compat import device as get_device

from ...array_api import asarray, namespace
from ...target import SequentialTarget
from ...transforms import ParticleTransform
from ..base import SMCStep
from ..context import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState


class GaussianMixtureLike(Protocol):
    n_components: int
    converged: bool
    n_iter: int

    def fit(self, particles: Any, context: Any = None): ...

    def sample(
        self,
        n_samples: int,
        context: Any = None,
        *,
        rng: Any = None,
    ): ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class GaussianMixtureRejuvenation(SMCStep):
    """Refit and directly redraw from an injected Gaussian-mixture proposal.

    This is an approximate rejuvenation step inspired by *Sequential Monte
    Carlo with Gaussian Mixtures*, arXiv:2503.16028. Direct GMM samples are not,
    in general, exact samples from the current SMC target. Unlike an MCMC
    mutation kernel, this step does not preserve the target exactly and has no
    Metropolis-Hastings correction.

    A supplied transform defines the latent fitting space. Draws are mapped
    back to physical coordinates before the target evaluation is refreshed.
    """

    def __init__(
        self,
        target: SequentialTarget,
        proposal: GaussianMixtureLike,
        *,
        transform: ParticleTransform | None = None,
    ) -> None:
        if not isinstance(target, SequentialTarget):
            raise TypeError(
                "GaussianMixtureRejuvenation requires a SequentialTarget."
            )
        for method in ("fit", "sample", "state_dict", "load_state_dict"):
            if not callable(getattr(proposal, method, None)):
                raise TypeError(
                    "proposal must implement fit(), sample(), state_dict(), "
                    "and load_state_dict()."
                )
        self._target = target
        self._proposal = proposal
        self._transform = transform

    @property
    def proposal(self) -> GaussianMixtureLike:
        """The proposal fitted by the most recent application."""
        return self._proposal

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        xp = namespace(state.particles)
        physical = asarray(state.particles, xp=xp, like=state.particles)
        fitted_transform = None
        latent = physical
        if self._transform is not None:
            fitted_transform = self._transform.fit(physical, context=context)
            latent, _ = fitted_transform.forward(physical, context=context)
            latent = asarray(latent, xp=xp, like=physical)

        fitted_proposal = self._proposal.fit(latent, context=context)
        if fitted_proposal is not None:
            self._proposal = fitted_proposal
        new_latent = self._proposal.sample(
            physical.shape[0], context=context, rng=rng
        )
        if fitted_transform is None:
            new_particles = new_latent
        else:
            new_particles, _ = fitted_transform.inverse(
                new_latent, context=context
            )
        new_particles = asarray(new_particles, xp=xp, like=physical)

        evaluation = self._target.prepare_evaluation(
            new_particles, context, evaluation=None
        )
        n_particles = new_particles.shape[0]
        log_weights = xp.full(
            (n_particles,),
            -math.log(n_particles),
            dtype=state.log_weights.dtype,
            device=get_device(state.log_weights),
        )
        updated = state.replace_particles(
            new_particles,
            evaluation=evaluation,
            log_weights=log_weights,
        )
        output = {
            "n_components": self._proposal.n_components,
            "converged": self._proposal.converged,
            "n_iter": self._proposal.n_iter,
            "approximate": True,
        }
        return updated, context, output

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, Any]:
        del state, context, random_state
        raise NotImplementedError(
            "GaussianMixtureRejuvenation uses eager fitted proposals and is "
            "not supported by the functional/JIT API."
        )

    def state_dict(self) -> dict[str, Any]:
        return {"proposal": dict(self._proposal.state_dict())}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        proposal_state = state.get("proposal", {})
        if not isinstance(proposal_state, Mapping):
            raise TypeError(
                "GMM rejuvenation proposal state must be a mapping."
            )
        self._proposal.load_state_dict(proposal_state)


__all__ = ["GaussianMixtureLike", "GaussianMixtureRejuvenation"]
