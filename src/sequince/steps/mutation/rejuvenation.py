"""Approximate direct rejuvenation using an injected generative proposal."""

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


class GenerativeProposal(Protocol):
    """Minimal fitted-proposal surface required for direct rejuvenation."""

    def fit(self, particles: Any, context: Any = None): ...

    def sample(
        self,
        n_samples: int,
        context: Any = None,
        *,
        rng: Any = None,
    ): ...


class ProposalRejuvenation(SMCStep):
    """Refit and directly redraw from an injected generative proposal.

    Direct proposal draws are not, in general, exact samples from the current
    SMC target. Unlike an MCMC mutation kernel, this step does not preserve the
    target exactly and has no Metropolis-Hastings correction.

    A supplied transform defines the latent fitting space. Draws are mapped
    back to physical coordinates before the target evaluation is refreshed.
    """

    def __init__(
        self,
        target: SequentialTarget,
        proposal: GenerativeProposal,
        *,
        transform: ParticleTransform | None = None,
    ) -> None:
        if not isinstance(target, SequentialTarget):
            raise TypeError(
                "ProposalRejuvenation requires a SequentialTarget."
            )
        for method in ("fit", "sample"):
            if not callable(getattr(proposal, method, None)):
                raise TypeError(
                    "proposal must implement fit() and RNG-aware sample()."
                )
        self._target = target
        self._proposal = proposal
        self._transform = transform

    @property
    def proposal(self) -> GenerativeProposal:
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
        output: dict[str, Any] = {
            "proposal": type(self._proposal).__name__,
            "approximate": True,
        }
        diagnostics = getattr(self._proposal, "diagnostics", None)
        if callable(diagnostics):
            diagnostics = diagnostics()
        if diagnostics is not None:
            if not isinstance(diagnostics, Mapping):
                raise TypeError("Proposal diagnostics must be a mapping.")
            output.update(diagnostics)
            output["approximate"] = True
        return updated, context, output

    def apply_functional(
        self,
        state: ParticleState,
        context: StepContext,
        random_state: Any,
    ) -> tuple[ParticleState, StepContext, Any, Any]:
        del state, context, random_state
        raise NotImplementedError(
            "ProposalRejuvenation uses eager fitted proposals and is "
            "not supported by the functional/JIT API."
        )

    def state_dict(self) -> dict[str, Any]:
        state_fn = getattr(self._proposal, "state_dict", None)
        if not callable(state_fn):
            return {}
        proposal_state = state_fn()
        if not isinstance(proposal_state, Mapping):
            raise TypeError("Proposal state_dict() must return a mapping.")
        return {"proposal": dict(proposal_state)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "proposal" not in state:
            return
        proposal_state = state["proposal"]
        if not isinstance(proposal_state, Mapping):
            raise TypeError("Proposal rejuvenation state must be a mapping.")
        load_fn = getattr(self._proposal, "load_state_dict", None)
        if not callable(load_fn):
            if proposal_state:
                raise ValueError(
                    f"{type(self._proposal).__name__} does not support "
                    "checkpoint state."
                )
            return
        load_fn(proposal_state)


__all__ = ["GenerativeProposal", "ProposalRejuvenation"]
