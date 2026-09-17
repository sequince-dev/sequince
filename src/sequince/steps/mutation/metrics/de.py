"""Per-component DE movement totals in mutation coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from ...._typing import Array
from ....array_api import asarray, get_device, namespace
from ...context import StepContext
from .esjd import _squared_jump_distances
from .state import MutationTransition


@dataclass(frozen=True)
class DEScaleMetrics:
    """Aggregate DE proposals, with the final bin reserved for unit moves.

    Jump sums include zero displacement for rejection. Dividing by attempts
    gives ESJD in the coordinates used for mutation (latent if transformed).
    The squared-jump calculation is shared with ESJDMetric.
    No histories or additional target evaluations are retained.
    """

    multipliers: tuple[float, ...]
    name: str = "de_scales"

    def initialize(
        self,
        particles: Array,
        log_weights: Array,
        context: StepContext,
    ) -> tuple[Array, Array, Array]:
        xp = namespace(particles)
        shape = (len(self.multipliers) + 1,)
        counts = xp.zeros(
            shape, dtype=xp.asarray(0).dtype, device=get_device(particles)
        )
        jumps = xp.zeros(
            shape, dtype=particles.dtype, device=get_device(particles)
        )
        return counts, counts, jumps

    def update(
        self,
        state: tuple[Array, Array, Array],
        particles: Array,
        accepted: Array,
        context: StepContext,
        *,
        transition: MutationTransition | None = None,
    ) -> tuple[Array, Array, Array]:
        if transition is None:
            return state
        displacement = transition.displacement
        auxiliary = transition.kernel_auxiliary
        xp = namespace(displacement)
        component = auxiliary["scale_component"]
        ids = asarray(
            (*range(len(self.multipliers)), -1), xp=xp, like=component
        )
        selected = component[:, None] == ids[None, :]
        accepted_selected = selected & accepted[:, None]
        jumps = _squared_jump_distances(displacement, accepted)
        return (
            state[0] + xp.sum(xp.astype(selected, state[0].dtype), axis=0),
            state[1]
            + xp.sum(xp.astype(accepted_selected, state[1].dtype), axis=0),
            state[2] + xp.sum(xp.where(selected, jumps[:, None], 0.0), axis=0),
        )

    def result(
        self,
        state: tuple[Array, Array, Array],
        context: StepContext,
    ) -> dict[str, Array]:
        attempted, accepted, jump_sum = state
        xp = namespace(jump_sum)
        denominator = xp.maximum(attempted, xp.ones_like(attempted))
        return {
            "component": asarray(
                (*range(len(self.multipliers)), -1), xp=xp, like=attempted
            ),
            "multiplier": asarray(
                (*self.multipliers, 1.0),
                xp=xp,
                like=jump_sum,
                dtype=jump_sum.dtype,
            ),
            "n_attempted": attempted,
            "n_accepted": accepted,
            "acceptance": accepted / denominator,
            "squared_jump_distance": jump_sum,
            "esjd": jump_sum / denominator,
        }
