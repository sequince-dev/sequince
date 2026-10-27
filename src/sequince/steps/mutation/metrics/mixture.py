"""Per-component mixture movement totals in mutation coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from array_api_compat import device as get_device

from ....array_api import asarray, namespace
from ....typing import Array
from ...context import StepContext
from .esjd import _squared_jump_distances
from .state import MutationTransition


@dataclass(frozen=True)
class MixtureComponentMetrics:
    """Aggregate attempts, acceptances and accepted squared jumps per child.

    Parameters
    ----------
    n_components : int
        Number of top-level mixture children. Rejections contribute zero jump.
    name : str
        Metric output key, default ``mixture_components``.
    """

    needs_transition: ClassVar[bool] = True

    n_components: int
    name: str = "mixture_components"

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, int)
            or self.n_components < 1
        ):
            raise ValueError("n_components must be a positive integer.")

    def initialize(
        self,
        particles: Array,
        log_weights: Array,
        context: StepContext,
    ) -> tuple[Array, Array, Array]:
        xp = namespace(particles)
        shape = (self.n_components,)
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
        component = auxiliary["mixture_component"]
        ids = asarray(tuple(range(self.n_components)), xp=xp, like=component)
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
                tuple(range(self.n_components)), xp=xp, like=attempted
            ),
            "n_attempted": attempted,
            "n_accepted": accepted,
            "acceptance": accepted / denominator,
            "squared_jump_distance": jump_sum,
            "esjd": jump_sum / denominator,
        }
