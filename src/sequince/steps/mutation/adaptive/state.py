"""Complete budget state shared by eager and functional adaptation."""

from dataclasses import dataclass
from typing import Generic, Self, TypeVar

from ....utils.pytrees import pytree_dataclass

_Scalar = TypeVar("_Scalar")


@pytree_dataclass(
    data_fields=(
        "current_iteration",
        "current_n_steps",
        "next_n_steps",
        "last_updated_iteration",
    )
)
@dataclass(frozen=True)
class MutationBudgetState(Generic[_Scalar]):
    """Immutable current and pending budgets with iteration markers.

    Eager fields are host integers; functional fields are scalar int32 arrays.
    Both budgets initially equal ``initial_steps``; unset iterations use -1."""

    current_iteration: _Scalar
    current_n_steps: _Scalar
    next_n_steps: _Scalar
    last_updated_iteration: _Scalar

    @classmethod
    def initial(cls, initial_steps: _Scalar, unset_iteration: _Scalar) -> Self:
        """Allocate both budget slots and mark both iterations as unset."""
        return cls(
            unset_iteration, initial_steps, initial_steps, unset_iteration
        )


__all__ = ["MutationBudgetState"]
