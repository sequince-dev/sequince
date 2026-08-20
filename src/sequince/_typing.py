from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Mapping, MutableMapping, TypeAlias

try:  # pragma: no cover - typing helper only
    from array_api_compat.typing import Array
except Exception:  # pragma: no cover - optional dependency missing
    Array = Any  # type: ignore

ArrayLike: TypeAlias = Array | Sequence[float] | float

MutationStrategy = Literal[
    "blackjax",
    "emcee",
    "metropolis-hastings",
    "minipcn",
    "none",
]

MetaData = MutableMapping[str, Any]
SharedContext = Mapping[str, Any]
Outputs = MutableMapping[str, Any]

__all__ = ["Array", "ArrayLike"]
