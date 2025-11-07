from __future__ import annotations

from typing import Any, Literal

try:  # pragma: no cover - typing helper only
    from array_api_compat.typing import Array
except Exception:  # pragma: no cover - optional dependency missing
    Array = Any  # type: ignore

MutationStrategy = Literal["emcee", "minipcn", "none"]

__all__ = ["Array"]
