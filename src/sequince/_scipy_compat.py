"""Process-wide opt-in for SciPy's experimental Array API support."""

from __future__ import annotations

import os
import warnings


def enable_scipy_array_api() -> None:
    """Enable SciPy's experimental Array API support before SciPy imports."""
    value = os.environ.get("SCIPY_ARRAY_API")
    if value is None:
        warnings.warn(
            "SCIPY_ARRAY_API is not set; Sequince is setting it to '1'.",
            ImportWarning,
            stacklevel=2,
        )
        os.environ["SCIPY_ARRAY_API"] = "1"
    elif value != "1":
        warnings.warn(
            f"SCIPY_ARRAY_API={value!r}; Sequince's Array API support "
            "requires SCIPY_ARRAY_API='1'.",
            RuntimeWarning,
            stacklevel=2,
        )
