from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from array_api_compat import is_array_api_obj

from ..array_api import copy as copy_array
from ..array_api import namespace


def array_safe_copy(value: Any) -> Any:
    """Recursively copy a value while preserving array backends.

    Array leaves are copied using :func:`sequince.array_api.copy`. This avoids
    calling :func:`copy.deepcopy` on backend arrays, which is unsupported for
    some arrays such as non-leaf PyTorch tensors. Mappings, lists, and tuples
    are copied recursively; other values use :func:`copy.deepcopy`.

    Parameters
    ----------
    value
        The value to copy.

    Returns
    -------
    Any
        An independent copy of ``value`` where supported by its array backend.
    """
    if is_array_api_obj(value):
        return copy_array(value, xp=namespace(value))
    if isinstance(value, Mapping):
        return {key: array_safe_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [array_safe_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(array_safe_copy(item) for item in value)
    return copy.deepcopy(value)
