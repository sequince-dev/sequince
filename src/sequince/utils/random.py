from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from orng import RandomGenerator

RNGLike: TypeAlias = RandomGenerator | np.random.Generator | int | None


def resolve_rng(
    rng: RNGLike = None,
    *,
    backend: str | None = None,
    xp: Any | None = None,
) -> RandomGenerator:
    """Return an ORNG generator for the supplied RNG-like value.

    Parameters
    ----------
    rng
        An RNG-like value, which can be an ORNG RandomGenerator, a NumPy
        Generator, an integer seed, or None (for a default generator).
    backend
        The backend to use for the ORNG RandomGenerator if `rng` is None or an
        integer seed. Ignored if `rng` is an ORNG RandomGenerator or a NumPy
        Generator.
    xp
        An array API namespace to use for the ORNG RandomGenerator if `rng` is
        None or an integer seed. Ignored if `rng` is an ORNG RandomGenerator or
        a NumPy Generator.

    Returns
    -------
    RandomGenerator
        An ORNG RandomGenerator instance corresponding to the provided `rng`
        value, using the specified `backend` or `xp` if necessary.

    Raises
    ------
    TypeError
        If `rng` is not an ORNG RandomGenerator, a NumPy Generator, an integer
        seed, or None.
    ValueError
        If both `backend` and `xp` are specified.
    """
    if isinstance(rng, RandomGenerator):
        return rng
    if isinstance(rng, np.random.Generator):
        return RandomGenerator(backend="numpy", generator=rng)
    if rng is not None and not isinstance(rng, int):
        raise TypeError(
            "rng must be an orng.RandomGenerator, numpy.random.Generator, "
            "an int seed, or None."
        )
    if backend is not None and xp is not None:
        raise ValueError("Specify either 'backend' or 'xp', not both.")
    if xp is not None:
        return RandomGenerator.from_xp(xp, seed=rng)
    return RandomGenerator(backend=backend or "numpy", seed=rng)
