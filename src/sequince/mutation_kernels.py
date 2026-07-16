from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from ._array_api import arange, asarray, full, namespace, to_numpy
from ._typing import Array

if TYPE_CHECKING:
    from orng import ArrayRNG
else:  # pragma: no cover - runtime fallback
    from orng import ArrayRNG  # type: ignore


class MutationKernel(Protocol):
    """Protocol defining the interface required by mutation kernels."""

    def sample(
        self,
        particles: Array,
        rng: ArrayRNG,
    ) -> Array:
        """Draw proposal particles corresponding to the supplied parents."""

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        """Return ``log q(from | to) - log q(to | from)`` per particle."""


class SymmetricMutationKernel:
    """Base class for mutation kernels with symmetric transition densities."""

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        xp = namespace(from_particles, to_particles)
        start = asarray(from_particles, xp=xp)
        end = asarray(to_particles, xp=xp)
        if start.shape != end.shape:
            raise ValueError(
                "Particles passed to log_transition_ratio must share shape."
            )
        return full(
            (start.shape[0],),
            0.0,
            xp=xp,
            dtype=start.dtype,
            like=start,
        )


ProposalFn = Callable[[Array, ArrayRNG], Array]
LogTransitionRatioFn = Callable[[Array, Array], Array]
LogDensityFn = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class CallableMutationKernel:
    """Adapt callables to the mutation-kernel interface.

    Exactly one of ``symmetric``, ``log_transition_ratio_fn``, or
    ``log_density_fn`` must describe the transition correction.
    """

    proposal_fn: ProposalFn
    symmetric: bool = False
    log_transition_ratio_fn: LogTransitionRatioFn | None = None
    log_density_fn: LogDensityFn | None = None

    def __post_init__(self) -> None:
        corrections = sum(
            (
                self.symmetric,
                self.log_transition_ratio_fn is not None,
                self.log_density_fn is not None,
            )
        )
        if corrections != 1:
            raise ValueError(
                "Specify exactly one of symmetric=True, "
                "log_transition_ratio_fn, or log_density_fn."
            )

    def sample(self, particles: Array, rng: ArrayRNG) -> Array:
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        return asarray(self.proposal_fn(arr, rng), xp=xp, like=arr)

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        xp = namespace(from_particles, to_particles)
        start = asarray(from_particles, xp=xp)
        end = asarray(to_particles, xp=xp)
        if start.shape != end.shape:
            raise ValueError(
                "Particles passed to log_transition_ratio must share shape."
            )
        if self.symmetric:
            return full(
                (start.shape[0],),
                0.0,
                xp=xp,
                dtype=start.dtype,
                like=start,
            )
        if self.log_transition_ratio_fn is not None:
            values = self.log_transition_ratio_fn(start, end)
        elif self.log_density_fn is not None:
            values = self.log_density_fn(end, start) - self.log_density_fn(
                start, end
            )
        else:  # pragma: no cover - guarded by __post_init__
            raise RuntimeError("Mutation kernel has no transition correction.")
        return asarray(values, xp=xp, like=start)


def evaluate_log_transition_ratio(
    kernel: object,
    from_particles: Array,
    to_particles: Array,
) -> Array:
    """Evaluate a kernel transition ratio, including legacy density kernels."""
    ratio_fn = getattr(kernel, "log_transition_ratio", None)
    if callable(ratio_fn):
        return ratio_fn(from_particles, to_particles)

    density_fn = getattr(kernel, "log_density", None)
    if callable(density_fn):
        return density_fn(to_particles, from_particles) - density_fn(
            from_particles, to_particles
        )
    raise TypeError(
        "Mutation kernel must implement log_transition_ratio() or log_density()."
    )


@dataclass(frozen=True)
class GaussianRandomWalk(SymmetricMutationKernel):
    """Isotropic Gaussian random-walk mutation kernel."""

    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("GaussianRandomWalk.scale must be positive.")

    def sample(
        self,
        particles: Array,
        rng: ArrayRNG,
    ) -> Array:
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        noise = asarray(
            rng.normal(
                loc=0.0,
                scale=self.scale,
                size=arr.shape,
            ),
            xp=xp,
            like=arr,
        )
        return arr + noise

    def log_density(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        xp = namespace(from_particles, to_particles)
        start = asarray(from_particles, xp=xp)
        end = asarray(to_particles, xp=xp)
        if start.shape != end.shape:
            raise ValueError(
                "Particles passed to log_density must share shape."
            )
        diff = end - start
        if diff.ndim == 1:
            diff = xp.reshape(diff, (diff.shape[0], 1))
        dim = diff.shape[1]
        var = float(self.scale) ** 2
        norm_const = -0.5 * dim * math.log(2.0 * math.pi * var)
        exponent = -0.5 * xp.sum((diff**2) / var, axis=1)
        return exponent + norm_const


@dataclass(frozen=True)
class DifferentialEvolution(SymmetricMutationKernel):
    """Ensemble differential-evolution mutation kernel.

    Proposals have the form ``x + gamma * (x_a - x_b)``. The default scale
    mixture follows the differential-evolution proposal used by Bilby's
    dynesty ensemble walks.
    """

    scale: float | None = None
    mix: float = 0.5
    mode_hop_scale: float = 1.0
    gamma_shape: float = 4.0
    gamma_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.scale is not None and self.scale <= 0.0:
            raise ValueError("DifferentialEvolution.scale must be positive.")
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("DifferentialEvolution.mix must be in [0, 1].")
        if self.mode_hop_scale <= 0.0:
            raise ValueError(
                "DifferentialEvolution.mode_hop_scale must be positive."
            )
        if self.gamma_shape <= 0.0 or self.gamma_scale <= 0.0:
            raise ValueError(
                "DifferentialEvolution gamma parameters must be positive."
            )

    def sample(self, particles: Array, rng: ArrayRNG) -> Array:
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        if arr.ndim not in (1, 2):
            raise ValueError(
                "DifferentialEvolution requires a 1-D or 2-D particle array."
            )

        n_particles = arr.shape[0]
        if n_particles < 3:
            raise ValueError(
                "DifferentialEvolution requires at least three particles."
            )

        working = arr
        squeeze_last = arr.ndim == 1
        if squeeze_last:
            working = xp.reshape(arr, (n_particles, 1))
        dimension = working.shape[1]

        donor_a_raw = asarray(
            rng.choice(n_particles - 1, size=n_particles, replace=True),
            xp=xp,
            like=working,
        )
        rows = arange(
            n_particles,
            xp=xp,
            dtype=donor_a_raw.dtype,
            like=working,
        )
        one = full(
            (n_particles,),
            1,
            xp=xp,
            dtype=donor_a_raw.dtype,
            like=working,
        )
        zero = one - one
        donor_a = donor_a_raw + xp.where(donor_a_raw >= rows, one, zero)

        donor_b_raw = asarray(
            rng.choice(n_particles - 2, size=n_particles, replace=True),
            xp=xp,
            like=working,
        )
        low = xp.minimum(rows, donor_a)
        high = xp.maximum(rows, donor_a)
        donor_b = donor_b_raw + xp.where(donor_b_raw >= low, one, zero)
        donor_b = donor_b + xp.where(donor_b >= high, one, zero)

        differences = working[donor_a] - working[donor_b]
        default_scale = 2.38 / math.sqrt(2.0 * dimension)
        local_scale = default_scale if self.scale is None else self.scale
        gamma_draws = asarray(
            rng.gamma(
                shape=self.gamma_shape,
                scale=self.gamma_scale,
                size=(n_particles, 1),
            ),
            xp=xp,
            like=working,
        )
        mix_draws = asarray(rng.random((n_particles, 1)), xp=xp, like=working)
        gamma = xp.where(
            mix_draws < self.mix,
            local_scale * gamma_draws,
            self.mode_hop_scale,
        )
        proposals = working + gamma * differences
        if squeeze_last:
            return proposals[:, 0]
        return proposals


@dataclass
class MixtureMutationKernel:
    """Mixture of mutation kernels.
    
    Randomly selects one component kernel for each proposal."""

    kernels: Sequence[MutationKernel]
    weights: Sequence[float] | None = None
    _selected_kernel: MutationKernel | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.kernels = tuple(self.kernels)
        if not self.kernels:
            raise ValueError(
                "MixtureMutationKernel requires at least one kernel."
            )
        for kernel in self.kernels:
            has_correction = hasattr(
                kernel, "log_transition_ratio"
            ) or hasattr(kernel, "log_density")
            if not hasattr(kernel, "sample") or not has_correction:
                raise TypeError(
                    "Mixture components must implement sample() and either "
                    "log_transition_ratio() or log_density()."
                )

        if self.weights is None:
            self.weights = tuple(1.0 / len(self.kernels) for _ in self.kernels)
            return
        if len(self.weights) != len(self.kernels):
            raise ValueError(
                "MixtureMutationKernel weights must match the number of kernels."
            )
        weights = tuple(float(weight) for weight in self.weights)
        if any(
            not math.isfinite(weight) or weight < 0.0 for weight in weights
        ):
            raise ValueError(
                "MixtureMutationKernel weights must be finite and non-negative."
            )
        total = sum(weights)
        if total <= 0.0:
            raise ValueError(
                "MixtureMutationKernel requires at least one positive weight."
            )
        self.weights = tuple(weight / total for weight in weights)

    @property
    def selected_kernel(self) -> MutationKernel | None:
        """Return the component selected by the latest call to ``sample``."""
        return self._selected_kernel

    def sample(self, particles: Array, rng: ArrayRNG) -> Array:
        selected_index = int(
            to_numpy(
                rng.choice(
                    len(self.kernels),
                    size=(),
                    p=self.weights,
                )
            )
        )
        self._selected_kernel = self.kernels[selected_index]
        return self._selected_kernel.sample(particles, rng)

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        if self._selected_kernel is None:
            raise RuntimeError(
                "MixtureMutationKernel.sample() must be called before "
                "log_transition_ratio()."
            )
        return evaluate_log_transition_ratio(
            self._selected_kernel,
            from_particles,
            to_particles,
        )


MUTATION_KERNELS: dict[str, type] = {
    "differential-evolution": DifferentialEvolution,
    "gaussian-random-walk": GaussianRandomWalk,
    "mixture": MixtureMutationKernel,
}


def get_mutation_kernel(name: str, **kwargs: Any) -> MutationKernel:
    """Construct a named mutation kernel."""
    try:
        kernel_cls = MUTATION_KERNELS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MUTATION_KERNELS))
        raise ValueError(
            f"Unknown mutation kernel '{name}'. Valid options are: {valid}."
        ) from exc
    if kernel_cls is MixtureMutationKernel:
        try:
            kernel_specs = kwargs.pop("kernels")
        except KeyError as exc:
            raise ValueError(
                "The mixture mutation kernel requires a 'kernels' option."
            ) from exc
        kwargs["kernels"] = [
            _resolve_mutation_kernel_spec(spec) for spec in kernel_specs
        ]
    return kernel_cls(**kwargs)


def _resolve_mutation_kernel_spec(spec: Any) -> MutationKernel:
    if isinstance(spec, str):
        return get_mutation_kernel(spec)
    if not isinstance(spec, Mapping):
        return spec

    config = dict(spec)
    try:
        name = config.pop("kernel")
    except KeyError as exc:
        raise ValueError(
            "Mixture component mappings require a 'kernel' option."
        ) from exc
    kernel_kwargs = dict(config.pop("kernel_kwargs", {}) or {})
    if config:
        raise ValueError(
            f"Unknown mixture component options: {', '.join(sorted(config))}."
        )
    return get_mutation_kernel(name, **kernel_kwargs)
