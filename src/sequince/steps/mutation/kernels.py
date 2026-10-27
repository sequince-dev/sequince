from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, overload

import array_api_extra as xpx
import numpy as np
import optree
from array_api_compat import device as get_device

from ...array_api import asarray, namespace, to_numpy
from ...target import EvaluationCounts
from ...typing import Array
from ...utils.pytrees import pytree_dataclass
from ..base import StepContext

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...proposals import ProposalDistribution
else:  # pragma: no cover - runtime fallback
    from orng import RandomGenerator  # type: ignore


@pytree_dataclass(data_fields=("particles", "auxiliary"))
@dataclass(frozen=True)
class KernelSample:
    """Particles and proposal-local information for one kernel draw.

    Parameters
    ----------
    particles : array
        Proposed coordinates, with the same shape as the parent particles.
    auxiliary : pytree, optional
        Kernel-specific pytree data needed for transition corrections or
        diagnostics. DE stores arrays aligned with the active particles;
        mixtures store their selected component and its proposal state.
        Complementary kernels must use arrays with an active-particle leading
        dimension so the mutation can combine both half-updates.

    Notes
    -----
    Acceptance decisions and displacement belong to the mutation layer.
    Neither kernels nor consumers should mutate this state or its contents.
    """

    particles: Array
    auxiliary: Any = None


def _as_kernel_sample(proposal: Array | KernelSample) -> KernelSample:
    """Normalize a kernel result without copying particles or auxiliary."""
    return (
        proposal
        if isinstance(proposal, KernelSample)
        else KernelSample(proposal)
    )


class _KernelSampler(Protocol):
    """Protocol defining the interface required by mutation kernels."""

    def sample(
        self,
        particles: Array,
        rng: RandomGenerator,
    ) -> Array | KernelSample:
        """Draw particles, optionally with proposal-local auxiliary."""


class TransitionRatioKernel(_KernelSampler, Protocol):
    """Kernel with an array-based transition correction."""

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        """Return ``log q(from | to) - log q(to | from)`` per particle."""


class SampleAwareMutationKernel(_KernelSampler, Protocol):
    """Kernel whose correction consumes proposal-local state."""

    def log_transition_ratio_from_sample(
        self,
        from_particles: Array,
        sample: KernelSample,
    ) -> Array: ...


class DensityMutationKernel(_KernelSampler, Protocol):
    """Kernel exposing its transition log density."""

    def log_density(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array: ...


MutationKernel = (
    TransitionRatioKernel | SampleAwareMutationKernel | DensityMutationKernel
)


def _zero_log_transition_ratio(
    from_particles: Array,
    to_particles: Array,
) -> Array:
    """Return a zero transition correction after validating particle shapes."""
    xp = namespace(from_particles, to_particles)
    start = asarray(from_particles, xp=xp)
    end = asarray(to_particles, xp=xp)
    if start.shape != end.shape:
        raise ValueError(
            "Particles passed to log_transition_ratio must share shape."
        )
    return xp.full(
        (start.shape[0],),
        0.0,
        dtype=start.dtype,
        device=get_device(start),
    )


class SymmetricMutationKernel:
    """Base class for mutation kernels with symmetric transition densities."""

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
    ) -> Array:
        return _zero_log_transition_ratio(from_particles, to_particles)


class ComplementaryEnsembleKernel(ABC):
    """Kernel using a frozen complement and either correction hook.

    Auxiliary pytrees must match between half-updates. Every leaf must be an
    array with an active-particle leading dimension; None nodes are permitted.
    Treat auxiliary data as read-only.
    """

    @abstractmethod
    def sample(
        self,
        particles: Array,
        complementary_particles: Array,
        rng: RandomGenerator,
    ) -> Array | KernelSample:
        """Draw particles or proposal state using the frozen complement."""

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
        complementary_particles: Array,
    ) -> Array:
        """Return the array correction, or override the sample-aware hook."""
        raise NotImplementedError("Implement a transition correction hook.")


class SymmetricComplementaryEnsembleKernel(ComplementaryEnsembleKernel):
    """Base class for symmetric complementary-ensemble kernels."""

    def log_transition_ratio(
        self,
        from_particles: Array,
        to_particles: Array,
        complementary_particles: Array,
    ) -> Array:
        del complementary_particles
        return _zero_log_transition_ratio(from_particles, to_particles)


ProposalFn = Callable[[Array, RandomGenerator], Array]
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

    def sample(self, particles: Array, rng: RandomGenerator) -> Array:
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
            return xp.full(
                (start.shape[0],),
                0.0,
                dtype=start.dtype,
                device=get_device(start),
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


@overload
def evaluate_log_transition_ratio(
    kernel: object,
    from_particles: Array,
    proposal: Array | KernelSample,
    complementary_particles: Array | None = None,
    *,
    return_counts: Literal[False] = False,
) -> Array: ...


@overload
def evaluate_log_transition_ratio(
    kernel: object,
    from_particles: Array,
    proposal: Array | KernelSample,
    complementary_particles: Array | None = None,
    *,
    return_counts: Literal[True],
) -> tuple[Array, EvaluationCounts]: ...


@overload
def evaluate_log_transition_ratio(
    kernel: object,
    from_particles: Array,
    proposal: Array | KernelSample,
    complementary_particles: Array | None = None,
    *,
    return_counts: bool,
) -> Array | tuple[Array, EvaluationCounts]: ...


def evaluate_log_transition_ratio(
    kernel: object,
    from_particles: Array,
    proposal: Array | KernelSample,
    complementary_particles: Array | None = None,
    *,
    return_counts: bool = False,
) -> Array | tuple[Array, EvaluationCounts]:
    """Evaluate a transition ratio."""
    proposal = _as_kernel_sample(proposal)
    sample_ratio_fn = getattr(kernel, "log_transition_ratio_from_sample", None)
    ratio_fn = getattr(kernel, "log_transition_ratio", None)
    if callable(sample_ratio_fn) or callable(ratio_fn):
        correction = sample_ratio_fn if callable(sample_ratio_fn) else ratio_fn
        assert callable(correction)
        destination = (
            proposal if callable(sample_ratio_fn) else proposal.particles
        )
        if complementary_particles is None:
            values = correction(from_particles, destination)
        else:
            values = correction(
                from_particles, destination, complementary_particles
            )
        counts = EvaluationCounts.zeros(
            xp=namespace(from_particles, proposal.particles)
        )
        return (values, counts) if return_counts else values

    density_fn = getattr(kernel, "log_density", None)
    if callable(density_fn):
        if complementary_particles is not None:
            raise TypeError(
                "Complementary kernels must implement log_transition_ratio()."
            )
        values = density_fn(proposal.particles, from_particles) - density_fn(
            from_particles, proposal.particles
        )
        counts = EvaluationCounts.zeros(
            xp=namespace(from_particles, proposal.particles)
        )
        return (values, counts) if return_counts else values
    raise TypeError(
        "Mutation kernel must implement log_transition_ratio() or log_density()."
    )


@dataclass(frozen=True)
class IndependentProposalKernel:
    """Independence MH proposals in the coordinates supplied to the kernel.

    Parameters
    ----------
    proposal : ProposalDistribution
        Distribution defining samples and densities in mutation coordinates.
        A physical-space prior is suitable only without a coordinate transform.
        Sample batches must have exactly the input particle shape; densities
        must have shape (n_particles,). Sampled densities are retained locally.
    """

    proposal: ProposalDistribution

    def sample(self, particles: Array, rng: RandomGenerator) -> KernelSample:
        """Draw a batch and retain its log densities for the MH correction."""
        xp = namespace(particles)
        values, density = self.proposal.sample_and_log_prob(
            particles.shape[0], rng=rng, xp=xp
        )
        values = asarray(values, xp=xp, like=particles, dtype=particles.dtype)
        density = asarray(
            density, xp=xp, like=particles, dtype=particles.dtype
        )
        if values.shape != particles.shape:
            raise ValueError("Proposal samples must match the particle shape.")
        if density.shape != (particles.shape[0],):
            raise ValueError(
                "Proposal densities must have shape (n_particles,)."
            )
        return KernelSample(values, {"log_proposal": density})

    def log_transition_ratio(
        self, from_particles: Array, to_particles: Array
    ) -> Array:
        """Return ``log q(current) - log q(proposed)`` per particle."""
        if from_particles.shape != to_particles.shape:
            raise ValueError("Proposal samples must match the particle shape.")
        return self.log_transition_ratio_from_sample(
            from_particles,
            KernelSample(
                to_particles,
                {"log_proposal": self.proposal.log_prob(to_particles)},
            ),
        )

    def log_transition_ratio_from_sample(
        self, from_particles: Array, sample: KernelSample
    ) -> Array:
        """Evaluate the correction using the cached proposed log densities."""
        if from_particles.shape != sample.particles.shape:
            raise ValueError("Proposal samples must match the particle shape.")
        xp = namespace(from_particles)
        current = asarray(
            self.proposal.log_prob(from_particles),
            xp=xp,
            like=from_particles,
            dtype=from_particles.dtype,
        )
        proposed = asarray(
            sample.auxiliary["log_proposal"],
            xp=xp,
            like=from_particles,
            dtype=from_particles.dtype,
        )
        if current.shape != (from_particles.shape[0],) or (
            proposed.shape != current.shape
        ):
            raise ValueError(
                "Proposal densities must have shape (n_particles,)."
            )
        return current - proposed


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
        rng: RandomGenerator,
    ) -> Array:
        xp = namespace(particles)
        arr = asarray(particles, xp=xp)
        noise = rng.normal(
            loc=0.0,
            scale=self.scale,
            size=arr.shape,
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


GammaRandomization = Literal["none", "bilby"]
_GAMMA_RANDOMIZATIONS = ("none", "bilby")


def _sample_donor_indices(
    n_active: int,
    n_complementary: int,
    rng: RandomGenerator,
) -> tuple[Array, Array]:
    """Sample uniform ordered pairs of distinct complementary indices."""
    donor_j = rng.choice(n_complementary, size=(n_active,))
    donor_k_raw = rng.choice(n_complementary - 1, size=(n_active,))
    xp = namespace(donor_j, donor_k_raw)
    donor_k = xp.where(
        donor_k_raw >= donor_j,
        donor_k_raw + 1,
        donor_k_raw,
    )
    return donor_j, donor_k


def _sample_effective_gamma(
    n_active: int,
    gamma_base: float,
    gamma_one_probability: float,
    gamma_randomization: GammaRandomization,
    rng: RandomGenerator,
    *,
    like: Array,
) -> Array:
    """Sample one DE scale for each active proposal."""
    return _sample_gamma_and_unit_mask(
        n_active,
        gamma_base,
        gamma_one_probability,
        gamma_randomization,
        rng,
        like=like,
    )[0]


def _sample_gamma_and_unit_mask(
    n_active: int,
    gamma_base: float,
    gamma_one_probability: float,
    gamma_randomization: GammaRandomization,
    rng: RandomGenerator,
    *,
    like: Array,
) -> tuple[Array, Array]:
    """Preserve legacy RNG ordering while retaining explicit unit identity."""
    xp = namespace(like)
    shape = (n_active,)
    if gamma_one_probability == 1.0:
        return (
            xp.ones(shape, dtype=like.dtype, device=get_device(like)),
            xp.ones(shape, dtype=xp.bool, device=get_device(like)),
        )

    if gamma_randomization == "bilby":
        factors = rng.gamma(
            shape=4.0,
            scale=0.25,
            size=shape,
            dtype=like.dtype,
        )
        ordinary_gamma = gamma_base * factors
    else:
        ordinary_gamma = xp.full(
            shape,
            gamma_base,
            dtype=like.dtype,
            device=get_device(like),
        )

    if gamma_one_probability == 0.0:
        return ordinary_gamma, xp.zeros(
            shape, dtype=xp.bool, device=get_device(like)
        )

    use_gamma_one = rng.random(shape) < gamma_one_probability
    return xp.where(use_gamma_one, 1.0, ordinary_gamma), use_gamma_one


@dataclass(frozen=True)
class DifferentialEvolutionKernel(SymmetricComplementaryEnsembleKernel):
    """Differential-evolution proposals from a frozen complementary ensemble.

    ``"none"`` uses standard fixed-scale DE-MCMC. ``"bilby"`` multiplies the
    base scale by an independent ``Gamma(shape=4, scale=0.25)`` draw for each
    proposal, matching Bilby-style dynesty scaling. When ``gamma`` is omitted,
    the base scale is ``2.38 / sqrt(2 * n_dimensions)``.

    ``gamma_one_probability`` controls occasional unit-scale moves. A
    :class:`RedBlueMetropolisHastingsMutation` supplies the frozen complementary
    population for each half-update. Sequince still updates an interacting MCMC
    population and its complete algorithm is not dynesty.

    Parameters
    ----------
    gamma : float or None
        Nominal scale, defaulting to ``2.38 / sqrt(2 * n_dimensions)``.
    gamma_one_probability : float
        Probability of an explicit unit move, overriding all scale factors.
    gamma_randomization : {"none", "bilby"}
        Optional independent Gamma randomization of the nominal scale.
    scale_multipliers : tuple of float
        Nonempty bank of finite positive factors, sampled per active particle.
        Factors also multiply Bilby-randomized scales, but never unit moves.
    scale_weights : tuple of float or None
        Finite non-negative categorical weights, normalized on construction.
        None assigns equal weights. Configuration is immutable and selection
        is state-independent, so the global proposal remains symmetric.
        Use ``dataclasses.replace`` to construct weights for a later stage.
    """

    gamma: float | None = None
    gamma_one_probability: float = 0.1
    gamma_randomization: GammaRandomization = "none"
    scale_multipliers: tuple[float, ...] = (1.0,)
    scale_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        multipliers = tuple(float(value) for value in self.scale_multipliers)
        weights = (
            (1.0,) * len(multipliers)
            if self.scale_weights is None
            else tuple(float(value) for value in self.scale_weights)
        )
        if not multipliers or len(multipliers) != len(weights):
            raise ValueError(
                "Scale multipliers and weights must be nonempty "
                "and have matching lengths."
            )
        if any(not math.isfinite(s) or s <= 0 for s in multipliers):
            raise ValueError("Scale multipliers must be finite and positive.")
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Scale weights must be finite and non-negative.")
        largest = max(weights)
        if largest == 0:
            raise ValueError("At least one scale weight must be positive.")
        scaled = tuple(w / largest for w in weights)
        total = math.fsum(scaled)
        object.__setattr__(self, "scale_multipliers", multipliers)
        object.__setattr__(
            self, "scale_weights", tuple(w / total for w in scaled)
        )
        if self.gamma is not None and (
            not math.isfinite(self.gamma) or self.gamma <= 0.0
        ):
            raise ValueError("gamma must be finite and strictly positive.")
        if (
            not math.isfinite(self.gamma_one_probability)
            or not 0.0 <= self.gamma_one_probability <= 1.0
        ):
            raise ValueError(
                "gamma_one_probability must be finite and between 0 and 1."
            )
        if self.gamma_randomization not in _GAMMA_RANDOMIZATIONS:
            supported = ", ".join(
                repr(value) for value in _GAMMA_RANDOMIZATIONS
            )
            raise ValueError(
                f"Unknown gamma_randomization {self.gamma_randomization!r}. "
                f"Supported strategies are: {supported}."
            )

    def _select_donors(
        self,
        particles: Array,
        complementary_particles: Array,
        rng: RandomGenerator,
    ) -> tuple[Array, Array]:
        """Select uniform ordered distinct donors from the fixed complement."""
        return _sample_donor_indices(
            particles.shape[0], complementary_particles.shape[0], rng
        )

    def sample(
        self,
        particles: Array,
        complementary_particles: Array,
        rng: RandomGenerator,
    ) -> KernelSample:
        xp = namespace(particles, complementary_particles)
        current = particles
        if current.ndim != 2:
            raise ValueError(
                "DifferentialEvolutionKernel requires a 2D active particle "
                "array."
            )
        complementary = complementary_particles
        if complementary.ndim != 2:
            raise ValueError(
                "DifferentialEvolutionKernel requires a 2D complementary "
                "particle array."
            )
        if complementary.shape[0] < 2:
            raise ValueError(
                "DifferentialEvolutionKernel requires at least two "
                "complementary particles."
            )
        if complementary.shape[1] != current.shape[1]:
            raise ValueError(
                "Active and complementary particles must have the same "
                "parameter dimension."
            )

        n_active, n_dimensions = current.shape
        donor_j, donor_k = self._select_donors(current, complementary, rng)
        gamma_base = self.gamma
        if gamma_base is None:
            gamma_base = 2.38 / math.sqrt(2.0 * n_dimensions)
        gamma_effective, unit_mask = _sample_gamma_and_unit_mask(
            n_active,
            gamma_base,
            self.gamma_one_probability,
            self.gamma_randomization,
            rng,
            like=current,
        )
        component = xp.zeros_like(donor_j)
        if len(self.scale_multipliers) > 1 and self.gamma_one_probability < 1:
            weights = asarray(
                self.scale_weights, xp=xp, like=current, dtype=current.dtype
            )
            component = rng.choice(
                len(self.scale_multipliers), size=(n_active,), p=weights
            )
        multiplier = asarray(
            self.scale_multipliers, xp=xp, like=current, dtype=current.dtype
        )[component]
        gamma_effective = xp.where(
            unit_mask, 1.0, gamma_effective * multiplier
        )
        component = xp.where(unit_mask, -1, component)
        multiplier = xp.where(unit_mask, 1.0, multiplier)
        differences = complementary[donor_j] - complementary[donor_k]
        return KernelSample(
            current + xp.reshape(gamma_effective, (n_active, 1)) * differences,
            {
                "scale_component": component,
                "scale_multiplier": multiplier,
                "donor_j": donor_j,
                "donor_k": donor_k,
            },
        )


def _selection_matrix(
    indices: tuple[int, ...],
    n_dimensions: int,
    *,
    like: Array,
) -> Array:
    """Return a matrix that embeds a parameter block in the full space."""
    matrix = np.zeros((len(indices), n_dimensions), dtype=float)
    matrix[np.arange(len(indices)), indices] = 1.0
    return asarray(matrix, xp=namespace(like), like=like)


def _validate_indices(indices: tuple[int, ...], n_dimensions: int) -> None:
    if not indices:
        raise ValueError("A parameter block must contain at least one index.")
    if len(set(indices)) != len(indices):
        raise ValueError("Parameter-block indices must be unique.")
    if min(indices) < 0 or max(indices) >= n_dimensions:
        raise IndexError(
            "Parameter-block indices must be within the particle dimension."
        )


@dataclass(frozen=True)
class MaskedGaussianRandomWalk(SymmetricMutationKernel):
    """Multivariate Gaussian random walk on selected particle coordinates.

    Inactive coordinates are copied without modification. ``covariance`` is the
    covariance within the active subspace; ``scale`` multiplies its Cholesky
    factor and therefore acts as a standard-deviation multiplier.
    """

    indices: tuple[int, ...]
    covariance: Array | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "indices", tuple(int(i) for i in self.indices)
        )
        if not self.indices:
            raise ValueError(
                "MaskedGaussianRandomWalk requires active indices."
            )
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(
                "MaskedGaussianRandomWalk indices must be unique."
            )
        if min(self.indices) < 0:
            raise ValueError(
                "MaskedGaussianRandomWalk indices must be non-negative."
            )
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(
                "MaskedGaussianRandomWalk.scale must be positive."
            )
        if self.covariance is not None:
            covariance = to_numpy(self.covariance)
            expected = (len(self.indices), len(self.indices))
            if covariance.shape != expected:
                raise ValueError(
                    f"covariance must have shape {expected}, got "
                    f"{covariance.shape}."
                )
            if not np.allclose(covariance, covariance.T):
                raise ValueError("covariance must be symmetric.")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "covariance must be positive definite."
                ) from exc

    def sample(self, particles: Array, rng: RandomGenerator) -> Array:
        xp = namespace(particles)
        values = asarray(particles, xp=xp)
        if values.ndim != 2:
            raise ValueError(
                "MaskedGaussianRandomWalk requires a 2D particle array."
            )
        _validate_indices(self.indices, values.shape[1])

        active_dimension = len(self.indices)
        noise = asarray(
            rng.normal(size=(values.shape[0], active_dimension)),
            xp=xp,
            like=values,
        )
        if self.covariance is not None:
            covariance = asarray(self.covariance, xp=xp, like=values)
            cholesky = xp.linalg.cholesky(covariance)
            noise = xp.matmul(noise, xp.permute_dims(cholesky, (1, 0)))
        noise = self.scale * noise

        selection = _selection_matrix(
            self.indices,
            values.shape[1],
            like=values,
        )
        return values + xp.matmul(noise, selection)


@dataclass(frozen=True)
class AdaptiveMaskedGaussianRandomWalk:
    """Build a masked Gaussian kernel from the current particle covariance.

    The covariance is estimated once at the beginning of each mutation block,
    keeping the resulting proposal symmetric throughout that block.
    """

    indices: tuple[int, ...]
    scale: float | None = None
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "indices", tuple(int(i) for i in self.indices)
        )
        if not self.indices:
            raise ValueError(
                "AdaptiveMaskedGaussianRandomWalk requires active indices."
            )
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(
                "AdaptiveMaskedGaussianRandomWalk indices must be unique."
            )
        if min(self.indices) < 0:
            raise ValueError(
                "AdaptiveMaskedGaussianRandomWalk indices must be non-negative."
            )
        if self.scale is not None and (
            not math.isfinite(self.scale) or self.scale <= 0.0
        ):
            raise ValueError(
                "AdaptiveGaussianRandomWalk.scale must be positive."
            )
        if (
            not math.isfinite(self.regularization)
            or self.regularization <= 0.0
        ):
            raise ValueError(
                "AdaptiveGaussianRandomWalk.regularization must be positive."
            )

    def __call__(
        self,
        particles: Array,
        context: StepContext,
    ) -> MaskedGaussianRandomWalk:
        del context
        xp = namespace(particles)
        values = asarray(particles, xp=xp)
        if values.ndim != 2:
            raise ValueError(
                "AdaptiveGaussianRandomWalk requires a 2D particle array."
            )
        _validate_indices(self.indices, values.shape[1])

        selection = _selection_matrix(
            self.indices,
            values.shape[1],
            like=values,
        )
        active = xp.matmul(values, xp.permute_dims(selection, (1, 0)))
        centered = active - xp.mean(active, axis=0)
        denominator = max(values.shape[0] - 1, 1)
        covariance = xp.matmul(xp.permute_dims(centered, (1, 0)), centered)
        covariance = covariance / denominator
        identity = asarray(
            np.eye(len(self.indices)),
            xp=xp,
            like=values,
        )
        covariance = covariance + self.regularization * identity
        scale = self.scale
        if scale is None:
            scale = 2.38 / math.sqrt(len(self.indices))
        return MaskedGaussianRandomWalk(
            indices=self.indices,
            covariance=covariance,
            scale=scale,
        )


@dataclass
class MixtureMutationKernel:
    """Mixture of mutation kernels.

    Randomly selects one component kernel for each proposal.
    """

    kernels: Sequence[MutationKernel]
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        self.kernels = tuple(self.kernels)
        if not self.kernels:
            raise ValueError(
                "MixtureMutationKernel requires at least one kernel."
            )
        for kernel in self.kernels:
            has_correction = (
                hasattr(kernel, "log_transition_ratio")
                or hasattr(kernel, "log_density")
                or hasattr(kernel, "log_transition_ratio_from_sample")
            )
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

    def sample(self, particles: Array, rng: RandomGenerator) -> KernelSample:
        selected_index = int(
            rng.choice(
                len(self.kernels),
                size=(1,),
                p=self.weights,
            )[0]
        )
        child = _as_kernel_sample(
            self.kernels[selected_index].sample(particles, rng)
        )
        return KernelSample(
            child.particles,
            {
                "mixture_component": selected_index,
                "mixture_sample": child,
            },
        )

    def log_transition_ratio_from_sample(
        self,
        from_particles: Array,
        sample: KernelSample,
    ) -> Array:
        return evaluate_log_transition_ratio(
            self.kernels[sample.auxiliary["mixture_component"]],
            from_particles,
            sample.auxiliary["mixture_sample"],
        )


@dataclass(frozen=True)
class MixtureComplementaryEnsembleKernel(ComplementaryEnsembleKernel):
    """Select a fixed-weight component independently per active particle.

    Parameters
    ----------
    kernels : sequence of mutation or complementary-ensemble kernels
        Children receive only selected particles, and the frozen complement
        when required. Corrections use the selected child, not a marginal
        mixture density. Component choices are independent of coordinates.
    weights : sequence of float or None
        Finite non-negative weights with positive sum; None means uniform.
        Configuration is normalized to immutable tuples.
    auxiliary_templates : sequence of pytree or None
        Optional child diagnostic schemas. Each leaf is an array describing
        the dtype and trailing shape (without a particle axis). DE and
        independence schemas are provided automatically; other children default
        to None, and require an explicit template if they return diagnostics.

    Notes
    -----
    Eager execution only: dispatch uses host indices and variable batch sizes.
    Auxiliary contains particle-aligned ``mixture_component``, ``correction``,
    and a fixed tuple ``children``. Each child has a ``selected`` mask and
    ``diagnostics`` padded with zeros outside that mask. DE scale metadata is
    retained there; padded values must only be interpreted under the mask.
    Floating diagnostic leaves use the input particle dtype.
    Both red-blue halves therefore have identical pytree structures, even
    when a component is never selected. Child schemas must remain fixed.
    """

    kernels: Sequence[MutationKernel | ComplementaryEnsembleKernel]
    weights: Sequence[float] | None = None
    auxiliary_templates: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        kernels = tuple(self.kernels)
        if not kernels:
            raise ValueError("A mixture requires at least one kernel.")
        for kernel in kernels:
            if not callable(getattr(kernel, "sample", None)) or not any(
                callable(getattr(kernel, name, None))
                for name in (
                    "log_transition_ratio",
                    "log_transition_ratio_from_sample",
                    "log_density",
                )
            ):
                raise TypeError(
                    "Mixture children need sampling and correction."
                )
        weights = np.asarray(
            self.weights
            if self.weights is not None
            else np.ones(len(kernels)),
            dtype=float,
        )
        if weights.shape != (len(kernels),) or (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError(
                "Weights must match kernels and be finite, "
                "non-negative, with positive total."
            )
        weights = weights / weights.max()
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(
            self, "weights", tuple(float(w) for w in weights / weights.sum())
        )
        templates = self.auxiliary_templates
        if templates is None:
            templates = []
            for kernel in kernels:
                if isinstance(kernel, DifferentialEvolutionKernel):
                    templates.append(
                        {
                            "scale_component": np.asarray(0),
                            "scale_multiplier": np.asarray(0.0),
                            "donor_j": np.asarray(0),
                            "donor_k": np.asarray(0),
                        }
                    )
                elif isinstance(kernel, IndependentProposalKernel):
                    templates.append({"log_proposal": np.asarray(0.0)})
                else:
                    templates.append(None)
        if len(templates) != len(kernels):
            raise ValueError("Diagnostic templates must match kernels.")
        object.__setattr__(self, "auxiliary_templates", tuple(templates))

    def sample(
        self,
        particles: Array,
        complementary_particles: Array,
        rng: RandomGenerator,
    ) -> KernelSample:
        """Draw selected child proposals against the frozen complement."""
        xp = namespace(particles, complementary_particles)
        n = particles.shape[0]
        component = asarray(
            rng.choice(len(self.kernels), size=(n,), p=self.weights),
            xp=xp,
            like=particles,
        )
        proposals = particles
        correction = xp.zeros(
            (n,), dtype=particles.dtype, device=get_device(particles)
        )
        children = []
        assert self.auxiliary_templates is not None
        for i, (kernel, template) in enumerate(
            zip(self.kernels, self.auxiliary_templates, strict=True)
        ):
            selected = component == i
            indices = asarray(
                np.flatnonzero(to_numpy(selected)),
                xp=xp,
                like=component,
                dtype=component.dtype,
            )

            def empty(leaf: Array) -> Array:
                value = asarray(leaf, xp=xp, like=particles)
                dtype = (
                    particles.dtype
                    if xp.isdtype(value.dtype, "real floating")
                    else value.dtype
                )
                return xp.zeros(
                    (n, *value.shape),
                    dtype=dtype,
                    device=get_device(particles),
                )

            diagnostics = optree.tree_map(empty, template)
            if indices.shape[0]:
                active = particles[indices]
                complement = (
                    complementary_particles
                    if isinstance(kernel, ComplementaryEnsembleKernel)
                    else None
                )
                child = _as_kernel_sample(
                    kernel.sample(active, complementary_particles, rng)
                    if isinstance(kernel, ComplementaryEnsembleKernel)
                    else kernel.sample(active, rng)
                )
                values = asarray(
                    child.particles,
                    xp=xp,
                    like=particles,
                    dtype=particles.dtype,
                )
                ratio = asarray(
                    evaluate_log_transition_ratio(
                        kernel, active, child, complement
                    ),
                    xp=xp,
                    like=particles,
                    dtype=particles.dtype,
                )
                if values.shape != active.shape or ratio.shape != (
                    indices.shape[0],
                ):
                    raise ValueError(
                        "Child proposals or corrections have "
                        "invalid particle shapes."
                    )

                def scatter(destination: Array, source: Array) -> Array:
                    source = asarray(
                        source,
                        xp=xp,
                        like=destination,
                        dtype=destination.dtype,
                    )
                    if source.shape != (
                        indices.shape[0],
                        *destination.shape[1:],
                    ):
                        raise ValueError(
                            "Child diagnostics must match templates."
                        )
                    return xpx.at(destination, indices).set(source, copy=True)

                if optree.tree_structure(diagnostics) != optree.tree_structure(
                    child.auxiliary
                ):
                    raise ValueError(
                        "Child diagnostics require a matching "
                        "auxiliary template."
                    )
                diagnostics = optree.tree_map(
                    scatter, diagnostics, child.auxiliary
                )
                proposals = xpx.at(proposals, indices).set(values, copy=True)
                correction = xpx.at(correction, indices).set(ratio, copy=True)
            children.append({"selected": selected, "diagnostics": diagnostics})
        return KernelSample(
            proposals,
            {
                "mixture_component": component,
                "correction": correction,
                "children": tuple(children),
            },
        )

    def log_transition_ratio_from_sample(
        self,
        from_particles: Array,
        sample: KernelSample,
        complementary_particles: Array,
    ) -> Array:
        """Return the selected children's cached transition corrections."""
        if sample.particles.shape != from_particles.shape:
            raise ValueError("Proposal samples must match the particle shape.")
        return sample.auxiliary["correction"]


MUTATION_KERNELS: dict[str, type] = {
    "gaussian-random-walk": GaussianRandomWalk,
    "mixture": MixtureMutationKernel,
    "independent-proposal": IndependentProposalKernel,
}

COMPLEMENTARY_ENSEMBLE_KERNELS: dict[
    str, type[ComplementaryEnsembleKernel]
] = {
    "differential-evolution": DifferentialEvolutionKernel,
    "mixture": MixtureComplementaryEnsembleKernel,
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


def get_complementary_ensemble_kernel(
    name: str,
    **kwargs: Any,
) -> ComplementaryEnsembleKernel:
    """Construct a named complementary-ensemble mutation kernel."""
    try:
        kernel_cls = COMPLEMENTARY_ENSEMBLE_KERNELS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(COMPLEMENTARY_ENSEMBLE_KERNELS))
        raise ValueError(
            f"Unknown complementary-ensemble kernel '{name}'. "
            f"Valid options are: {valid}."
        ) from exc
    if kernel_cls is MixtureComplementaryEnsembleKernel:
        try:
            kernel_specs = kwargs.pop("kernels")
        except KeyError as exc:
            raise ValueError(
                "The mixture ensemble kernel requires a 'kernels' option."
            ) from exc
        kwargs["kernels"] = [
            _resolve_ensemble_kernel_spec(spec) for spec in kernel_specs
        ]
    return kernel_cls(**kwargs)


def _resolve_ensemble_kernel_spec(
    spec: Any,
) -> MutationKernel | ComplementaryEnsembleKernel:
    """Resolve a mixture child from either kernel registry."""
    if isinstance(spec, str):
        if spec in COMPLEMENTARY_ENSEMBLE_KERNELS:
            return get_complementary_ensemble_kernel(spec)
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
    kwargs = dict(config.pop("kernel_kwargs", {}) or {})
    if config:
        raise ValueError(
            f"Unknown mixture component options: {', '.join(sorted(config))}."
        )
    factory = (
        get_complementary_ensemble_kernel
        if name in COMPLEMENTARY_ENSEMBLE_KERNELS
        else get_mutation_kernel
    )
    return factory(name, **kwargs)


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
