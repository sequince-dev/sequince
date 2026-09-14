from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Self

import numpy as np
from numpy.random import Generator as NumPyGenerator

from ._typing import Array
from .array_api import asarray, get_device, namespace, to_numpy
from .state import ParticleState
from .steps.context import StepContext
from .steps.meta import META_SMC_OUTPUT_KEY
from .utils.random import resolve_rng

if TYPE_CHECKING:
    from orng import RandomGenerator

    from .target import EvaluationCounts


NormalizingConstantVarianceMethod = Literal[
    "incremental",
    "lee-whiteley",
    "fixed-lag-lee-whiteley",
]


@dataclass(frozen=True)
class FixedLagLeeWhiteleyDiagnostics:
    """Diagnostics for a fixed-lag Lee--Whiteley normalizing-constant estimate.

    ``family_ess`` is the inverse concentration of the final normalized
    particle weight carried by each selected-lag ancestor. It lies between one
    and ``distinct_ancestors``; low values indicate genealogical depletion.
    """

    requested_lag: int
    traced_resampling_events: int
    total_resampling_events: int
    distinct_ancestors: int
    family_ess: float


def _prepare_axis(ax: Any, **subplot_kwargs: Any) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt  # Imported lazily to avoid hard dependency at import time.

    _, ax = plt.subplots(**subplot_kwargs)
    return ax


def _find_resampling_outputs(output: Any):
    """Yield resampling output mappings nested in an iteration output."""
    if isinstance(output, Mapping):
        if "resampled" in output and "ess" in output:
            yield output
        for value in output.values():
            yield from _find_resampling_outputs(value)
    elif isinstance(output, (tuple, list)):
        for value in output:
            yield from _find_resampling_outputs(value)
    elif hasattr(output, "child_output"):
        yield from _find_resampling_outputs(output.child_output)


def _warn_if_resampling_is_not_multinomial(
    resampling_outputs: Sequence[Mapping[str, Any]],
    *,
    estimator: str,
    stacklevel: int,
) -> None:
    """Warn once when an estimator's resampling assumption is not verified."""
    schemes: set[str] = set()
    for output in resampling_outputs:
        if not bool(to_numpy(output["resampled"])):
            continue
        scheme = output.get("resampling_scheme")
        schemes.add(
            scheme if isinstance(scheme, str) and scheme else "unknown"
        )

    if schemes and schemes != {"multinomial"}:
        observed = ", ".join(sorted(schemes))
        warnings.warn(
            f"{estimator} assumes multinomial resampling, but the recorded "
            f"resampling schemes were: {observed}. The returned estimate may "
            "not be theoretically justified.",
            UserWarning,
            stacklevel=stacklevel,
        )


def _supported_output_record_types() -> tuple[type[Any], ...]:
    from .steps.control import ConditionalStepOutput
    from .steps.mutation.blackjax import BlackJAXOutput
    from .steps.mutation.metrics import MutationMetricsOutput
    from .steps.mutation.minipcn import MiniPCNOutput
    from .steps.resampling import ResamplingOutput

    return (
        ConditionalStepOutput,
        ResamplingOutput,
        MiniPCNOutput,
        BlackJAXOutput,
        MutationMetricsOutput,
    )


def _encode_result_output(
    value: Any,
    path: str,
    active: set[int],
    *,
    strict: bool,
) -> Any:
    """Normalize portable result output without copying array leaves."""
    if isinstance(value, SMCResult):
        return value._encode_for_hdf5(
            active=active,
            include_version=False,
            strict=strict,
        )
    if isinstance(value, _supported_output_record_types()):
        identity = id(value)
        if identity in active:
            raise ValueError(f"Cyclic result output at {path}.")
        active.add(identity)
        try:
            return {
                item.name: _encode_result_output(
                    getattr(value, item.name),
                    f"{path}/{item.name}",
                    active,
                    strict=strict,
                )
                for item in fields(value)
            }
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"Cyclic result output at {path}.")
        active.add(identity)
        try:
            return {
                key: _encode_result_output(
                    child, f"{path}/{key}", active, strict=strict
                )
                for key, child in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"Cyclic result output at {path}.")
        active.add(identity)
        try:
            children = [
                _encode_result_output(
                    child, f"{path}/{index}", active, strict=strict
                )
                for index, child in enumerate(value)
            ]
            return children if isinstance(value, list) else tuple(children)
        finally:
            active.remove(identity)
    encoder = getattr(value, "encode_for_hdf5", None)
    if callable(encoder):
        identity = id(value)
        if identity in active:
            raise ValueError(f"Cyclic result output at {path}.")
        active.add(identity)
        try:
            encoded = encoder()
            if not isinstance(encoded, Mapping):
                raise TypeError(
                    f"{type(value).__name__}.encode_for_hdf5() at {path} "
                    "must return a mapping."
                )
            return _encode_result_output(encoded, path, active, strict=strict)
        finally:
            active.remove(identity)
    if hasattr(type(value), "__dataclass_fields__"):
        if not strict:
            warnings.warn(
                "Converting unrecognized result-output dataclass "
                f"{type(value).__module__}.{type(value).__qualname__} at "
                f"{path} to a dictionary; its class will not be restored.",
                UserWarning,
                stacklevel=4,
            )
            identity = id(value)
            if identity in active:
                raise ValueError(f"Cyclic result output at {path}.")
            active.add(identity)
            try:
                return {
                    item.name: _encode_result_output(
                        getattr(value, item.name),
                        f"{path}/{item.name}",
                        active,
                        strict=strict,
                    )
                    for item in fields(value)
                }
            finally:
                active.remove(identity)
        raise TypeError(
            f"Unsupported result output type {type(value).__module__}."
            f"{type(value).__qualname__} at {path}."
        )
    return value


def _decode_result_output(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("_sequince_kind") == "smc_result":
            return SMCResult._decode_from_hdf5(value)
        return {
            key: _decode_result_output(child) for key, child in value.items()
        }
    if isinstance(value, list):
        return [_decode_result_output(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_decode_result_output(child) for child in value)
    return value


@dataclass(frozen=True)
class SMCResult:
    """Lightweight container exposing diagnostics for a completed SMC run."""

    states: Sequence[ParticleState]
    context: StepContext
    initial_state: ParticleState | None = None
    outputs: Sequence[tuple[Any, ...]] = ()

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("SMCResults requires at least one ParticleState.")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> ParticleState:
        return self.states[index]

    def append_state(self, state: ParticleState) -> SMCResult:
        """Return a new result with one additional state appended."""
        return SMCResult(
            states=[*self.states, state],
            context=self.context,
            outputs=self.outputs,
            initial_state=self.initial_state,
        )

    def encode_for_hdf5(self, *, strict: bool = False) -> dict[str, Any]:
        """Return a portable HDF5 payload for this result.

        By default, unrecognized output dataclasses are converted to
        dictionaries with a warning. ``strict=True`` rejects them instead.
        """
        return self._encode_for_hdf5(
            active=set(), include_version=True, strict=strict
        )

    def _encode_for_hdf5(
        self,
        *,
        active: set[int],
        include_version: bool,
        strict: bool,
    ) -> dict[str, Any]:
        """Return the logical HDF5 payload for this result."""
        from .utils.hdf5 import nested, schema_header

        identity = id(self)
        if identity in active:
            raise ValueError("Cyclic nested SMCResult in result outputs.")
        active.add(identity)
        try:
            return {
                **schema_header("smc_result", include_version=include_version),
                "states": [
                    nested(state.encode_for_hdf5()) for state in self.states
                ],
                "context": self.context._encode_for_result_hdf5(),
                "initial_state": (
                    None
                    if self.initial_state is None
                    else nested(self.initial_state.encode_for_hdf5())
                ),
                "outputs": _encode_result_output(
                    self.outputs, "outputs", active, strict=strict
                ),
            }
        finally:
            active.remove(identity)

    @classmethod
    def _decode_from_hdf5(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a result from its logical HDF5 payload."""
        from .utils.hdf5 import validate_schema

        validate_schema(
            payload,
            kinds={"smc_result"},
            required={"states", "context", "initial_state", "outputs"},
        )
        states_payload = payload["states"]
        if not isinstance(states_payload, (list, tuple)):
            raise TypeError("SMCResult states must be a sequence.")
        initial_payload = payload["initial_state"]
        return cls(
            states=[
                ParticleState._decode_from_hdf5(state)
                for state in states_payload
            ],
            context=StepContext._decode_from_result_hdf5(payload["context"]),
            initial_state=(
                None
                if initial_payload is None
                else ParticleState._decode_from_hdf5(initial_payload)
            ),
            outputs=_decode_result_output(payload["outputs"]),
        )

    def save(
        self,
        h5_file_or_group: Any,
        path: str = "result",
        *,
        strict: bool = False,
    ) -> None:
        """Save this result below ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import recursively_save_to_h5_file

        recursively_save_to_h5_file(
            h5_file_or_group, path, self.encode_for_hdf5(strict=strict)
        )

    def save_to_file(
        self,
        file_path: str | Path,
        path: str = "result",
        *,
        strict: bool = False,
    ) -> None:
        """Save this result to an HDF5 file without replacing existing data."""
        import h5py

        with h5py.File(file_path, "a") as h5_file:
            self.save(h5_file, path=path, strict=strict)

    @classmethod
    def load_from_file(
        cls,
        file_path: str | Path,
        path: str = "result",
    ) -> Self:
        """Load a result from an HDF5 file."""
        import h5py

        with h5py.File(file_path, "r") as h5_file:
            return cls.load(h5_file, path=path)

    @classmethod
    def load(cls, h5_file_or_group: Any, path: str = "result") -> Self:
        """Load a result from ``path`` in an open HDF5 handle."""
        from .utils.hdf5 import load_from_h5_file

        return cls._decode_from_hdf5(load_from_h5_file(h5_file_or_group, path))

    def get_meta_results(
        self,
        key: str = META_SMC_OUTPUT_KEY,
    ) -> MetaSMCResult | None:
        """Return nested SMC results recorded by meta steps."""

        def find_nested(output: Any):
            if isinstance(output, Mapping):
                if key in output:
                    yield output[key]
                for value in output.values():
                    yield from find_nested(value)
            elif isinstance(output, (tuple, list)):
                for value in output:
                    yield from find_nested(value)
            elif hasattr(output, "child_output"):
                yield from find_nested(output.child_output)

        meta_results = list(find_nested(self.outputs))
        if meta_results:
            return MetaSMCResult(results=meta_results)
        return None

    @property
    def final_state(self) -> ParticleState:
        """Return the last state produced by the run."""
        return self.states[-1]

    @property
    def evaluation_counts(self) -> EvaluationCounts:
        """Return the probabilistic work accumulated during the run."""
        return self.context.evaluation_counts

    def log_normalizing_constant(self) -> Array:
        """Return the log normalizing constant of the final state."""
        return self.final_state.log_normalizing_constant

    def log_evidence(self) -> Array:
        """Alias for :meth:`log_normalizing_constant`."""
        return self.log_normalizing_constant()

    def log_normalizing_constant_variance(
        self,
        method: NormalizingConstantVarianceMethod = "incremental",
        *,
        lag: int | None = None,
    ) -> Array:
        """Estimate variance of the final log normalizing-constant estimate.

        ``"incremental"`` sums the per-iteration variance diagnostics. The
        result ignores dependence between SMC iterations.

        ``"lee-whiteley"`` uses the particle genealogy to estimate relative
        normalizing-constant variance, which is also a delta-method estimate of
        log normalizing-constant variance. This estimator requires constant
        particle count and multinomial resampling at every iteration. The
        resampling scheme is not currently retained by :class:`SMCResult`, so
        callers are responsible for ensuring that multinomial resampling was
        used.

        ``"fixed-lag-lee-whiteley"`` This estimator truncates the Lee-Whiteley
        genealogical normalizing-constant variance estimate to the most recent
        ``lag`` resampling events. Particle families are therefore defined by
        their ancestors ``lag`` resampling events before the final population
        rather than by their time-zero ancestors.
        """
        if method != "fixed-lag-lee-whiteley" and lag is not None:
            raise ValueError(
                "lag is only valid for method='fixed-lag-lee-whiteley'."
            )
        if method == "incremental":
            return self._log_normalizing_constant_variance_incremental()
        if method == "lee-whiteley":
            return self._log_normalizing_constant_variance_lee_whiteley()
        if method == "fixed-lag-lee-whiteley":
            if lag is None:
                raise ValueError(
                    "method='fixed-lag-lee-whiteley' requires a non-negative lag."
                )
            variance, _ = (
                self._log_normalizing_constant_variance_fixed_lag_lee_whiteley(
                    lag
                )
            )
            return variance
        raise ValueError(
            f"Unknown log normalizing-constant variance method: {method!r}"
        )

    def log_evidence_variance(
        self,
        method: NormalizingConstantVarianceMethod = "incremental",
        *,
        lag: int | None = None,
    ) -> Array:
        """Alias for :meth:`log_normalizing_constant_variance`."""
        return self.log_normalizing_constant_variance(method=method, lag=lag)

    def _log_normalizing_constant_variance_incremental(self) -> Array:
        xp = namespace(self.final_state.log_weights)
        log_normalizing_constant_increments_variance = self.metadata_history(
            "log_normalizing_constant_increment_variance",
        )
        return xp.sum(log_normalizing_constant_increments_variance)

    def _log_normalizing_constant_variance_lee_whiteley(self) -> Array:
        """Return the Lee--Whiteley normalizing-constant variance estimate."""
        if len(self.outputs) != len(self.states):
            raise ValueError(
                "Lee-Whiteley variance estimation requires one output record "
                "per particle state."
            )

        n_particles = self.final_state.particles.shape[0]
        if n_particles <= 1:
            raise ValueError(
                "Lee-Whiteley variance estimation requires at least two "
                "particles."
            )

        recorded_resampling_outputs: list[Mapping[str, Any]] = []
        for iteration, (state, output) in enumerate(
            zip(self.states, self.outputs, strict=True)
        ):
            if state.particles.shape[0] != n_particles:
                raise ValueError(
                    "Lee-Whiteley variance estimation requires a constant "
                    "particle count."
                )

            resampling_outputs = list(_find_resampling_outputs(output))
            if len(resampling_outputs) != 1:
                raise ValueError(
                    "Lee-Whiteley variance estimation requires exactly one "
                    f"resampling output per iteration; found "
                    f"{len(resampling_outputs)} at iteration {iteration}."
                )
            if not bool(to_numpy(resampling_outputs[0]["resampled"])):
                raise ValueError(
                    "Lee-Whiteley variance estimation currently requires "
                    "resampling at every iteration."
                )

            resampling_output = resampling_outputs[0]
            recorded_resampling_outputs.append(resampling_output)
            if "ancestors" not in resampling_output:
                raise ValueError(
                    "Lee-Whiteley variance estimation requires recorded "
                    f"ancestor indices at iteration {iteration}."
                )
            parents = np.asarray(to_numpy(resampling_output["ancestors"]))
            if parents.shape != (n_particles,):
                raise ValueError(
                    "Ancestor indices must have shape "
                    f"({n_particles},), got {parents.shape} at iteration "
                    f"{iteration}."
                )
            if not np.issubdtype(parents.dtype, np.integer):
                raise ValueError(
                    "Ancestor indices must have an integer dtype."
                )
            if np.any((parents < 0) | (parents >= n_particles)):
                raise ValueError(
                    f"Ancestor indices are out of bounds at iteration {iteration}."
                )
        _warn_if_resampling_is_not_multinomial(
            recorded_resampling_outputs,
            estimator="Lee-Whiteley variance estimation",
            stacklevel=4,
        )
        initial_ancestor_indices = np.asarray(
            to_numpy(self.final_state.initial_ancestor_indices)
        )
        if initial_ancestor_indices.shape != (n_particles,):
            raise ValueError(
                "Initial ancestor indices must match the final particle "
                "population."
            )
        family_sizes = np.bincount(
            initial_ancestor_indices,
            minlength=n_particles,
        )
        different_eve_pairs = n_particles**2 - np.sum(family_sizes**2)
        correction = (n_particles / (n_particles - 1)) ** len(self.states)
        variance = 1.0 - correction * different_eve_pairs / (
            n_particles * (n_particles - 1)
        )
        xp = namespace(self.final_state.log_weights)
        return asarray(variance, xp=xp, like=self.final_state.log_weights)

    def _resampling_history(self) -> list[np.ndarray]:
        """Return validated parent maps for resampling events in run order."""
        if len(self.outputs) != len(self.states):
            raise ValueError(
                "Fixed-lag Lee-Whiteley variance estimation requires one "
                "output record per particle state."
            )

        n_particles = self.final_state.particles.shape[0]
        if n_particles <= 1:
            raise ValueError(
                "Fixed-lag Lee-Whiteley variance estimation requires at least "
                "two particles."
            )

        history: list[np.ndarray] = []
        recorded_resampling_outputs: list[Mapping[str, Any]] = []
        for iteration, (state, output) in enumerate(
            zip(self.states, self.outputs, strict=True)
        ):
            if state.particles.shape[0] != n_particles:
                raise ValueError(
                    "Fixed-lag Lee-Whiteley variance estimation requires a "
                    "constant particle count."
                )
            resampling_outputs = list(_find_resampling_outputs(output))
            if len(resampling_outputs) != 1:
                raise ValueError(
                    "Fixed-lag Lee-Whiteley variance estimation requires "
                    "exactly one resampling output per iteration; found "
                    f"{len(resampling_outputs)} at iteration {iteration}."
                )
            resampling_output = resampling_outputs[0]
            recorded_resampling_outputs.append(resampling_output)
            if not bool(to_numpy(resampling_output["resampled"])):
                continue
            if "ancestors" not in resampling_output:
                raise ValueError(
                    "Fixed-lag Lee-Whiteley variance estimation requires "
                    "recorded ancestor indices at resampling iteration "
                    f"{iteration}."
                )
            parents = np.asarray(to_numpy(resampling_output["ancestors"]))
            if parents.shape != (n_particles,):
                raise ValueError(
                    "Ancestor indices must have shape "
                    f"({n_particles},), got {parents.shape} at iteration "
                    f"{iteration}."
                )
            if not np.issubdtype(parents.dtype, np.integer):
                raise ValueError(
                    "Ancestor indices must have an integer dtype."
                )
            if np.any((parents < 0) | (parents >= n_particles)):
                raise ValueError(
                    f"Ancestor indices are out of bounds at iteration {iteration}."
                )
            history.append(parents)
        _warn_if_resampling_is_not_multinomial(
            recorded_resampling_outputs,
            estimator="Fixed-lag Lee-Whiteley variance estimation",
            stacklevel=5,
        )
        return history

    def _log_normalizing_constant_variance_fixed_lag_lee_whiteley(
        self,
        lag: int,
    ) -> tuple[Array, FixedLagLeeWhiteleyDiagnostics]:
        """Return the fixed-lag Lee--Whiteley variance and diagnostics."""
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            raise ValueError("lag must be a non-negative integer.")

        history = self._resampling_history()
        n_particles = self.final_state.particles.shape[0]
        traced = min(lag, len(history))
        ancestors = np.arange(n_particles)
        if traced:
            for parents in reversed(history[-traced:]):
                ancestors = parents[ancestors]

        weights = np.asarray(to_numpy(self.final_state.weights), dtype=float)
        family_weights = np.bincount(
            ancestors,
            weights=weights,
            minlength=n_particles,
        )
        family_concentration = float(np.sum(family_weights**2))
        different_family_mass = max(0.0, 1.0 - family_concentration)
        if different_family_mass == 0.0:
            variance = 1.0
        else:
            log_pair_term = (traced + 1) * math.log1p(
                1.0 / (n_particles - 1)
            ) + math.log(different_family_mass)
            variance = float(-np.expm1(log_pair_term))
        # Finite-sample genealogical estimators can be slightly negative before
        # their large-N limit; preserve that signal rather than clipping it.
        diagnostics = FixedLagLeeWhiteleyDiagnostics(
            requested_lag=lag,
            traced_resampling_events=traced,
            total_resampling_events=len(history),
            distinct_ancestors=int(np.unique(ancestors).size),
            family_ess=1.0 / family_concentration,
        )
        xp = namespace(self.final_state.log_weights)
        return (
            asarray(variance, xp=xp, like=self.final_state.log_weights),
            diagnostics,
        )

    def fixed_lag_lee_whiteley_diagnostics(
        self,
        lag: int,
    ) -> FixedLagLeeWhiteleyDiagnostics:
        """Return diagnostics for a fixed-lag Lee--Whiteley estimate."""
        _, diagnostics = (
            self._log_normalizing_constant_variance_fixed_lag_lee_whiteley(lag)
        )
        return diagnostics

    def log_normalizing_constant_error(
        self,
        method: NormalizingConstantVarianceMethod = "incremental",
        *,
        lag: int | None = None,
    ) -> Array:
        """Return the standard error of the log normalizing constant."""
        variance = self.log_normalizing_constant_variance(
            method=method, lag=lag
        )
        if float(to_numpy(variance)) < 0.0:
            raise ValueError(
                "The estimated log normalizing-constant variance is negative; "
                "a real standard error cannot be computed."
            )
        xp = namespace(variance)
        return xp.sqrt(variance)

    def log_evidence_error(
        self,
        method: NormalizingConstantVarianceMethod = "incremental",
        *,
        lag: int | None = None,
    ) -> Array:
        """Alias for :meth:`log_normalizing_constant_error`."""
        return self.log_normalizing_constant_error(method=method, lag=lag)

    def log_normalizing_constant_history(self) -> Array:
        """Return the log normalizing-constant trajectory."""
        reference = self.final_state.log_weights if self.states else None
        xp = namespace(reference)
        values = [
            float(state.log_normalizing_constant) for state in self.states
        ]
        return asarray(values, xp=xp, like=reference)

    def log_evidence_history(self) -> Array:
        """Alias for :meth:`log_normalizing_constant_history`."""
        return self.log_normalizing_constant_history()

    def effective_sample_sizes(self) -> Array:
        """Return the effective sample size (ESS) for every iteration."""
        reference = self.final_state.log_weights if self.states else None
        xp = namespace(reference)
        values = [
            float(state.effective_sample_size()) for state in self.states
        ]
        return asarray(values, xp=xp, like=reference)

    def effective_sample_size_ratios(self) -> Array:
        """Return ESS / n_particles for each iteration."""
        reference = self.final_state.log_weights if self.states else None
        xp = namespace(reference)
        values = [
            float(state.effective_sample_size() / state.particles.shape[0])
            for state in self.states
        ]
        return asarray(values, xp=xp, like=reference)

    def metadata_history(
        self, key: str, *, default: Any = math.nan
    ) -> Array | list[Any]:
        """Return an array containing a metadata field pulled from each state."""
        values = [state.metadata.get(key, default) for state in self.states]
        if isinstance(default, (int, float)):
            reference = self.final_state.log_weights if self.states else None
            xp = namespace(reference)
            return asarray(values, xp=xp, like=reference)
        return values

    def summary(self) -> dict[str, Any]:
        """Return a concise dictionary of high-level diagnostics."""
        ess = self.effective_sample_sizes()
        ess_ratio = self.effective_sample_size_ratios()
        log_normalizing_constant = self.log_normalizing_constant_history()
        summary_dict: dict[str, Any] = {
            "iterations": len(self),
            "final_ess": float(to_numpy(ess)[-1]),
            "final_ess_ratio": float(to_numpy(ess_ratio)[-1]),
            "final_log_normalizing_constant": float(
                to_numpy(log_normalizing_constant)[-1]
            ),
        }

        beta_history = self.metadata_history("beta", default=math.nan)
        summary_dict["final_beta"] = float(to_numpy(beta_history)[-1])
        return summary_dict

    def posterior_samples(
        self,
        *,
        n_samples: int | None = None,
        rng: "RandomGenerator" | NumPyGenerator | int | None = None,
        method: Literal["multinomial", "rejection", "weights"] = "weights",
    ) -> tuple[Array, Array]:
        """Return samples from the weighted posterior represented by the final state.

        Parameters
        ----------
        n_samples
            Number of samples to return. For ``"weights"``, this uniformly
            down-selects the weighted particle population without replacement.
            For ``"rejection"``, this is an upper limit on the number of accepted
            particles. For ``"multinomial"``, exactly this many samples are drawn
            with replacement. If ``None``, all weighted or rejection-thinned
            particles are returned, while multinomial sampling draws the same
            number of samples as there are particles.
        rng
            Random number generator or seed.
        method
            Method used to construct the returned posterior representation:

            - ``"weights"``: return weighted particles, optionally uniformly
            down-selected without replacement.
            - ``"rejection"``: independently retain each particle with probability
            proportional to its weight, returning equal-weight unique samples.
            - ``"multinomial"``: draw equal-weight samples with replacement
            according to the particle weights.

        Returns
        -------
        samples
            Posterior samples. Scalar particles are returned as a column
            with shape ``(n_samples, 1)``.
        sample_weights
            Normalized weights associated with ``samples``.
        """
        final_state = self.final_state
        xp = namespace(final_state.particles, final_state.weights)

        particles = final_state.particles
        if particles.ndim == 1:
            particles = xp.reshape(particles, (-1, 1))
        weights = asarray(final_state.weights, xp=xp)
        weights = weights / xp.sum(weights)

        n_particles = particles.shape[0]

        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if (
            method in {"weights", "rejection"}
            and n_samples is not None
            and n_samples > n_particles
        ):
            raise ValueError(
                f"n_samples cannot exceed the particle count for method={method!r}."
            )

        rng_obj: RandomGenerator = resolve_rng(rng)

        if method == "weights":
            if n_samples is None or n_samples == n_particles:
                samples = particles
                sample_weights = weights
            else:
                indices = asarray(
                    rng_obj.choice(
                        n_particles,
                        size=n_samples,
                        replace=False,
                    ),
                    xp=xp,
                    like=particles,
                )
                samples = particles[indices]
                sample_weights = weights[indices]
                sample_weights = sample_weights / xp.sum(sample_weights)

        elif method == "rejection":
            u = asarray(
                rng_obj.uniform(size=n_particles),
                xp=xp,
                like=weights,
            )
            acceptance_probabilities = weights / xp.max(weights)
            accept = u < acceptance_probabilities
            samples = particles[accept]

            if n_samples is not None:
                if samples.shape[0] < n_samples:
                    warnings.warn(
                        f"Rejection sampling returned {samples.shape[0]} samples, "
                        f"fewer than the requested {n_samples}. Consider using "
                        "method='multinomial' or method='weights' instead.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif samples.shape[0] > n_samples:
                    indices = asarray(
                        rng_obj.choice(
                            samples.shape[0],
                            size=n_samples,
                            replace=False,
                        ),
                        xp=xp,
                        like=particles,
                    )
                    samples = samples[indices]

            sample_weights = xp.full(
                (samples.shape[0],),
                1.0 / samples.shape[0],
                dtype=weights.dtype,
                device=get_device(weights),
            )

        elif method == "multinomial":
            if n_samples is None:
                n_samples = n_particles

            indices = asarray(
                rng_obj.choice(
                    n_particles,
                    size=n_samples,
                    replace=True,
                    p=weights,
                ),
                xp=xp,
                like=particles,
            )
            samples = particles[indices]
            sample_weights = xp.full(
                (n_samples,),
                1.0 / n_samples,
                dtype=weights.dtype,
                device=get_device(weights),
            )

        else:
            raise ValueError(
                f"Unknown posterior sampling method: {method!r}. "
                "Valid options are 'multinomial', 'rejection' or 'weights'."
            )

        return samples, sample_weights

    def weighted_mean(
        self,
        *,
        state_index: int = -1,
    ) -> Array:
        """Return the weighted mean of the specified state (defaults to final)."""
        state = self.states[state_index]
        xp = namespace(state.particles)
        particles = xp.atleast_2d(state.particles)
        weights = asarray(state.weights, xp=xp)
        weights = xp.reshape(weights, (weights.shape[0], 1))
        return xp.sum(weights * particles, axis=0)

    def weighted_covariance(
        self,
        *,
        state_index: int = -1,
        ddof: int = 0,
    ) -> Array:
        """Return the weighted covariance matrix for the specified state."""
        state = self.states[state_index]
        xp = namespace(state.particles)
        particles = xp.atleast_2d(state.particles)
        weights = asarray(state.weights, xp=xp)
        mean = asarray(
            self.weighted_mean(state_index=state_index),
            xp=xp,
            like=particles,
        )
        centered = particles - mean
        weights_column = xp.reshape(weights, (weights.shape[0], 1))
        weighted = centered * weights_column
        cov = xp.matmul(xp.transpose(centered), weighted)
        weight_sum = float(to_numpy(xp.sum(weights)))
        if ddof != 0:
            weight_sum = max(weight_sum - ddof, 1e-12)
        return cov / weight_sum

    def apply_weighted(
        self,
        fn: Callable[[Any], Any],
        *,
        state_index: int = -1,
    ) -> Array:
        """Evaluate a function under the weighted empirical distribution."""
        state = self.states[state_index]
        xp = namespace(state.particles)
        particles = xp.atleast_2d(state.particles)
        xp = namespace(particles)
        values = asarray(fn(particles), xp=xp, like=particles)
        weights = asarray(state.weights, xp=xp)
        if values.ndim == 1:
            return xp.sum(weights * values)
        reshape_dims = (weights.shape[0],) + (1,) * (values.ndim - 1)
        expanded_weights = xp.reshape(weights, reshape_dims)
        return xp.sum(expanded_weights * values, axis=0)

    def plot_log_normalizing_constant(
        self, *, ax: Any = None, label_axes: bool = True, **plot_kwargs: Any
    ) -> Any:
        """Plot the log normalizing-constant trajectory."""
        x = to_numpy(range(len(self)))
        y = to_numpy(self.log_normalizing_constant_history())
        axis = _prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Log normalizing constant")
            axis.set_title("Log normalizing-constant trajectory")
        return axis

    def plot_log_evidence(
        self, *, ax: Any = None, label_axes: bool = True, **plot_kwargs: Any
    ) -> Any:
        """Alias for :meth:`plot_log_normalizing_constant`."""
        return self.plot_log_normalizing_constant(
            ax=ax, label_axes=label_axes, **plot_kwargs
        )

    def plot_effective_sample_size(
        self,
        *,
        normalize: bool = False,
        ax: Any = None,
        label_axes: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot ESS or the ESS ratio across iterations."""
        x = to_numpy(range(len(self)))
        if normalize:
            y = to_numpy(self.effective_sample_size_ratios())
            ylabel = "ESS / n"
            title = "Effective sample size ratio"
        else:
            y = to_numpy(self.effective_sample_sizes())
            ylabel = "ESS"
            title = "Effective sample size"
        axis = _prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
        return axis

    def plot_metadata_history(
        self,
        key: str,
        *,
        ax: Any = None,
        default: Any = math.nan,
        label_axes: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot the evolution of a metadata entry stored in each state."""
        x = to_numpy(range(len(self)))
        y = to_numpy(self.metadata_history(key, default=default))
        axis = _prepare_axis(ax)
        axis.plot(x, y, **plot_kwargs)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel(key)
            axis.set_title(f"Metadata '{key}' over iterations")
        return axis

    def plot_corner(
        self,
        *,
        state_index: int = -1,
        include_weights: bool = False,
        parameter_inds: Sequence[int] | None = None,
        filename: str | Path | None = None,
        **corner_kwargs: Any,
    ) -> Any:
        """
        Plot a corner plot of the particles in the specified state.

        Parameters
        ----------
        state_index
            Index of the state to plot (default is -1, the final state).
        include_weights
            If ``True``, include particle weights in the corner plot.
        parameter_inds
            Indices of the parameters to include in the corner plot.
            If ``None`` (default) all parameters are included.
        filename
            If provided, the plot is saved to this file instead of being returned.
        corner_kwargs
            Additional keyword arguments passed to the `corner` function.

        Returns
        -------
        fig
            The corner plot figure, if `filename` is not provided.
        """
        from .plot import plot_corner

        state = self.states[state_index]
        return plot_corner(
            state.particles,
            weights=state.weights if include_weights else None,
            parameter_inds=parameter_inds,
            filename=filename,
            **corner_kwargs,
        )

    def plot_trace(
        self,
        *,
        ax: Any = None,
        parameter_inds: Sequence[int] | None = None,
        parameter_labels: Sequence[str] | None = None,
        filename: str | Path | None = None,
        kind: str = "violin",
        **plot_kwargs: Any,
    ) -> Any:
        """
        Plot a trace plot of the particles as a function of state index.

        Parameters
        ----------
        ax
            Axis on which to plot.  If ``None``, a new figure and axis are created.
        parameter_inds
            Indices of the parameters to include in the trace plot.
            If ``None`` (default) all parameters are included.
        parameter_labels
            Labels for the parameters to include in the trace plot.
            If ``None`` (default) generic labels `{x_i}` are used.
        filename
            If provided, the plot is saved to this file instead of being returned.
        kind
            The kind of trace plot to create.  Currently only "violin" or "scatter" are supported.
        plot_kwargs
            Additional keyword arguments passed to the `ax.plot` function.

        Returns
        -------
        ax
            The trace plot axis, if `filename` is not provided.
        """
        import matplotlib.pyplot as plt

        assert kind in ("violin", "scatter"), (
            "Only 'violin' and 'scatter' kinds are supported."
        )

        default_kwargs: dict[str, Any] = dict()

        if kind == "scatter":
            default_kwargs.update(
                dict(
                    marker=",",
                    linestyle="",
                )
            )
        elif kind == "violin":
            default_kwargs.update(
                dict(
                    side="high",
                    widths=2,
                    showextrema=False,
                )
            )

        if plot_kwargs:
            default_kwargs.update(plot_kwargs)

        all_particles = np.array(
            [np.atleast_2d(to_numpy(state.particles)) for state in self.states]
        )  # (n_iterations, n_particles, n_parameters)
        n_iterations = len(all_particles)

        if parameter_inds is None:
            parameter_inds = list(range(all_particles.shape[2]))

        if parameter_labels is None:
            parameter_labels = [f"x_{i}" for i in parameter_inds]

        assert len(parameter_inds) == len(parameter_labels)

        axis = _prepare_axis(
            ax,
            nrows=len(parameter_inds),
            figsize=(5, 3 * len(parameter_inds)),
            sharex=True,
        )

        if len(parameter_inds) == 1:
            axis = [axis]

        if kind == "scatter":
            x_iter = np.repeat(np.arange(n_iterations), all_particles.shape[1])
            all_particles = np.array(all_particles)

        for i, param_ind in enumerate(parameter_inds):
            if kind == "scatter":
                axis[i].plot(
                    x_iter,
                    all_particles[:, :, param_ind].flatten(),
                    **default_kwargs,
                )
            elif kind == "violin":
                axis[i].violinplot(
                    all_particles[:, :, param_ind].T,
                    **default_kwargs,
                )

            axis[i].set_ylabel(parameter_labels[i])

        axis[-1].set_xlabel("Iteration")

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            return ax.get_figure()


@dataclass(frozen=True)
class MetaSMCResult:
    """Aggregate results from a SMC run with a meta SMC step."""

    results: Sequence[SMCResult]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError(
                "MetaSMCResult requires at least one meta-step result."
            )

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> SMCResult:
        return self.results[index]

    def encode_for_hdf5(self, *, strict: bool = False) -> dict[str, Any]:
        """Return the logical HDF5 payload for this meta-SMC result."""
        from .utils.hdf5 import schema_header

        active: set[int] = set()
        return {
            **schema_header("meta_smc_result"),
            "results": [
                result._encode_for_hdf5(
                    active=active,
                    include_version=False,
                    strict=strict,
                )
                for result in self.results
            ],
        }

    @classmethod
    def _decode_from_hdf5(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a meta-SMC result from its logical payload."""
        from .utils.hdf5 import validate_schema

        validate_schema(
            payload,
            kinds={"meta_smc_result"},
            required={"results"},
        )
        results = payload["results"]
        if not isinstance(results, (list, tuple)):
            raise TypeError("MetaSMCResult results must be a sequence.")
        return cls(
            results=[SMCResult._decode_from_hdf5(item) for item in results]
        )

    def save(
        self,
        h5_file_or_group: Any,
        path: str = "meta_result",
        *,
        strict: bool = False,
    ) -> None:
        """Save this meta-SMC result in an open HDF5 handle."""
        from .utils.hdf5 import recursively_save_to_h5_file

        recursively_save_to_h5_file(
            h5_file_or_group, path, self.encode_for_hdf5(strict=strict)
        )

    @classmethod
    def load(cls, h5_file_or_group: Any, path: str = "meta_result") -> Self:
        """Load a meta-SMC result from an open HDF5 handle."""
        from .utils.hdf5 import load_from_h5_file

        return cls._decode_from_hdf5(load_from_h5_file(h5_file_or_group, path))

    @property
    def final_result(self) -> SMCResult:
        """Return the final meta-step result."""
        return self.results[-1]

    @property
    def final_state(self) -> ParticleState:
        """Return the final particle state from the last meta step."""
        return self.final_result.final_state

    @property
    def all_states(self) -> list[ParticleState]:
        """Flatten particle states across all meta steps in execution order."""
        return [state for result in self.results for state in result.states]

    def log_normalizing_constant_history(self) -> Array:
        """Return the log normalizing-constant trajectory across meta steps."""
        reference = self.final_state.log_weights
        xp = namespace(reference)
        values = [
            float(state.log_normalizing_constant) for state in self.all_states
        ]
        return asarray(values, xp=xp, like=reference)

    def log_evidence_history(self) -> Array:
        """Alias for :meth:`log_normalizing_constant_history`."""
        return self.log_normalizing_constant_history()

    def plot_log_normalizing_constant(
        self,
        *,
        ax: Any = None,
        label_axes: bool = True,
        cmap: str = "viridis",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot the log normalizing-constant trajectory across meta steps."""
        import matplotlib.pyplot as plt

        axis = _prepare_axis(ax)
        x0 = 0.0
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(self.results)))
        for result, color in zip(self.results, colors):
            x = np.arange(x0, x0 + len(result))
            y = to_numpy(result.log_normalizing_constant_history())
            axis.plot(x, y, color=color, **plot_kwargs)
            x0 += len(result)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Log normalizing constant")
            axis.set_title(
                "Log normalizing-constant trajectory across meta steps"
            )
        return axis

    def plot_log_evidence(
        self,
        *,
        ax: Any = None,
        label_axes: bool = True,
        cmap: str = "viridis",
        **plot_kwargs: Any,
    ) -> Any:
        """Alias for :meth:`plot_log_normalizing_constant`."""
        return self.plot_log_normalizing_constant(
            ax=ax,
            label_axes=label_axes,
            cmap=cmap,
            **plot_kwargs,
        )

    def plot_metadata_history(
        self,
        key: str,
        *,
        ax: Any = None,
        default: Any = math.nan,
        label_axes: bool = True,
        cmap: str = "viridis",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot metadata history across all meta steps."""
        import matplotlib.pyplot as plt

        axis = _prepare_axis(ax)
        x0 = 0.0

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(self.results)))
        for result, color in zip(self.results, colors):
            x = np.arange(x0, x0 + len(result))
            y = to_numpy(result.metadata_history(key, default=default))
            axis.plot(x, y, color=color, **plot_kwargs)
            x0 += len(result)
        if label_axes:
            axis.set_xlabel("Iteration")
            axis.set_ylabel(key)
            axis.set_title(f"Metadata '{key}' over iterations")
        return axis

    def summary(self) -> dict[str, Any]:
        """Return a concise dictionary of high-level diagnostics."""
        summary_dict: dict[str, Any] = {
            "meta_steps": len(self.results),
            "total_iterations": len(self.all_states),
            "final_log_normalizing_constant": float(
                self.final_state.log_normalizing_constant
            ),
        }
        try:
            ess = float(self.final_state.effective_sample_size())
            summary_dict["final_ess"] = ess
            summary_dict["final_ess_ratio"] = float(
                ess / self.final_state.particles.shape[0]
            )
        except Exception:
            pass
        return summary_dict
