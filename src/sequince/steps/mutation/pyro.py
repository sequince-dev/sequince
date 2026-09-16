"""Optional, sequential Pyro NUTS transitions for Torch particles."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import TYPE_CHECKING, Any

from ..._typing import Array
from ...array_api import to_numpy
from ...target import EvaluationCounts, Target
from ...transforms import (
    CompositeTransform,
    ParticleTransform,
    PeriodicBoundaryTransform,
    select_transform_auxiliary,
)
from ...utils.copying import array_safe_copy
from ...utils.random import RNGLike, resolve_rng
from ..base import validate_n_steps
from ..context import StepContext
from .base import MutationStep, TransformRefit
from .metrics import MutationMetric, MutationStepState

if TYPE_CHECKING:
    import torch

    from ...state import ParticleState


def _tensor(value: Any, like: torch.Tensor, name: str) -> torch.Tensor:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must return a Torch tensor preserving autograd."
        )
    if value.dtype != like.dtype or value.device != like.device:
        raise TypeError(f"{name} must preserve the particle dtype and device.")
    return value


def _validate_transform(transform: ParticleTransform | None) -> None:
    if transform is not None and not isinstance(transform, ParticleTransform):
        raise TypeError("transform must be a ParticleTransform or None.")
    if isinstance(transform, PeriodicBoundaryTransform):
        raise ValueError(
            "NUTS requires a differentiable bijection; periodic wrapping "
            "is not supported."
        )
    if isinstance(transform, CompositeTransform):
        for component in transform.transforms:
            _validate_transform(component)


class PyroNUTSMutation(MutationStep):
    """Apply independent, fixed-tuning NUTS transitions to Torch particles.

    Parameters
    ----------
    log_prob_fn : Target or callable
        Differentiable target in the stored particle coordinates. A callable
        receives one position of shape (n_dimensions,) and optionally context,
        and must return a scalar Torch tensor. Sampled parameters must be
        continuous; discrete parameters and non-differentiable components are
        unsupported, including when encoded with a floating-point dtype.
    n_steps : int, default=1
        Complete NUTS transitions per particle, not leapfrog steps.
    step_size : float, default=0.1
        Fixed positive integration step size; the mass matrix is identity.
    max_tree_depth : int, default=10
        Maximum NUTS tree depth.
    use_multinomial_sampling : bool, default=True
        Use Pyro's multinomial selection instead of slice sampling.
    transform : ParticleTransform or None
        Differentiable bijection from particle to mutation coordinates.
        Its inverse log absolute Jacobian is added exactly once.
    transform_refit : str or callable, default="always"
        Existing Sequince transform fitting policy, applied before mutation.
    store_chains : bool, default=False
        Include particle-coordinate chains in the output.
    metrics : sequence of MutationMetric, default=()
        Existing population metrics. Their acceptance field measures movement.

    Notes
    -----
    Requires ``sequince[pyro]`` and Torch particles on CPU or CUDA. Pyro uses
    global Torch generators; a seed drawn from Sequince's RNG is installed in a
    forked CPU/active-CUDA RNG context and restored even on failure. This is
    sequential and not thread safe. Python and NumPy RNGs are not seeded.
    Tuning and the fitted transform are frozen for the entire call: no particle
    can adapt the transition applied to subsequent particles.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | Target,
        *,
        n_steps: int = 1,
        step_size: float = 0.1,
        max_tree_depth: int = 10,
        use_multinomial_sampling: bool = True,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
        metrics: tuple[MutationMetric, ...] | list[MutationMetric] = (),
    ) -> None:
        for name, value in (
            ("n_steps", n_steps),
            ("max_tree_depth", max_tree_depth),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(step_size, bool)
            or not isinstance(step_size, Real)
            or not math.isfinite(step_size)
            or step_size <= 0
        ):
            raise ValueError("step_size must be finite and positive.")
        if not isinstance(use_multinomial_sampling, bool):
            raise TypeError("use_multinomial_sampling must be a bool.")
        if not isinstance(store_chains, bool):
            raise TypeError("store_chains must be a bool.")
        if not isinstance(log_prob_fn, Target) and not callable(log_prob_fn):
            raise TypeError("log_prob_fn must be a Target or callable.")
        _validate_transform(transform)
        try:
            from pyro.infer.mcmc import NUTS  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyroNUTSMutation requires `pip install 'sequince[pyro]'` "
                "(pyro-ppl)."
            ) from exc
        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
            metrics=metrics,
        )
        self._n_steps = n_steps
        self._step_size: float = step_size
        self._max_tree_depth: int = max_tree_depth
        self._use_multinomial_sampling: bool = use_multinomial_sampling

    def _potential(
        self,
        position: torch.Tensor,
        context: StepContext,
        transform: ParticleTransform | None = None,
        auxiliary: Any = None,
    ) -> tuple[torch.Tensor, EvaluationCounts]:
        """Evaluate fresh target components, never an SMC state's cache."""
        import torch

        with torch.enable_grad():
            original, log_det = (
                (position, None)
                if transform is None
                else transform.inverse(
                    position, context=context, auxiliary=auxiliary
                )
            )
            original = _tensor(original, position, "Transform inverse")
            if not original.requires_grad:
                raise ValueError(
                    "Transform inverse is detached; autograd is required."
                )
            result, _, counts = self._evaluate_target(original, context)
            result = _tensor(result, position, "Target")
            if result.ndim != 0:
                raise ValueError(
                    "Target must be scalar for a single particle."
                )
            if not result.requires_grad:
                raise ValueError(
                    "Target is detached; Torch autograd is required."
                )
            if log_det is not None:
                log_det = _tensor(log_det, position, "Transform Jacobian")
                if log_det.ndim != 0:
                    raise ValueError("Transform Jacobian must be scalar.")
                result = result + log_det
            return -result, counts

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RNGLike,
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self.n_steps
        )

    def apply_with_mutation_state(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RNGLike,
        *,
        n_steps: int,
        mutation_state: MutationStepState | None = None,
    ) -> tuple[ParticleState, StepContext, dict[str, Any], MutationStepState]:
        import torch
        from pyro.infer.mcmc import NUTS

        n_steps = validate_n_steps(n_steps)
        particles = state.particles
        if not isinstance(particles, torch.Tensor):
            raise TypeError(
                "PyroNUTSMutation requires Torch-backed particles."
            )
        if not particles.is_floating_point():
            raise TypeError(
                "Particles must have a floating-point dtype; no discrete parameters."
            )
        if particles.device.type not in {"cpu", "cuda"}:
            raise ValueError("PyroNUTSMutation supports only CPU and CUDA.")
        if particles.ndim != 2 or particles.shape[1] == 0:
            raise ValueError(
                "Particles must have shape (n_particles, n_dimensions)."
            )
        if not torch.isfinite(particles).all():
            raise ValueError("Initial particles must be finite.")
        # Snapshot metadata (including tensor beta), before any target calls.
        frozen = context.with_metadata(
            **array_safe_copy(dict(context.metadata))
        )
        generator = resolve_rng(rng)
        with torch.no_grad():
            transform, frozen = self._prepare_transform(
                particles, frozen, rng=generator
            )
            _validate_transform(transform)
            positions, auxiliary = self._forward_transform(
                particles, transform, frozen, rng=generator
            )
        positions = _tensor(positions, particles, "Transform forward").detach()
        if (
            positions.shape != particles.shape
            or not torch.isfinite(positions).all()
        ):
            raise ValueError(
                "Transform must preserve shape and yield finite positions."
            )
        mutation_state = (
            self.metrics.initialize(positions, state.log_weights, frozen)
            if mutation_state is None
            else self.metrics.prepare(mutation_state, positions, frozen)
        )
        n_particles = particles.shape[0]
        output: dict[str, Any] = dict(
            n_particles=n_particles,
            steps=n_steps,
            step_size=self._step_size,
            max_tree_depth=self._max_tree_depth,
            divergences=0,
            pyro_acceptance_rate=None,
            chain=None,
            metrics=None,
        )
        if n_steps == 0:
            if self.store_chains:
                output["chain"] = particles.new_empty((0, *particles.shape))
            return state, context, output, mutation_state
        # Freeze configuration before processing any particle. No warmup/adaptation.
        config = dict(
            step_size=self._step_size,
            max_tree_depth=self._max_tree_depth,
            use_multinomial_sampling=self._use_multinomial_sampling,
            adapt_step_size=False,
            adapt_mass_matrix=False,
        )
        counts = EvaluationCounts()
        chains = positions.new_empty((n_steps, *positions.shape))
        rates = []
        devices = [particles.device.index] if particles.is_cuda else []
        # This scalar transfer is only a seed, never a target/position conversion.
        seed = int(to_numpy(generator.random((1,)))[0] * (2**31 - 1))
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.random.default_generator.manual_seed(seed)
            if particles.is_cuda:
                with torch.cuda.device(particles.device):
                    torch.cuda.manual_seed(seed)
            for index, initial in enumerate(positions):
                particle_auxiliary = select_transform_auxiliary(
                    auxiliary, index, n_particles=n_particles
                )

                def potential_fn(
                    params: dict[str, torch.Tensor],
                ) -> torch.Tensor:
                    nonlocal counts
                    potential, work = self._potential(
                        params["position"],
                        frozen,
                        transform,
                        particle_auxiliary,
                    )
                    counts = counts + work
                    return potential

                position = initial.detach().clone().requires_grad_(True)
                # Validate once per particle, not once per leapfrog/transition.
                potential = potential_fn({"position": position})
                try:
                    (gradient,) = torch.autograd.grad(potential, position)
                except RuntimeError as exc:
                    raise ValueError(
                        "Target/transform must provide gradients for continuous parameters."
                    ) from exc
                if (
                    not torch.isfinite(potential)
                    or not torch.isfinite(gradient).all()
                ):
                    raise ValueError(
                        "Initial potential and gradients must be finite."
                    )
                kernel = NUTS(potential_fn=potential_fn, **config)
                params = {"position": position.detach()}
                kernel.initial_params = params
                try:
                    kernel.setup(warmup_steps=0)
                    for step in range(n_steps):
                        params = kernel.sample(params)
                        chains[step, index] = params["position"].detach()
                    diagnostics = kernel.diagnostics()
                    output["divergences"] += len(diagnostics["divergences"])
                    rates.append(diagnostics["acceptance rate"])
                finally:
                    kernel.cleanup()
        with torch.no_grad():
            previous = positions
            for population in chains:
                moved = torch.any(population != previous, dim=-1)
                mutation_state = self.metrics.update(
                    mutation_state, population, moved, frozen
                )
                previous = population
            final, _ = self._inverse_transform(
                chains[-1], transform, frozen, auxiliary=auxiliary
            )
            final = _tensor(final, particles, "Transform inverse").detach()
            _, evaluation, work = self._evaluate_target(final, frozen)
            counts = counts + work
            if self.store_chains:
                output["chain"] = self._inverse_chain(
                    chains,
                    transform,
                    frozen,
                    auxiliary=auxiliary,
                    like=particles,
                ).detach()
        output["pyro_acceptance_rate"] = sum(rates) / n_particles
        return (
            state.replace_particles(final, evaluation=evaluation),
            frozen.add_evaluation_counts(counts),
            output,
            mutation_state,
        )
