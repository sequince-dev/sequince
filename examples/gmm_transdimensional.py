"""Adaptive annealing with on-the-fly Gaussian mixture expansion.

This example demonstrates sequential Monte Carlo with adaptive tempering and
MCMC mutation to fit a Gaussian mixture model to one-dimensional data.  The
number of mixture components is not fixed in advance; instead, new components
are added dynamically during the SMC run whenever the current annealing cycle
completes.  Each new component's parameters are proposed from data-informed
distributions to encourage efficient exploration.
"""

from __future__ import annotations

from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, softmax

from sequince import ParticleState, SequentialMonteCarlo
from sequince.steps import (
    AdaptiveTemperedWeighting,
    ContinueWhile,
    DimensionalityExpansion,
    EmceeMutation,
    SystematicResampling,
)
from sequince.steps.base import StepContext


def generate_gmm_data(
    weights: Iterable[float],
    means: Iterable[float],
    *,
    std: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw observations from a 1D Gaussian mixture."""
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    means = np.array(means, dtype=float)
    assignments = rng.choice(len(weights), size=n_samples, p=weights)
    return rng.normal(loc=means[assignments], scale=std, size=n_samples)


def log_prior(position: np.ndarray, context: StepContext) -> float:
    """Uniform prior over component means within specified bounds."""
    if position.ndim == 1:
        position = position[np.newaxis, :]

    bounds: tuple[float, float] = context.static["mean_bounds"]
    n_components = position.shape[1] // 2
    means = position[:, n_components:]
    if np.all((means >= bounds[0]) & (means <= bounds[1])):
        # Normalizing constant for uniform prior over means
        return -np.log(bounds[1] - bounds[0]) * n_components
    else:
        return float("-inf")


def gmm_log_likelihood(
    particles: np.ndarray, context: StepContext
) -> np.ndarray:
    """Log likelihood of the dataset under a shared-variance mixture."""
    data = context.static["data"]
    std = context.static["component_std"]
    variance = std**2

    n_particles, param_dim = particles.shape
    if param_dim % 2 != 0:
        raise ValueError(
            "Particle array must contain logits followed by means."
        )

    n_components = param_dim // 2
    logits = particles[:, :n_components]
    means = particles[:, n_components:]
    log_weights = logits - logsumexp(logits, axis=1, keepdims=True)

    residual = data[np.newaxis, :, np.newaxis] - means[:, np.newaxis, :]
    log_component = -0.5 * (residual**2) / variance - 0.5 * np.log(
        2.0 * np.pi * variance
    )
    log_mix = logsumexp(log_weights[:, np.newaxis, :] + log_component, axis=2)
    return np.sum(log_mix, axis=1)


def tempered_log_posterior(
    position: np.ndarray,
    context: StepContext,
) -> float:
    """Tempered log posterior for a single particle position."""
    if position.ndim == 1:
        position = position[np.newaxis, :]

    log_prior_val = log_prior(position, context)
    if not np.isfinite(log_prior_val):
        return float("-inf")

    temperature = float(context.static.get("temperature", 0.0))
    if temperature <= 0.0:
        return 0.0

    log_like = gmm_log_likelihood(position, context)[0]
    return temperature * log_like


def component_sampler(
    state: ParticleState,
    *,
    context: StepContext,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a new component's logit and mean from data-informed proposals."""
    n = state.particles.shape[0]
    bounds: tuple[float, float] = context.static["mean_bounds"]
    data = context.static["data"]
    spread = context.static["component_std"]

    new_logits = rng.normal(loc=0.0, scale=0.4, size=(n, 1))
    anchors = rng.choice(data, size=(n, 1))
    new_means = anchors + rng.normal(scale=spread, size=(n, 1))
    new_means = np.clip(new_means, bounds[0], bounds[1])
    return np.concatenate([new_logits, new_means], axis=1)


def make_continue_annealing_predicate(
    target_components: int,
    trigger_temperature: float,
) -> Callable[[ParticleState, StepContext], bool]:
    """Return predicate that keeps iterating until target components are ready."""

    def _predicate(state: ParticleState, context: StepContext) -> bool:
        components = state.particles.shape[1] // 2
        temperature = float(state.metadata.get("temperature", 0.0))
        return (
            components < target_components or temperature < trigger_temperature
        )

    return _predicate


class TemperatureTriggeredExpansion(DimensionalityExpansion):
    """Append a component whenever the current annealing cycle finishes."""

    def __init__(
        self,
        *,
        target_components: int,
        trigger_temperature: float = 0.999,
    ):
        super().__init__(
            component_sampler,
            expected_block_size=2,
            metadata_key="param_dim",
            context_key="param_dim",
            context_delta_key="param_delta",
        )
        self._target_components = target_components
        self._trigger_temperature = trigger_temperature

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: np.random.Generator,
    ) -> ParticleState:
        current_components = state.particles.shape[1] // 2
        if current_components >= self._target_components:
            return state

        temperature = float(state.metadata.get("temperature", 0.0))
        if temperature < self._trigger_temperature:
            return state

        pre_particles = np.asarray(state.particles)
        if pre_particles.ndim == 1:
            pre_particles = pre_particles[:, np.newaxis]

        expanded = super().apply(state, context, rng)
        post_particles = np.asarray(expanded.particles)
        if post_particles.ndim == 1:
            post_particles = post_particles[:, np.newaxis]

        # Reorder so that logits always precede means.
        new_block = post_particles[:, -2:]
        current_logits = pre_particles[:, :current_components]
        current_means = pre_particles[:, current_components:]
        new_logits = new_block[:, 0][:, np.newaxis]
        new_means = new_block[:, 1][:, np.newaxis]
        reordered = np.concatenate(
            [
                np.concatenate([current_logits, new_logits], axis=1),
                np.concatenate([current_means, new_means], axis=1),
            ],
            axis=1,
        )

        expanded = expanded.with_updates(particles=reordered)
        context.static["components"] = expanded.particles.shape[1] // 2
        # Reset temperature bookkeeping so the next annealing cycle starts at 0.
        context.static["temperature"] = 0.0
        context.static["temperature_delta"] = 0.0
        metadata = dict(expanded.metadata)
        metadata["temperature"] = 0.0
        metadata.pop("temperature_delta", None)
        return expanded.with_updates(metadata=metadata)


def posterior_summary(state: ParticleState) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean weights and component means."""
    particles = np.asarray(state.particles)
    if particles.ndim == 1:
        particles = particles[:, np.newaxis]
    weights = state.weights[:, np.newaxis]
    n_components = particles.shape[1] // 2

    logits = particles[:, :n_components]
    means = particles[:, n_components:]
    component_weights = softmax(logits)

    avg_weights = np.sum(weights * component_weights, axis=0)
    avg_weights /= avg_weights.sum()
    avg_means = np.sum(weights * means, axis=0)
    return avg_weights, avg_means


def mixture_density(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    std: float,
) -> np.ndarray:
    """Evaluate mixture density on grid x."""
    variance = std**2
    normal = np.exp(
        -0.5 * ((x[:, np.newaxis] - means[np.newaxis, :]) ** 2) / variance
    ) / np.sqrt(2.0 * np.pi * variance)
    return normal @ weights


def main() -> None:
    rng = np.random.default_rng(seed=2048)

    true_weights = np.array([0.2, 0.5, 0.3])
    true_means = np.array([-4.0, 0.5, 3.5])
    component_std = 0.75
    observations = generate_gmm_data(
        true_weights,
        true_means,
        std=component_std,
        n_samples=400,
        rng=rng,
    )

    n_particles = 2_000
    data_mean = np.mean(observations)
    mean_bounds = (
        observations.min() - 3.0 * component_std,
        observations.max() + 3.0 * component_std,
    )

    initial_logits = rng.normal(loc=0.0, scale=0.3, size=(n_particles, 1))
    initial_means = rng.normal(
        loc=data_mean, scale=1.5 * component_std, size=(n_particles, 1)
    )
    initial_means = np.clip(initial_means, mean_bounds[0], mean_bounds[1])
    initial_particles = np.concatenate([initial_logits, initial_means], axis=1)
    log_weights = np.full(n_particles, -np.log(n_particles))
    initial_state = ParticleState(
        particles=initial_particles,
        log_weights=log_weights,
        metadata={"temperature": 0.0, "components": 1},
    )

    target_components = 4
    trigger_temperature = 0.999

    smc = SequentialMonteCarlo(
        steps=[
            TemperatureTriggeredExpansion(
                target_components=target_components,
                trigger_temperature=trigger_temperature,
            ),
            AdaptiveTemperedWeighting(
                gmm_log_likelihood,
                target_ess_ratio=0.7,
                max_temperature=1.0,
                min_delta=0.02,
            ),
            SystematicResampling(always_resample=True),
            EmceeMutation(tempered_log_posterior, n_steps=8),
            ContinueWhile(
                make_continue_annealing_predicate(
                    target_components=target_components,
                    trigger_temperature=trigger_temperature,
                )
            ),
        ],
        auto_normalize=True,
    )

    states = smc.run(
        initial_state,
        rng=rng,
        static_context={
            "data": observations,
            "component_std": component_std,
            "temperature": 0.0,
            "temperature_delta": 0.0,
            "components": 1,
            "mean_bounds": mean_bounds,
        },
    )

    final_state = states[-1]
    est_weights, est_means = posterior_summary(final_state)
    print("Posterior mixture weights:", np.round(est_weights, 3))
    print("Posterior component means:", np.round(est_means, 3))
    print(f"Final log evidence: {final_state.log_evidence:.2f}")

    iterations = np.arange(len(states))
    component_counts = [s.particles.shape[1] // 2 for s in states]
    temperatures = [s.metadata.get("temperature", 0.0) for s in states]
    temperature_deltas = [
        s.metadata.get("temperature_delta", 0.0) for s in states
    ]

    x_grid = np.linspace(mean_bounds[0], mean_bounds[1], 400)
    true_density = mixture_density(
        x_grid, true_weights, true_means, component_std
    )
    posterior_density = mixture_density(
        x_grid, est_weights, est_means, component_std
    )
    rmse = np.sqrt(np.mean((posterior_density - true_density) ** 2))
    print(f"Density RMSE vs. true mixture: {rmse:.4f}")

    new_component_iters = np.where(np.diff(component_counts) > 0)[0] + 1
    if new_component_iters.size > 0:
        print(
            "New components introduced after iterations:",
            new_component_iters.tolist(),
        )
        precursor_evidence = [
            {
                "iteration": int(iter_idx - 1),
                "components": int(component_counts[iter_idx - 1]),
                "log_evidence": float(states[iter_idx - 1].log_evidence),
            }
            for iter_idx in new_component_iters
            if iter_idx > 0
        ]
        print("Pre-expansion log evidence:", precursor_evidence)
    else:
        print("No component expansions detected during run.")

    evidence_by_components = {}
    for comp_count, state in zip(component_counts, states):
        evidence_by_components[comp_count] = float(state.log_evidence)
    if evidence_by_components:
        print("Log evidence by active component count:")
        for comp in sorted(evidence_by_components):
            print(
                f"  components={comp}: log_evidence={evidence_by_components[comp]:.3f}"
            )

    fig, axes = plt.subplots(nrows=3, figsize=(10, 9))

    axes[0].hist(
        observations,
        bins=30,
        density=True,
        alpha=0.4,
        color="tab:gray",
        label="Observations",
    )
    axes[0].plot(
        x_grid,
        posterior_density,
        color="tab:blue",
        linewidth=2.0,
        label="Posterior mixture",
    )
    axes[0].plot(
        x_grid,
        true_density,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label="True mixture",
    )
    axes[0].set_title("Adaptive mixture fit")
    axes[0].set_xlabel("Observation")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")

    axes[1].step(
        iterations,
        component_counts,
        where="post",
        color="tab:purple",
        linewidth=2.0,
        label="Active components",
    )
    axes[1].scatter(iterations, component_counts, color="tab:purple", s=15)
    axes[1].set_title("Component count over time")
    axes[1].set_xlabel("SMC iteration")
    axes[1].set_ylabel("Components")
    axes[1].legend(loc="upper left")

    ax_temp = axes[2]
    ax_temp.plot(
        iterations,
        temperatures,
        color="tab:cyan",
        linewidth=2.0,
        label="Temperature",
    )
    ax_delta = ax_temp.twinx()
    ax_delta.step(
        iterations,
        temperature_deltas,
        where="post",
        color="tab:orange",
        linewidth=1.5,
        label="Temperature increment",
    )
    ax_temp.set_title("Adaptive annealing")
    ax_temp.set_xlabel("SMC iteration")
    ax_temp.set_ylabel("Temperature")
    ax_delta.set_ylabel("Increment")
    ax_temp.legend(loc="upper left")
    ax_delta.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig("gmm_dimensionality_adaptive_mcmc.png", dpi=150)
    print("Saved gmm_dimensionality_adaptive_mcmc.png")


if __name__ == "__main__":
    main()
