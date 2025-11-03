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
from scipy.special import gammaln, logsumexp, softmax

from sequince import ParticleState, SequentialMonteCarlo, TemperedPosterior
from sequince.logging_utils import configure_logging
from sequince.steps import (
    AdaptiveTemperedWeighting,
    ContinueWhile,
    DimensionalityExpansion,
    LoggingStep,
    MiniPCNMutation,
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


def log_prior(position: np.ndarray, context: StepContext) -> np.ndarray:
    """Factorized prior over logits (Gaussian) and means (uniform)."""
    points = np.atleast_2d(position)

    bounds: tuple[float, float] = context.shared["mean_bounds"]
    n_components = points.shape[1] // 2
    logits = points[:, :n_components]
    means = points[:, n_components:]

    width = bounds[1] - bounds[0]
    if width <= 0.0:
        raise ValueError("mean_bounds must define a positive-width interval.")

    scale = float(context.shared.get("logit_prior_scale", 2.0))
    if scale <= 0.0:
        raise ValueError("logit_prior_scale must be positive.")

    in_bounds = np.all((means >= bounds[0]) & (means <= bounds[1]), axis=1)
    log_prior_vals = np.full(points.shape[0], -np.inf, dtype=float)

    valid_logits = logits[in_bounds]
    if valid_logits.size > 0:
        norm_const = np.log(np.sqrt(2.0 * np.pi) * scale)
        log_prior_means = -n_components * np.log(width)
        log_prior_logits = -0.5 * np.sum((valid_logits / scale) ** 2, axis=1)
        log_prior_logits -= n_components * norm_const

        rate = float(context.shared.get("component_count_rate", 3.0))
        if rate <= 0.0:
            raise ValueError("component_count_rate must be positive.")
        log_component_count = (
            n_components * np.log(rate) - rate - gammaln(n_components + 1)
        )

        log_prior_vals[in_bounds] = (
            log_prior_means + log_prior_logits + log_component_count
        )

    if position.ndim == 1:
        return np.array(log_prior_vals[0], ndmin=1)
    return log_prior_vals


def gmm_log_likelihood(
    particles: np.ndarray, context: StepContext
) -> np.ndarray:
    """Log likelihood of the dataset under a shared-variance mixture."""
    data = context.shared["data"]
    std = context.shared["component_std"]
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


def component_sampler(
    state: ParticleState,
    *,
    context: StepContext,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a new component's logit and mean from data-informed proposals."""
    n = state.particles.shape[0]
    bounds: tuple[float, float] = context.shared["mean_bounds"]
    data = context.shared["data"]
    spread = context.shared["component_std"]
    logit_prior_scale = float(context.shared["logit_prior_scale"])

    new_logits = rng.normal(loc=0.0, scale=logit_prior_scale, size=(n, 1))
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
        context.shared["components"] = expanded.particles.shape[1] // 2
        # Reset temperature bookkeeping so the next annealing cycle starts at 0.
        context.shared["temperature"] = 0.0
        context.shared["temperature_delta"] = 0.0
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


def logging_str_fn(state: ParticleState, context: StepContext) -> str:
    temperature = context.shared["temperature"]
    components = context.shared["components"]
    ess = state.effective_sample_size()
    n_particles = state.log_weights.shape[0]
    ess_ratio = ess / n_particles if n_particles else 0.0
    log_evidence = state.log_evidence
    return (
        f"it {context.t}:, "
        f"Temperature: {temperature:.4f}, "
        f"Components: {components}, "
        f"ESS: {ess:.1f} ({ess_ratio:.3f}), "
        f"Log Evidence: {log_evidence:.1f}"
    )


def main() -> None:
    configure_logging()

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

    n_particles = 500
    mean_bounds = (
        observations.min() - 3.0 * component_std,
        observations.max() + 3.0 * component_std,
    )

    logit_prior_scale = 2.0
    initial_logits = rng.normal(
        loc=0.0, scale=logit_prior_scale, size=(n_particles, 1)
    )
    initial_means = rng.uniform(
        low=mean_bounds[0], high=mean_bounds[1], size=(n_particles, 1)
    )
    initial_particles = np.concatenate([initial_logits, initial_means], axis=1)
    log_weights = np.full(n_particles, -np.log(n_particles))
    initial_state = ParticleState(
        particles=initial_particles,
        log_weights=log_weights,
        metadata={"temperature": 0.0, "components": 1},
    )

    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=gmm_log_likelihood,
    )

    target_components = 5
    trigger_temperature = 0.999

    smc = SequentialMonteCarlo(
        steps=[
            TemperatureTriggeredExpansion(
                target_components=target_components,
                trigger_temperature=trigger_temperature,
            ),
            AdaptiveTemperedWeighting(
                posterior,
                target_ess_ratio=0.7,
                max_temperature=1.0,
                min_delta=1e-4,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(posterior, n_steps=20),
            LoggingStep(logging_str_fn),
            ContinueWhile(
                make_continue_annealing_predicate(
                    target_components=target_components,
                    trigger_temperature=trigger_temperature,
                )
            ),
        ],
        auto_normalize=True,
    )

    results = smc.run(
        initial_state,
        rng=rng,
        shared_context={
            "data": observations,
            "component_std": component_std,
            "temperature": 0.0,
            "temperature_delta": 0.0,
            "components": 1,
            "mean_bounds": mean_bounds,
            "logit_prior_scale": logit_prior_scale,
            "component_count_rate": 3.0,
        },
    )

    final_state = results.final_state
    est_weights, est_means = posterior_summary(final_state)
    print("Posterior mixture weights:", np.round(est_weights, 3))
    print("Posterior component means:", np.round(est_means, 3))
    print(f"Final log evidence: {final_state.log_evidence:.2f}")

    iterations = np.arange(len(results))
    component_counts = [s.particles.shape[1] // 2 for s in results.states]
    temperatures = [s.metadata.get("temperature", 0.0) for s in results.states]
    temperature_deltas = [
        s.metadata.get("temperature_delta", 0.0) for s in results.states
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
                "log_evidence": float(
                    results.states[iter_idx - 1].log_evidence
                ),
            }
            for iter_idx in new_component_iters
            if iter_idx > 0
        ]
        print("Pre-expansion log evidence:", precursor_evidence)
    else:
        print("No component expansions detected during run.")

    evidence_by_components = {}
    for comp_count, state in zip(component_counts, results.states):
        evidence_by_components[comp_count] = float(state.log_evidence)
    if evidence_by_components:
        print("Log evidence by active component count:")
        for comp in sorted(evidence_by_components):
            print(
                f"  components={comp}: log_evidence={evidence_by_components[comp]:.3f}"
            )

    log_bayes_factors = {}
    base_evidence = evidence_by_components.get(1, None)
    if base_evidence is not None:
        for comp, log_evid in evidence_by_components.items():
            if comp == 1:
                continue
            log_bayes_factors[comp] = log_evid - base_evidence
        print("Bayes factors vs. 1-component model:")
        for comp in sorted(log_bayes_factors):
            print(f"  components={comp}: ln BF={log_bayes_factors[comp]:.3f}")

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
    fig.savefig("gmm_transdimensional.png", dpi=150)
    print("Saved gmm_transdimensional.png")


if __name__ == "__main__":
    main()
