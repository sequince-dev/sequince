"""Adaptive annealing with on-the-fly Gaussian mixture expansion.

This example demonstrates sequential Monte Carlo with adaptive tempering and
MCMC mutation to fit a Gaussian mixture model to one-dimensional data.  The
number of mixture components is not fixed in advance; instead, new components
are added dynamically during the SMC run whenever the current annealing cycle
completes.  Each new component's parameters are proposed from data-informed
distributions to encourage efficient exploration.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import matplotlib.pyplot as plt
import numpy as np
from orng import RandomGenerator
from scipy.special import gammaln, logsumexp, softmax

from sequince import (
    ParticleState,
    SequentialMonteCarlo,
    TargetEvaluation,
    TemperedPosterior,
)
from sequince.log import configure_logging
from sequince.result import NestedSMCResult
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MiniPCNMutation,
    SMCTransformation,
    SystematicResampling,
    TransformationStep,
)
from sequince.steps.base import StepContext
from sequince.steps.control import StopIf
from sequince.steps.nested import NestedSMCStep


def generate_gmm_data(
    weights: Sequence[float],
    means: Sequence[float],
    *,
    std: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw observations from a 1D Gaussian mixture."""
    weights = np.array(weights, dtype=float)
    weights /= np.sum(weights)
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


def _blocked_gmm_log_likelihoods(
    particles: np.ndarray,
    context: StepContext,
    component_counts: Sequence[int],
) -> dict[int, np.ndarray]:
    """Evaluate mixture likelihoods using bounded observation blocks."""
    points = np.atleast_2d(np.asarray(particles, dtype=float))
    n_particles, param_dim = points.shape
    if param_dim % 2 != 0:
        raise ValueError(
            "Particle array must contain logits followed by means."
        )

    available_components = param_dim // 2
    counts = tuple(dict.fromkeys(int(count) for count in component_counts))
    if not counts:
        raise ValueError("At least one component count must be requested.")
    if any(count < 1 or count > available_components for count in counts):
        raise ValueError(
            "Requested component counts must be between one and the number "
            "of components represented by the particle array."
        )

    data = np.asarray(context.shared["data"], dtype=float)
    std = float(context.shared["component_std"])
    variance = std**2
    if variance <= 0.0:
        raise ValueError("component_std must be positive.")
    block_size = int(context.shared.get("likelihood_block_size", 64))
    if block_size < 1:
        raise ValueError("likelihood_block_size must be positive.")

    max_components = max(counts)
    # Every particle is laid out as [all logits, all component means].
    logits = points[:, :max_components]
    means = points[
        :, available_components : available_components + max_components
    ]
    log_normalizers = {
        count: logsumexp(logits[:, :count], axis=1) for count in counts
    }
    totals = {count: np.zeros(n_particles, dtype=float) for count in counts}
    log_gaussian_normalizer = 0.5 * np.log(2.0 * np.pi * variance)

    # Reuse each bounded data block for every requested mixture order.
    for start in range(0, data.size, block_size):
        block = data[start : start + block_size]
        log_terms = block[np.newaxis, :, np.newaxis] - means[:, np.newaxis, :]
        np.square(log_terms, out=log_terms)
        log_terms *= -0.5 / variance
        log_terms -= log_gaussian_normalizer
        log_terms += logits[:, np.newaxis, :]

        for count in counts:
            log_mix = logsumexp(log_terms[:, :, :count], axis=2)
            log_mix -= log_normalizers[count][:, np.newaxis]
            totals[count] += np.sum(log_mix, axis=1)

    return totals


def gmm_log_likelihood(
    particles: np.ndarray,
    context: StepContext,
    *,
    active_components: int | None = None,
) -> np.ndarray:
    """Log likelihood of the dataset under a shared-variance mixture."""
    param_dim = particles.shape[1]
    if active_components is None:
        n_components = int(context.metadata.get("active_components", 0))
        if n_components <= 0:
            if param_dim % 2 != 0:
                raise ValueError(
                    "Particle array must contain logits followed by means."
                )
            n_components = param_dim // 2
    else:
        n_components = active_components
    return _blocked_gmm_log_likelihoods(particles, context, (n_components,))[
        n_components
    ]


class GMMExpansionTransformation(SMCTransformation):
    """Add one mixture component using the component prior as proposal."""

    def forward(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> ParticleState:
        current_components = int(context.metadata["active_components"])
        particles = np.asarray(state.particles)
        if particles.ndim == 1:
            particles = particles[:, np.newaxis]

        n = particles.shape[0]
        bounds: tuple[float, float] = context.shared["mean_bounds"]
        logit_prior_scale = float(context.shared["logit_prior_scale"])

        # Drawing the added parameters from their priors makes their prior and
        # auxiliary-proposal terms cancel in the subsequent weight update.
        new_logits = rng.normal(loc=0.0, scale=logit_prior_scale, size=(n, 1))
        new_means = rng.uniform(low=bounds[0], high=bounds[1], size=(n, 1))

        # Restore the [all logits, all means] layout after appending the pair.
        existing_logits = particles[:, :current_components]
        existing_means = particles[:, current_components:]
        logits = np.concatenate([existing_logits, new_logits], axis=1)
        means = np.concatenate([existing_means, new_means], axis=1)
        expanded_particles = np.concatenate([logits, means], axis=1)

        next_components = current_components + 1
        # The nested SMC run anneals from the old posterior to this larger model.
        metadata = dict(state.metadata)
        metadata["previous_active_components"] = current_components
        metadata["active_components"] = next_components
        metadata["temperature"] = 0.0
        metadata.pop("temperature_delta", None)

        context.metadata["previous_active_components"] = current_components
        context.metadata["active_components"] = next_components
        context.metadata["temperature"] = 0.0
        context.metadata["temperature_delta"] = 0.0

        return state.with_updates(
            particles=expanded_particles,
            metadata=metadata,
        )

    def inverse(
        self,
        particles: np.ndarray,
        context: StepContext,
    ) -> np.ndarray:
        pts = np.atleast_2d(particles)
        current_components = int(context.metadata["active_components"])
        previous_components = int(
            context.metadata.get(
                "previous_active_components", current_components
            )
        )
        # Removing the appended pair maps an expanded particle back to the
        # particle on which the previous posterior is evaluated.
        logits = pts[:, :current_components]
        means = pts[:, current_components : 2 * current_components]
        old_logits = logits[:, :previous_components]
        old_means = means[:, :previous_components]
        inverse_particles = np.concatenate([old_logits, old_means], axis=1)
        if particles.ndim == 1:
            return inverse_particles[0]
        return inverse_particles

    def log_auxiliary_density(
        self,
        particles: np.ndarray,
        context: StepContext,
    ) -> np.ndarray:
        """The log density for the new component parameters."""
        pts = np.atleast_2d(particles)
        current_components = int(context.metadata["active_components"])
        previous_components = int(
            context.metadata.get(
                "previous_active_components", current_components
            )
        )
        if previous_components >= current_components:
            return np.zeros(pts.shape[0], dtype=float)

        bounds: tuple[float, float] = context.shared["mean_bounds"]
        width = bounds[1] - bounds[0]
        scale = float(context.shared["logit_prior_scale"])
        if width <= 0.0 or scale <= 0.0:
            raise ValueError(
                "Invalid prior configuration for expansion proposal."
            )

        logits = pts[:, :current_components]
        means = pts[:, current_components : 2 * current_components]
        # These are exactly the random variables introduced by forward().
        new_logits = logits[:, previous_components]
        new_means = means[:, previous_components]

        norm_const = np.log(np.sqrt(2.0 * np.pi) * scale)
        logit_density = -0.5 * (new_logits / scale) ** 2 - norm_const

        in_bounds = (new_means >= bounds[0]) & (new_means <= bounds[1])
        mean_density = np.full(pts.shape[0], -np.inf, dtype=float)
        mean_density[in_bounds] = -np.log(width)
        density = logit_density + mean_density
        if particles.ndim == 1:
            return np.array(density[0], ndmin=1)
        return density


class GMMTemperedPosterior(TemperedPosterior):
    """Posterior with a custom increment for one-component expansions.

    When a new mixture component is introduced via the prior-driven
    transformation below, the prior term cancels against the auxiliary
    proposal density and the increment reduces to the likelihood change.
    This method computes the old and new mixture likelihoods from one pass
    over the component log densities.
    """

    def _expansion_component_counts(
        self, context: StepContext
    ) -> tuple[int, int] | None:
        # A transformed proposal marks the bridge between adjacent model sizes.
        if context.metadata.get(self.log_proposal_override_key) is None:
            return None
        previous_components = context.metadata.get(
            "previous_active_components"
        )
        current_components = int(context.metadata.get("active_components", 0))
        if (
            previous_components is None
            or int(previous_components) != current_components - 1
            or current_components <= 1
        ):
            return None
        return int(previous_components), current_components

    def evaluate(
        self,
        particles: np.ndarray,
        context: StepContext,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
    ) -> TargetEvaluation:
        component_counts = self._expansion_component_counts(context)
        if component_counts is None:
            # The first model uses ordinary prior-to-posterior tempering.
            return super().evaluate(
                particles,
                context,
                components=components,
                evaluation=evaluation,
            )

        if evaluation is not None:
            self.validate_evaluation(evaluation, particles, context)
        allowed = {
            "log_prior",
            "previous_log_likelihood",
            "log_likelihood",
        }
        requested = allowed if components is None else set(components)
        unknown = requested.difference(allowed)
        if unknown:
            raise ValueError(
                f"Unknown expansion evaluation components: {sorted(unknown)}"
            )

        values = {} if evaluation is None else dict(evaluation.components)
        points = np.atleast_2d(particles)
        if "log_prior" in requested:
            values["log_prior"] = log_prior(points, context)

        # Cache both endpoints so weighting and mutation share one likelihood pass.
        previous_components, current_components = component_counts
        likelihood_counts = []
        if "previous_log_likelihood" in requested:
            likelihood_counts.append(previous_components)
        if "log_likelihood" in requested:
            likelihood_counts.append(current_components)
        if likelihood_counts:
            log_likelihoods = _blocked_gmm_log_likelihoods(
                points,
                context,
                likelihood_counts,
            )
            if "previous_log_likelihood" in requested:
                values["previous_log_likelihood"] = log_likelihoods[
                    previous_components
                ]
            if "log_likelihood" in requested:
                values["log_likelihood"] = log_likelihoods[current_components]

        return self.bind_evaluation(
            TargetEvaluation(components=values), context
        )

    def log_increment(
        self,
        particles: np.ndarray,
        context: StepContext,
        *,
        evaluation=None,
    ) -> np.ndarray:
        if self._expansion_component_counts(context) is None:
            return super().log_increment(
                particles,
                context,
                evaluation=evaluation,
            )

        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        # The added-parameter prior has cancelled, leaving only L_new / L_old.
        increments = (
            evaluation["log_likelihood"]
            - evaluation["previous_log_likelihood"]
        )

        if particles.ndim == 1:
            return np.array(increments[0], ndmin=1)
        return increments

    def log_prob(
        self,
        particles: np.ndarray,
        context: StepContext,
        *,
        evaluation=None,
    ) -> np.ndarray:
        if self._expansion_component_counts(context) is None:
            return super().log_prob(
                particles,
                context,
                evaluation=evaluation,
            )

        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        beta = float(context.metadata.get("temperature", 0.0))
        # Mutation follows the same geometric bridge used by the weight update.
        posterior = (
            evaluation["log_prior"]
            + (1.0 - beta) * evaluation["previous_log_likelihood"]
            + beta * evaluation["log_likelihood"]
        )

        if particles.ndim == 1:
            return np.array(posterior[0], ndmin=1)
        return posterior


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
    temperature = context.metadata["temperature"]
    components = context.metadata["active_components"]
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


class ExpandComponents(TransformationStep):
    """Increase the mixture order between outer SMC iterations."""

    def __init__(self, target_components: int, posterior: TemperedPosterior):
        super().__init__(
            transformation=GMMExpansionTransformation(),
            posterior=posterior,
        )
        self._target_components = target_components

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> ParticleState:
        current_components = int(context.metadata["active_components"])

        if context.t == 0:
            return state

        if current_components >= self._target_components:
            context.request_stop()
            return state

        return super().apply(state, context, rng)


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
        metadata={"active_components": 1},
    )

    target_components = 6

    posterior = GMMTemperedPosterior(
        log_prior=log_prior,
        log_likelihood=gmm_log_likelihood,
    )

    # The inner loop anneals and rejuvenates particles for one model size.
    inner_runner = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                posterior,
                target_ess_ratio=0.6,
                max_temperature=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(posterior, n_steps=10),
            LoggingStep(logging_str_fn),
        ]
    )

    shared_context = {
        "data": observations,
        "component_std": component_std,
        "mean_bounds": mean_bounds,
        "logit_prior_scale": logit_prior_scale,
        "component_count_rate": 3.0,
        "likelihood_block_size": 256,
    }

    # The outer loop adds one component, runs the inner loop, then repeats.
    outer_runner = SequentialMonteCarlo(
        steps=[
            ExpandComponents(target_components, posterior),
            NestedSMCStep(
                inner_runner,
                posterior=posterior,
                record_outputs=True,
                output_key="inner_results",
            ),
            StopIf(
                lambda state, context: (
                    context.metadata["active_components"] >= target_components
                )
            ),
        ]
    )

    metadata = {"active_components": 1}

    outer_result = outer_runner.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )
    meta_result = NestedSMCResult(
        outer_result.context.outputs["inner_results"]
    )

    all_states = meta_result.all_states
    final_state = meta_result.final_state
    est_weights, est_means = posterior_summary(final_state)
    print("Posterior mixture weights:", np.round(est_weights, 3))
    print("Posterior component means:", np.round(est_means, 3))
    print(f"Final log evidence: {final_state.log_evidence:.2f}")

    iterations = np.arange(len(all_states))
    component_counts = [
        s.metadata.get("active_components", s.particles.shape[1] // 2)
        for s in all_states
    ]
    temperatures = [s.metadata.get("temperature", 0.0) for s in all_states]
    temperature_deltas = [
        s.metadata.get("temperature_delta", 0.0) for s in all_states
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
                "log_evidence": float(all_states[iter_idx - 1].log_evidence),
            }
            for iter_idx in new_component_iters
            if iter_idx > 0
        ]
        print("Pre-expansion log evidence:", precursor_evidence)
    else:
        print("No component expansions detected during run.")

    evidence_by_components: dict[int, float] = {}
    for inter_result in meta_result.results:
        comp = int(
            inter_result.final_state.metadata.get(
                "active_components",
                inter_result.final_state.particles.shape[1] // 2,
            )
        )
        evidence_by_components[comp] = float(
            inter_result.final_state.log_evidence
        )
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
        best_comp = max(
            evidence_by_components, key=lambda c: evidence_by_components[c]
        )
        print(f"Favoured model by log evidence: {best_comp} components")

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
