"""Adaptive tempering starting from an informative proposal distribution."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sequince import ParticleState, SequentialMonteCarlo, TemperedPosterior
from sequince.logging_utils import configure_logging
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MiniPCNMutation,
    StepContext,
    SystematicResampling,
)


def log_likelihood(particles: np.ndarray, context) -> np.ndarray:
    """Log-likelihood of a multivariate normal observation."""
    mu = context.shared["mu"]
    sigma = context.shared["sigma"]
    dims = particles.shape[1]
    residual = particles - mu
    return -0.5 * np.sum(
        (residual / sigma) ** 2, axis=1
    ) - 0.5 * dims * np.log(2.0 * np.pi * sigma**2)


def log_prior(particles: np.ndarray, context) -> np.ndarray:
    """Uniform prior inside the hyper-rectangle, -inf outside."""
    lower = context.shared["prior_lower"]
    upper = context.shared["prior_upper"]
    in_bounds = (particles >= lower) & (particles <= upper)
    norm = -np.log(upper - lower) * particles.shape[1]
    return np.where(np.all(in_bounds, axis=1), norm, -np.inf)


def log_proposal(particles: np.ndarray, context) -> np.ndarray:
    """Gaussian proposal centred away from the true mean."""
    proposal_mean = context.shared["proposal_mean"]
    proposal_sigma = context.shared["proposal_sigma"]
    dims = particles.shape[1]
    residual = particles - proposal_mean
    return -0.5 * np.sum(
        (residual / proposal_sigma) ** 2, axis=1
    ) - 0.5 * dims * np.log(2.0 * np.pi * proposal_sigma**2)


def logging_fn(state: ParticleState, context: StepContext) -> str:
    current_temp = context.metadata.get("temperature", 0.0)
    return (
        f"SMC iteration {context.t}: "
        f"Temperature={current_temp:.4f}, "
        f"Log-evidence={state.log_evidence:.3f}"
    )


def main() -> None:
    rng = np.random.default_rng(321)

    configure_logging()

    dims = 2
    mu = np.zeros(dims)
    sigma = 1.0
    prior_lower, prior_upper = -10.0, 10.0

    proposal_mean = np.full(dims, 3.0)
    proposal_sigma = 2.0

    n_particles = 500
    initial_particles = rng.normal(
        loc=proposal_mean, scale=proposal_sigma, size=(n_particles, dims)
    )
    initial_log_weights = np.full(n_particles, -np.log(n_particles))
    initial_state = ParticleState(
        particles=initial_particles,
        log_weights=initial_log_weights,
        metadata={"temperature": 0.0},
    )

    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_proposal=log_proposal,
    )

    smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(posterior, target_ess_ratio=0.8),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(posterior, n_steps=8),
            LoggingStep(str_fn=logging_fn, frequency=1),
        ],
    )

    results = smc.run(
        initial_state,
        rng=rng,
        shared_context={
            "mu": mu,
            "sigma": sigma,
            "prior_lower": prior_lower,
            "prior_upper": prior_upper,
            "proposal_mean": proposal_mean,
            "proposal_sigma": proposal_sigma,
        },
    )

    final_state = results.final_state
    log_evidence = final_state.log_evidence
    print(f"Estimated log evidence: {log_evidence:.3f}")

    posterior_mean = final_state.particles.mean(axis=0)
    posterior_cov = np.cov(final_state.particles, rowvar=False)
    print(f"Posterior mean estimate: {np.round(posterior_mean, 3)}")
    print("Posterior covariance estimate:")
    print(np.round(posterior_cov, 3))

    # Plot temperature and log-evidence history
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    results.plot_metadata_history("temperature", ax=axes[0])
    results.plot_log_evidence(ax=axes[1])
    plt.tight_layout()
    fig.savefig("adaptive_temperature_proposal_history.png")

    # Plot initial and final particle distributions on the same axis
    fig, ax = plt.subplots()
    ax.scatter(
        initial_state.particles[:, 0],
        initial_state.particles[:, 1],
        color="C0",
        label="Initial particles",
        alpha=0.5,
    )
    ax.scatter(
        final_state.particles[:, 0],
        final_state.particles[:, 1],
        color="C1",
        label="Final particles",
        alpha=0.5,
    )
    ax.legend()
    fig.savefig("adaptive_particle_distributions.png")

    # Plot the intermediate 1-d marginal distributions
    fig, axes = plt.subplots(dims, 1, figsize=(6, 3 * dims))
    for d in range(dims):
        axes[d].hist(
            initial_state.particles[:, d],
            bins=16,
            density=True,
            label="Initial",
            histtype="step",
            color="grey",
        )
        axes[d].hist(
            final_state.particles[:, d],
            bins=16,
            density=True,
            label="Final",
            histtype="step",
            color="black",
        )
        for state in results.states[1:-1]:
            temp = state.metadata.get("temperature", 0.0)
            axes[d].hist(
                state.particles[:, d],
                bins=16,
                density=True,
                color=plt.cm.viridis(temp),
                histtype="step",
                ls="--",
            )
        axes[d].set_title(f"Dimension {d + 1}")
        axes[d].legend()
    plt.tight_layout()
    fig.savefig("adaptive_marginal_distributions.png")


if __name__ == "__main__":
    main()
