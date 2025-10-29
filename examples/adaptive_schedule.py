import matplotlib.pyplot as plt
import numpy as np

from sequince import ParticleState, SequentialMonteCarlo, TemperedPosterior
from sequince.steps import (
    AdaptiveTemperedWeighting,
    EmceeMutation,
    SystematicResampling,
)

# Number of dimensions in the multivariate normal model
dims = 4
# Mean and standard deviation of the target normal distribution
mu = 0.0
sigma = 1.0
# Prior bounds
bounds = (-10, 10)
# True log evidence for the model with uniform prior over the bounds
true_log_evidence = -dims * np.log(np.ptp(bounds))


def log_likelihood(particles: np.ndarray, context) -> np.ndarray:
    """Log-likelihood for a multivariate standard normal observation."""
    return -0.5 * np.sum(np.square(particles - mu) / sigma**2, axis=-1) - (
        dims / 2
    ) * np.log(2 * np.pi * sigma**2)


def log_prior(particles: np.ndarray, context) -> np.ndarray:
    """Uniform prior over a bounded region."""
    lower_bound = context.static["lower_bound"]
    upper_bound = context.static["upper_bound"]
    return np.where(
        (particles >= lower_bound) & (particles <= upper_bound), 0.0, -np.inf
    ).sum(axis=-1)


def main() -> None:
    rng = np.random.default_rng(seed=123)

    # Prior over the static mean parameter
    lower_bound, upper_bound = bounds
    n_particles = 200
    initial_particles = rng.uniform(
        low=lower_bound, high=upper_bound, size=(n_particles, dims)
    )
    initial_log_weights = np.full(n_particles, -np.log(n_particles))
    initial_state = ParticleState(
        particles=initial_particles, log_weights=initial_log_weights
    )

    # Small helper to create the tempered posterior
    # This allows the different steps to use either the log prior, log likelihood,
    # or the full tempered posterior as needed.
    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
    )

    smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(posterior, target_ess_ratio=0.8),
            SystematicResampling(always_resample=True),
            EmceeMutation(posterior, n_steps=20, vectorize=True),
        ]
    )

    results = smc.run(
        initial_state,
        rng=rng,
        static_context={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "temperature": 0.0,
        },
    )
    posterior_state = results.final_state

    log_evidence = posterior_state.log_evidence
    print(f"Estimated log evidence: {log_evidence:.4f}")
    print(f"True log evidence:      {true_log_evidence:.4f}")

    # Analyze the posterior particles
    particles = posterior_state.particles
    mean_estimate = np.mean(particles, axis=0)
    cov_estimate = np.cov(particles, rowvar=False)
    print(f"Posterior mean estimate: {mean_estimate}")
    print(f"Posterior covariance estimate:\n{cov_estimate}")

    fig, axes = plt.subplots(2, 1)
    results.plot_log_evidence(ax=axes[1])
    results.plot_metadata_history("temperature", ax=axes[0])
    fig.savefig("adaptive_temperature_history.png")


if __name__ == "__main__":
    main()
