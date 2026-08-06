"""Run a JAX-backed standard Bayesian SMC sampler.

This example demonstrates how to use sequince's functional API with JAX to
JIT-compile the SMC iteration.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sequince import ParticleState, make_standard_bayesian_smc
from sequince.jax import compile_functional_iteration
from sequince.log import configure_logging
from sequince.steps import StepContext


@dataclass(frozen=True)
class RosenbrockModel:
    """Rosenbrock target with a uniform prior on a bounded box."""

    n_dimensions: int
    lower_bound: float = -5.0
    upper_bound: float = 5.0

    def log_prior(
        self, particles: jax.Array, context: StepContext
    ) -> jax.Array:
        """Evaluate the uniform prior on ``[lower_bound, upper_bound]``."""
        del context
        in_bounds = jnp.all(
            (particles >= self.lower_bound) & (particles <= self.upper_bound),
            axis=-1,
        )
        log_density = -self.n_dimensions * jnp.log(
            self.upper_bound - self.lower_bound
        )
        return jnp.where(in_bounds, log_density, -jnp.inf)

    def log_likelihood(
        self, particles: jax.Array, context: StepContext
    ) -> jax.Array:
        """Evaluate the negative Rosenbrock function."""
        del context
        current = particles[..., :-1]
        following = particles[..., 1:]
        return -jnp.sum(
            100.0 * (following - current**2) ** 2 + (1.0 - current) ** 2,
            axis=-1,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample a bounded Rosenbrock target with JAX-backed SMC."
    )
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        default=2,
        help="number of Rosenbrock dimensions (default: 2)",
    )
    args = parser.parse_args()
    if args.dimensions < 2:
        parser.error("--dimensions must be at least 2")
    return args


def main(n_dimensions: int) -> None:
    """Sample the Rosenbrock target."""
    configure_logging()

    n_particles = 500
    model = RosenbrockModel(n_dimensions=n_dimensions)

    random_state = jax.random.key(42)
    particle_key, random_state = jax.random.split(random_state)
    particles = jax.random.uniform(
        particle_key,
        shape=(n_particles, model.n_dimensions),
        minval=model.lower_bound,
        maxval=model.upper_bound,
    )
    initial_state = ParticleState(particles=particles)

    smc, _, metadata = make_standard_bayesian_smc(
        log_prior=model.log_prior,
        log_likelihood=model.log_likelihood,
        target_ess_ratio=0.8,
        always_resample=True,
        mutation="blackjax",
        mutation_n_steps=10,
        mutation_kwargs={
            "algorithm": "nuts",
            "algorithm_kwargs": {
                "step_size": 0.02,
                "inverse_mass_matrix": jnp.ones(model.n_dimensions),
            },
        },
        particle_transform=None,  # Particle transforms aren't compatible with JIT yet.
        logging_frequency=0,
    )

    # We use a helper function to compile the functional iteration. This jax.jit
    # and the lowers the functional iteration to XLA. The compiled function can then be used in
    # smc.run_functional to perform the SMC iterations with JIT compilation.
    # Alternatively, you can call ``jax.jit(smc.functional_iteration)`` directly,
    # but this will not warm up the JIT compiler and may result in a slower first
    # iteration.
    print("Compiling JAX-backed standard SMC sampler")
    compile_start_time = time.time()
    iteration = compile_functional_iteration(
        smc,
        initial_state,
        random_state,
        metadata=metadata,
    )
    compile_end_time = time.time()
    print(
        f"Compiled functional iteration in "
        f"{compile_end_time - compile_start_time:.3f} seconds"
    )

    print("Running JAX-backed standard SMC sampler")
    start_time = time.time()
    result, final_random_state = smc.run_functional(
        initial_state,
        random_state=random_state,
        metadata=metadata,
        iteration=iteration,
    )
    jax.block_until_ready(result.states[-1].particles)
    end_time = time.time()
    print(
        f"Completed {len(result)} adaptive temperature steps in "
        f"{end_time - start_time:.3f} seconds"
    )

    print(f"Array backend: {jax.default_backend()}")
    print(f"Dimensions: {model.n_dimensions}")
    print(f"Adaptive temperature steps: {len(result)}")
    mutation_output = result.outputs[-1][-1]
    print(f"Final BlackJAX acceptance: {mutation_output['acceptance']:.3f}")

    corner_filename = f"rosenbrock_{model.n_dimensions}d_corner.png"
    result.plot_corner(
        include_weights=True,
        filename=corner_filename,
    )


if __name__ == "__main__":
    main(parse_args().dimensions)
