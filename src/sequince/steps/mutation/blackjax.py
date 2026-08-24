from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from array_api_compat import is_jax_namespace

from ..._typing import Array
from ...array_api import asarray, namespace, to_numpy
from ...transforms import (
    ParticleTransform,
    has_particle_aligned_auxiliary,
    select_transform_auxiliary,
)
from ...utils.pytrees import pytree_dataclass
from ..context import StepContext
from .base import MutationStep, TransformRefit

if TYPE_CHECKING:
    from orng import RandomGenerator

    from ...state import ParticleState
    from ...target import Target

logger = logging.getLogger(__name__)

BlackJAXAlgorithmFactory = Callable[..., Any]


@pytree_dataclass(
    data_fields=("acceptance", "divergence", "chain"),
    meta_fields=("algorithm", "steps"),
)
@dataclass(frozen=True)
class BlackJAXOutput(Mapping[str, Any]):
    """Fixed-structure BlackJAX diagnostics returned by a mutation step."""

    acceptance: Any
    divergence: Any
    algorithm: str
    steps: int
    chain: Any

    def __getitem__(self, key: str) -> Any:
        if key not in {
            "acceptance",
            "divergence",
            "algorithm",
            "steps",
            "chain",
        }:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(
            ("acceptance", "divergence", "algorithm", "steps", "chain")
        )

    def __len__(self) -> int:
        return 5


class BlackJAXMutation(MutationStep):
    """Run independent BlackJAX chains to rejuvenate a JAX particle cloud.

    Parameters
    ----------
    log_prob_fn
        JAX-compatible scalar log-probability function or a
        :class:`~sequince.target.Target` whose component functions
        are JAX compatible.
    n_steps
        Number of BlackJAX transitions performed for every particle.
    algorithm
        Named algorithm (``"nuts"``, ``"hmc"``, or ``"random-walk"``), or a
        callable that accepts the scalar log-density function and
        ``algorithm_kwargs`` and returns a BlackJAX ``SamplingAlgorithm``.
    algorithm_kwargs
        Options passed to the selected BlackJAX algorithm. NUTS and HMC default
        to a step size of ``1e-3`` and a unit diagonal inverse mass matrix. HMC
        additionally defaults to 10 integration steps. The Gaussian random walk
        accepts ``sigma`` and defaults to ``0.1``.
    transform
        Optional particle transform applied before mutation.
    transform_refit
        Policy controlling when the particle transform is refitted.
    store_chains
        Whether to include all BlackJAX positions in the step output.

    Notes
    -----
    The mutation derives a fresh JAX key from Sequince's supplied RNG on every
    application. Consequently checkpointing and resumption do not require a
    separate BlackJAX RNG state.
    """

    def __init__(
        self,
        log_prob_fn: Callable[..., Array | float] | "Target",
        *,
        n_steps: int = 1,
        algorithm: str | BlackJAXAlgorithmFactory = "nuts",
        algorithm_kwargs: Mapping[str, Any] | None = None,
        transform: ParticleTransform | None = None,
        transform_refit: TransformRefit = "always",
        store_chains: bool = False,
    ) -> None:
        super().__init__(
            log_prob_fn,
            transform=transform,
            transform_refit=transform_refit,
            store_chains=store_chains,
        )
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if not isinstance(algorithm, str) and not callable(algorithm):
            raise TypeError("algorithm must be a name or callable factory.")
        if isinstance(algorithm, str) and algorithm not in {
            "hmc",
            "nuts",
            "random-walk",
            "random_walk",
            "rmh",
        }:
            raise ValueError(
                "algorithm must be 'nuts', 'hmc', 'random-walk', or a callable."
            )
        self._n_steps = n_steps
        self._algorithm = algorithm
        self._algorithm_kwargs = dict(algorithm_kwargs or {})

    @property
    def algorithm_name(self) -> str:
        """Human-readable name of the configured BlackJAX algorithm."""
        if isinstance(self._algorithm, str):
            if self._algorithm in {"random_walk", "rmh"}:
                return "random-walk"
            return self._algorithm
        return getattr(
            self._algorithm, "__name__", type(self._algorithm).__name__
        )

    def _build_algorithm(
        self, logdensity_fn: Callable, dimensions: int
    ) -> Any:
        try:
            import blackjax
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "BlackJAX mutation requires the 'blackjax' extra; install it "
                "with `pip install 'sequince[blackjax]'`."
            ) from exc
        import jax.numpy as jnp

        kwargs = dict(self._algorithm_kwargs)
        if callable(self._algorithm):
            return self._algorithm(logdensity_fn, **kwargs)

        name = self.algorithm_name
        if name == "random-walk":
            sigma = kwargs.pop("sigma", 0.1)
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise ValueError(
                    f"Unknown random-walk algorithm options: {unknown}."
                )
            return blackjax.normal_random_walk(logdensity_fn, sigma=sigma)

        kwargs.setdefault("step_size", 1e-3)
        if kwargs.get("inverse_mass_matrix") is None:
            kwargs["inverse_mass_matrix"] = jnp.ones(dimensions)
        if name == "hmc":
            kwargs.setdefault("num_integration_steps", 10)
            return blackjax.hmc(logdensity_fn, **kwargs)
        return blackjax.nuts(logdensity_fn, **kwargs)

    @staticmethod
    def _jax_key(rng: "RandomGenerator") -> Any:
        import jax

        if rng.backend.lower() == "jax":
            key, next_state = jax.random.split(rng.state())
            rng.set_state(next_state)
            return key

        draw = float(to_numpy(rng.random((1,)))[0])
        seed = int(draw * np.iinfo(np.uint32).max)
        return jax.random.key(seed)

    def apply(
        self,
        state: "ParticleState",
        context: StepContext,
        rng: "RandomGenerator",
    ) -> tuple["ParticleState", StepContext, BlackJAXOutput]:
        import jax
        import jax.numpy as jnp

        xp = namespace(state.particles)
        if not is_jax_namespace(xp):
            raise TypeError("BlackJAXMutation requires JAX-backed particles.")

        transform, context = self._prepare_transform(
            state.particles, context, rng=rng
        )
        latent_particles, transform_auxiliary = self._forward_transform(
            state.particles, transform, context, rng=rng
        )
        latent_particles = asarray(latent_particles, xp=xp)
        squeeze_last = latent_particles.ndim == 1
        initial_positions = (
            xp.reshape(latent_particles, (-1, 1))
            if squeeze_last
            else latent_particles
        )
        n_particles, dimensions = initial_positions.shape

        def logdensity_fn(position: Array, particle_index: Array) -> Array:
            particle_auxiliary = select_transform_auxiliary(
                transform_auxiliary,
                particle_index,
                n_particles=n_particles,
            )
            original, log_det = self._inverse_transform(
                position,
                transform,
                context,
                auxiliary=particle_auxiliary,
            )
            result, _ = self._evaluate_target(original, context)
            result = self._combine_with_log_det(result, log_det)
            scalar = jnp.reshape(jnp.asarray(result), ())
            return jnp.where(jnp.isnan(scalar), -jnp.inf, scalar)

        particle_aligned = has_particle_aligned_auxiliary(
            transform_auxiliary, n_particles=n_particles
        )
        particle_indices = jnp.arange(n_particles)
        shared_algorithm: Any = None
        if particle_aligned:

            def initialize_particle(position: Array, particle_index: Array):
                algorithm = self._build_algorithm(
                    lambda proposed: logdensity_fn(proposed, particle_index),
                    dimensions,
                )
                return algorithm.init(position)

            states = jax.vmap(initialize_particle)(
                initial_positions, particle_indices
            )
        else:
            shared_algorithm = self._build_algorithm(
                lambda proposed: logdensity_fn(proposed, jnp.asarray(0)),
                dimensions,
            )
            states = jax.vmap(shared_algorithm.init)(initial_positions)
        step_keys = jax.random.split(self._jax_key(rng), self._n_steps)

        def one_step(states: Any, key: Any) -> tuple[Any, tuple[Any, Any]]:
            chain_keys = jax.random.split(key, n_particles)

            def step_particle(
                particle_key: Any,
                particle_state: Any,
                particle_index: Array,
            ):
                algorithm = self._build_algorithm(
                    lambda proposed: logdensity_fn(proposed, particle_index),
                    dimensions,
                )
                return algorithm.step(particle_key, particle_state)

            if particle_aligned:
                updated, info = jax.vmap(step_particle)(
                    chain_keys, states, particle_indices
                )
            else:
                updated, info = jax.vmap(shared_algorithm.step)(
                    chain_keys, states
                )
            return updated, (updated, info)

        final_states, (chain_states, infos) = jax.lax.scan(
            one_step, states, step_keys
        )
        latent_result = final_states.position
        if squeeze_last:
            latent_result = latent_result[:, 0]
        new_particles, _ = self._inverse_transform(
            latent_result,
            transform,
            context,
            auxiliary=transform_auxiliary,
        )
        _, final_evaluation = self._evaluate_target(new_particles, context)

        acceptance = getattr(infos, "acceptance_rate", None)
        if acceptance is None:
            acceptance = getattr(infos, "is_accepted", None)
        if acceptance is not None:
            acceptance = jnp.mean(acceptance)
        divergence = getattr(infos, "is_divergent", None)
        if divergence is not None:
            divergence = jnp.mean(divergence)
        stored_chain = None
        if self._store_chains:
            stored_chain = self._inverse_chain(
                chain_states.position,
                transform,
                context,
                auxiliary=transform_auxiliary,
                like=initial_positions,
            )

        logger.debug(
            "BlackJAXMutation completed with %s after %d steps.",
            self.algorithm_name,
            self._n_steps,
        )
        output = BlackJAXOutput(
            acceptance=acceptance,
            divergence=divergence,
            algorithm=self.algorithm_name,
            steps=self._n_steps,
            chain=stored_chain,
        )
        return (
            state.replace_particles(
                new_particles,
                evaluation=final_evaluation,
            ),
            context,
            output,
        )
