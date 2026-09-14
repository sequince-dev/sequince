from __future__ import annotations

import emcee
import numpy as np
import pytest
from orng import ArrayRNG

from sequince.state import ParticleState
from sequince.steps.context import StepContext
from sequince.steps.mutation import (
    EmceeMutation,
    get_emcee_move,
    resolve_emcee_moves,
)
from sequince.steps.waste_free import WasteFreeResampleMove
from sequince.transforms import AffineTransform, ParticleTransform


@pytest.mark.parametrize("scalar_particles", [False, True])
@pytest.mark.parametrize("transform", [None, AffineTransform()])
@pytest.mark.parametrize("vectorize", [False, True])
def test_emcee_chain_matches_particle_shape(
    scalar_particles, transform, vectorize, standard_normal_log_prob
) -> None:
    particles = np.linspace(-2.0, 2.0, 20)
    if not scalar_particles:
        particles = particles[:, None]
    step = EmceeMutation(
        standard_normal_log_prob,
        n_steps=3,
        store_chains=True,
        transform=transform,
        vectorize=vectorize,
    )

    result, _, output = step(
        ParticleState(particles), StepContext(t=0), rng=123
    )

    assert result.particles.shape == particles.shape
    assert output["chain"].shape == (3, *particles.shape)
    np.testing.assert_array_equal(output["chain"][-1], result.particles)


def test_waste_free_emcee_accepts_scalar_chains(
    standard_normal_log_prob,
) -> None:
    particles = np.linspace(-2.0, 2.0, 20)
    step = WasteFreeResampleMove(
        EmceeMutation(standard_normal_log_prob), chain_length=2
    )

    result, _, output = step(
        ParticleState(particles), StepContext(t=0), rng=123
    )

    assert result.particles.shape == particles.shape
    np.testing.assert_array_equal(
        result.particles[10:], output["mutation"]["chain"][0]
    )
    np.testing.assert_allclose(result.weights, 1.0 / particles.size)


class ParticleAlignedAuxiliaryTransform(ParticleTransform):
    def forward(self, particles, context=None, *, rng=None):
        del context, rng
        values = np.asarray(particles)
        return values, np.zeros(values.shape[0]), np.arange(values.shape[0])


class SharedAuxiliaryTransform(ParticleTransform):
    def forward(self, particles, context=None, *, rng=None):
        del context, rng
        values = np.asarray(particles)
        shift = np.asarray(1.0)
        return values - shift, np.zeros(values.shape[0]), {"shift": shift}

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        del context, rng
        values = np.asarray(values)
        return values + auxiliary["shift"], np.zeros_like(values[..., 0])


def test_emcee_mutation_uses_supplied_rng(
    particle_state, context_factory, standard_normal_log_prob
) -> None:
    step = EmceeMutation(standard_normal_log_prob, n_steps=3, move="bilby-de")
    first, _, first_output = step(
        particle_state,
        context_factory(),
        ArrayRNG(seed=2468),
    )
    second, _, second_output = step(
        particle_state,
        context_factory(),
        ArrayRNG(seed=2468),
    )

    np.testing.assert_array_equal(first.particles, second.particles)
    assert (
        first_output["metrics"].acceptance
        == second_output["metrics"].acceptance
    )
    assert first_output["steps"] == 3


def test_emcee_mutation_resolves_bilby_de_move() -> None:
    step = EmceeMutation(lambda position: 0.0, move="bilby-de")

    assert len(step._move) == 2  # noqa: SLF001
    assert isinstance(step._move[0][0], emcee.moves.DEMove)  # noqa: SLF001
    assert step._move[0][1] == 0.9  # noqa: SLF001
    assert step._move[1][0].gamma0 == 1.0  # noqa: SLF001
    assert step._move[1][1] == 0.1  # noqa: SLF001


def test_emcee_rejects_particle_aligned_transform_auxiliary(
    particle_state, context_factory, standard_normal_log_prob
) -> None:
    step = EmceeMutation(
        standard_normal_log_prob,
        transform=ParticleAlignedAuxiliaryTransform(),
    )

    with pytest.raises(
        NotImplementedError, match="walker-specific auxiliary data"
    ):
        step(particle_state, context_factory(), ArrayRNG(seed=2468))


def test_emcee_accepts_non_particle_aligned_transform_auxiliary(
    particle_state, context_factory, standard_normal_log_prob
) -> None:
    step = EmceeMutation(
        standard_normal_log_prob,
        transform=SharedAuxiliaryTransform(),
    )

    result, _, _ = step(particle_state, context_factory(), ArrayRNG(seed=2468))

    assert result.particles.shape == particle_state.particles.shape
    assert np.all(np.isfinite(result.particles))


@pytest.mark.parametrize("vectorize", [False, True])
def test_emcee_uses_blobs_for_final_target_evaluation(
    particle_state,
    context_factory,
    standard_normal_posterior,
    vectorize,
) -> None:
    n_steps = 3
    step = EmceeMutation(
        standard_normal_posterior,
        n_steps=n_steps,
        move="bilby-de",
        vectorize=vectorize,
    )

    result, context, _ = step(
        particle_state,
        context_factory(metadata={"temperature": 1.0}),
        ArrayRNG(seed=2468),
    )

    assert result.evaluation is not None
    expected = standard_normal_posterior.evaluate(result.particles, context)
    for component in ("log_prior", "log_likelihood"):
        np.testing.assert_allclose(
            result.evaluation[component], expected[component]
        )
    n_walkers = particle_state.particles.shape[0]
    assert context.evaluation_counts.log_target_evaluations == n_walkers * (
        n_steps + 1
    )


def test_get_emcee_move_constructs_named_move() -> None:
    move = get_emcee_move("de", gamma0=0.75, sigma=2e-5)

    assert isinstance(move, emcee.moves.DEMove)
    assert move.gamma0 == 0.75
    assert move.sigma == 2e-5


def test_resolve_emcee_moves_builds_bilby_de_preset() -> None:
    moves = resolve_emcee_moves("bilby-de")

    assert len(moves) == 2
    default_scale, default_weight = moves[0]
    mode_hop, mode_hop_weight = moves[1]
    assert isinstance(default_scale, emcee.moves.DEMove)
    assert default_scale.gamma0 is None
    assert default_scale.sigma == 1e-5
    assert default_weight == 0.9
    assert isinstance(mode_hop, emcee.moves.DEMove)
    assert mode_hop.gamma0 == 1.0
    assert mode_hop.sigma == 1e-5
    assert mode_hop_weight == 0.1


def test_resolve_emcee_moves_builds_weighted_configuration() -> None:
    moves = resolve_emcee_moves(
        [
            {
                "move": "de",
                "move_kwargs": {"gamma0": None, "sigma": 1e-5},
                "weight": 0.8,
            },
            {
                "move": "stretch",
                "move_kwargs": {"a": 2.5},
                "weight": 0.2,
            },
        ]
    )

    assert isinstance(moves[0][0], emcee.moves.DEMove)
    assert moves[0][1] == 0.8
    assert isinstance(moves[1][0], emcee.moves.StretchMove)
    assert moves[1][0].a == 2.5
    assert moves[1][1] == 0.2


def test_resolve_emcee_moves_preserves_move_object() -> None:
    move = emcee.moves.WalkMove(s=3)

    assert resolve_emcee_moves(move) is move


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("unknown", "Unknown emcee move"),
        ({"move_kwargs": {}}, "require a 'move'"),
        ([{"move": "de", "weight": -1.0}], "finite and non-negative"),
        ([{"move": "de", "weight": 0.0}], "must be positive"),
    ],
)
def test_resolve_emcee_moves_validates_configuration(spec, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_emcee_moves(spec)
