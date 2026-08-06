from __future__ import annotations

import emcee
import numpy as np
import pytest
from orng import ArrayRNG

from sequince.steps.mutation import (
    EmceeMutation,
    get_emcee_move,
    resolve_emcee_moves,
)


def test_emcee_mutation_uses_supplied_rng(
    particle_state, context_factory, standard_normal_log_prob
) -> None:
    step = EmceeMutation(standard_normal_log_prob, n_steps=3, move="bilby-de")
    first, _, _ = step(
        particle_state,
        context_factory(),
        ArrayRNG(seed=2468),
    )
    second, _, _ = step(
        particle_state,
        context_factory(),
        ArrayRNG(seed=2468),
    )

    np.testing.assert_array_equal(first.particles, second.particles)
    assert (
        first.metadata["emcee_acceptance"]
        == second.metadata["emcee_acceptance"]
    )


def test_emcee_mutation_resolves_bilby_de_move() -> None:
    step = EmceeMutation(lambda position: 0.0, move="bilby-de")

    assert len(step._move) == 2  # noqa: SLF001
    assert isinstance(step._move[0][0], emcee.moves.DEMove)  # noqa: SLF001
    assert step._move[0][1] == 0.9  # noqa: SLF001
    assert step._move[1][0].gamma0 == 1.0  # noqa: SLF001
    assert step._move[1][1] == 0.1  # noqa: SLF001


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
