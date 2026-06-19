from __future__ import annotations

import numpy as np

from sequince.posterior import draw_additional_samples
from sequince.result import SMCResult
from sequince.state import ParticleState


def test_draw_additional_samples_respects_terminal_weights(
    context_factory,
    rng,
) -> None:
    particles = np.array([[0.0], [1.0], [2.0]], dtype=float)
    state = ParticleState(
        particles=particles,
        log_weights=np.array([0.0, -1.0e9, -1.0e9]),
        metadata={"temperature": 1.0},
    )
    result = SMCResult([state], context_factory(metadata={"temperature": 1.0}))

    updated_state, updated_result = draw_additional_samples(
        result=result,
        rng=rng,
        n_particles=8,
        mutation="none",
    )

    assert updated_state.particles.shape == (8, 1)
    assert np.all(updated_state.particles == 0.0)
    assert updated_state.log_weights.shape == (8,)
    assert np.allclose(updated_state.weights, np.full(8, 1.0 / 8.0))
    assert updated_result.final_state is updated_state
