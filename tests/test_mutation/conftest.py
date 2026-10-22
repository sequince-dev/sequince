import numpy as np
import pytest

from sequince.steps.base import SMCStep


def _flat_log_prob(particles, context):
    del context
    return np.zeros(particles.shape[0])


class NegateStep(SMCStep):
    def apply(self, state, context, rng):
        del rng
        return state.replace_particles(-state.particles), context, None


@pytest.fixture
def flat_log_prob():
    return _flat_log_prob


@pytest.fixture
def negate_step():
    return NegateStep()
