import logging
from unittest.mock import Mock

import numpy as np
import pytest

from sequince.steps.log import LoggingStep
from sequince.steps.weighting import IncrementalWeighting


@pytest.mark.parametrize("enabled", [False, True])
def test_logging_step_only_builds_enabled_messages(
    enabled, particle_state, context_factory, rng, caplog
) -> None:
    caplog.set_level(
        logging.DEBUG if enabled else logging.INFO,
        logger="sequince.steps.log",
    )
    callback = Mock(return_value="diagnostic")
    context = context_factory()
    result = LoggingStep(callback, level=logging.DEBUG).apply(
        particle_state, context, rng
    )
    assert callback.call_count == int(enabled)
    assert ("diagnostic" in caplog.text) == enabled
    assert result == (particle_state, context, None)


@pytest.mark.parametrize("enabled", [False, True])
def test_weighting_only_converts_debug_diagnostics_when_enabled(
    enabled, particle_state, context_factory, rng, caplog, monkeypatch
) -> None:
    import sequince.steps.weighting as weighting

    caplog.set_level(
        logging.DEBUG if enabled else logging.INFO,
        logger=weighting.__name__,
    )
    conversion = Mock(wraps=weighting.to_numpy)
    monkeypatch.setattr(weighting, "to_numpy", conversion)
    step = IncrementalWeighting(lambda particles: np.ones(len(particles)))
    updated, _, _ = step.apply(particle_state, context_factory(), rng)
    assert conversion.call_count == int(enabled)
    assert ("mean increment 1.000000" in caplog.text) == enabled
    np.testing.assert_allclose(updated.weights, particle_state.weights)
