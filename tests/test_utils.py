import numpy as np
import pytest

from sequince import DegenerateWeightsError
from sequince.stats import ess_from_log_weights
from sequince.utils import in_bounds, normalize_log_weights


def test_normalize_log_weights():
    log_weights = np.log([1.0, 3.0]).tolist()

    normalized, log_norm = normalize_log_weights(log_weights)

    np.testing.assert_allclose(np.exp(normalized), [0.25, 0.75])
    assert np.isclose(log_norm, np.log(4.0))


@pytest.mark.parametrize(
    ("log_weights", "message"),
    [
        ([-np.inf, -np.inf], "zero weight"),
        ([0.0, np.nan], "NaN"),
        ([0.0, np.inf], "positive infinity"),
    ],
)
def test_normalize_degenerate_log_weights_raises(log_weights, message):
    with pytest.raises(DegenerateWeightsError, match=message) as exc_info:
        normalize_log_weights(log_weights)

    assert not np.isfinite(exc_info.value.log_normalizer)


def test_ess_rejects_degenerate_log_weights():
    with pytest.raises(DegenerateWeightsError, match="zero weight"):
        ess_from_log_weights(np.array([-np.inf, -np.inf]))


def test_in_bounds():
    particles = np.array(
        [
            [0.0, -1.0, 0.25],
            [1.0, 2.0, 0.5],
            [-1.5, 0.5, 0.9],
        ],
        dtype=float,
    )
    bounds = np.array(
        [
            [-1.0, -2.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    expected = np.array([True, True, False])
    result = in_bounds(particles, bounds)
    np.testing.assert_array_equal(result, expected)
