import numpy as np
import pytest

from sequince.stats import (
    cess_ratio_from_log_weights_and_increments,
    ess_from_log_weights,
    ess_ratio_from_log_weights,
)


def test_ess_returns_nonfinite_for_degenerate_log_weights():
    assert not np.isfinite(ess_from_log_weights(np.array([-np.inf, -np.inf])))


def test_ess_can_be_jitted():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    ess = jax.jit(ess_from_log_weights)(jnp.asarray([0.0, -1.0]))

    assert ess == pytest.approx(1.6480542)


@pytest.mark.parametrize("shift", [-1000.0, 0.0, 1000.0])
def test_ess_uses_relative_weights_including_zero_support(
    shift: float,
) -> None:
    # Normalized weights are (1/4, 1/4, 1/2, 0): sum W_i^2 = 3/8.
    log_weights = np.array([0.0, 0.0, np.log(2.0), -np.inf]) + shift
    assert ess_from_log_weights(log_weights) == pytest.approx(8 / 3)
    assert ess_ratio_from_log_weights(log_weights) == pytest.approx(2 / 3)


@pytest.mark.parametrize("shift", [-1000.0, 0.0, 1000.0])
def test_cess_differs_from_ess_for_unequal_initial_weights(
    shift: float,
) -> None:
    log_weights = np.log([0.25, 0.75])
    increments = np.log([2.0, 1.0]) + shift
    # E_W[g] = 5/4, E_W[g^2] = 7/4; final weights are (2/5, 3/5).
    assert cess_ratio_from_log_weights_and_increments(
        log_weights, increments
    ) == pytest.approx(25 / 28)
    assert ess_ratio_from_log_weights(
        log_weights + increments
    ) == pytest.approx(25 / 26)
