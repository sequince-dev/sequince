import numpy as np
import pytest

from sequince.stats import ess_from_log_weights


def test_ess_returns_nonfinite_for_degenerate_log_weights():
    assert not np.isfinite(ess_from_log_weights(np.array([-np.inf, -np.inf])))


def test_ess_can_be_jitted():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    ess = jax.jit(ess_from_log_weights)(jnp.asarray([0.0, -1.0]))

    assert ess == pytest.approx(1.6480542)
