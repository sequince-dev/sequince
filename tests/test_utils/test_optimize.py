import numpy as np
import pytest

from sequince.utils.optimize import bisect_monotone_decreasing


def test_bisection_refines_bracket_without_endpoint_policy() -> None:
    evaluated: list[float] = []

    def criterion(value: np.ndarray) -> np.ndarray:
        evaluated.append(float(value))
        return 1.0 - value

    low, high = bisect_monotone_decreasing(
        criterion,
        np.asarray(0.0),
        np.asarray(1.0),
        np.asarray(0.3),
        tolerance=np.asarray(1e-8),
        max_iterations=40,
    )

    assert float(low) <= 0.7 <= float(high)
    assert float(high - low) <= 1e-8
    assert 0.0 not in evaluated
    assert 1.0 not in evaluated


def test_bisection_uses_backend_native_torch_condition() -> None:
    torch = pytest.importorskip("torch")

    low, high = bisect_monotone_decreasing(
        lambda value: 1.0 - value,
        torch.asarray(0.0),
        torch.asarray(1.0),
        torch.asarray(0.3),
        tolerance=torch.asarray(1e-6),
        max_iterations=40,
    )

    assert float(low) <= 0.7 <= float(high)
    assert float(high - low) <= 1e-6
