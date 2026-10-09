from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sequince import ParticleState
from sequince.steps import StepContext, apply_functional_if


@pytest.mark.parametrize("execute", [False, True])
def test_conditional_step_backends(
    backend: str, backend_factory: Any, execute: bool
) -> None:
    case = backend_factory(backend)
    xp = case.xp
    state = case.state()
    context = StepContext(t=0)
    key = xp.asarray(12)

    def apply(state: ParticleState, context: StepContext, key: Any) -> Any:
        return (
            state.with_updates(particles=state.particles + 1),
            context,
            key + 1,
            xp.asarray(1),
        )

    result, result_context, result_key, output = apply_functional_if(
        xp.asarray(execute), apply, state, context, key
    )
    np.testing.assert_allclose(
        np.asarray(result.particles),
        np.asarray(state.particles) + int(execute),
    )
    assert int(result_key) == 12 + int(execute)
    assert result_context.t == context.t
    if execute:
        assert int(output) == 1
    elif backend == "jax":
        assert int(output) == 0
    else:
        assert output is None


@pytest.mark.parametrize("execute", [False, True])
def test_eager_conditional_only_calls_selected_branch(execute: bool) -> None:
    state = ParticleState(np.zeros((4, 1)))
    context = StepContext(t=0)
    calls = []

    def apply(state: ParticleState, context: StepContext, key: Any) -> Any:
        calls.append(True)
        return state, context, key + 1, {"executed": True}

    _, _, key, output = apply_functional_if(execute, apply, state, context, 42)
    assert calls == ([True] if execute else [])
    assert key == 42 + int(execute)
    assert output == ({"executed": True} if execute else None)
