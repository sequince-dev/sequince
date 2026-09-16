"""Use the actual sibling Bilby checkout in isolated backend processes."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_local_bilby(backend: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    checkout = Path(__file__).resolve().parents[3] / "bilby"
    if not (checkout / "bilby" / "__init__.py").exists():
        pytest.skip("The sibling Bilby development checkout is not available")
    if importlib.util.find_spec("bilby") is None:
        pytest.skip(
            "Bilby optional integration dependencies are not installed"
        )
    example = (
        Path(__file__).resolve().parents[2] / "docs/examples/pyro_bilby.py"
    )
    # Bilby selects Array API support at import. Subprocesses exercise the
    # public environment mechanism without changing the parent test backend.
    env = dict(os.environ, SCIPY_ARRAY_API="1")
    env["BILBY_ARRAY_API"] = "1" if backend == "torch" else ""
    env["PYTHONPATH"] = str(checkout) + os.pathsep + env.get("PYTHONPATH", "")
    script = """
import pathlib, runpy, sys
import bilby, torch
from sequince import ParticleState
from sequince.steps import PyroNUTSMutation, StepContext
assert pathlib.Path(bilby.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve() / 'bilby/__init__.py'
print('Local Bilby:', bilby.__file__)
example = runpy.run_path(sys.argv[2])
target = example['make_target']()
context = StepContext(t=0, metadata={'beta': 0.4})
x = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
mutation = PyroNUTSMutation(target, n_steps=2, step_size=0.25, max_tree_depth=4)
state = ParticleState(torch.tensor([[-0.5], [0.3], [1.2]], dtype=x.dtype))
if sys.argv[3] == 'numpy':
    try:
        mutation.apply(state, context, 42)
    except TypeError as error:
        assert 'Bilby Torch backend required' in str(error)
    else:
        raise AssertionError('NumPy-backed Bilby must fail')
    sys.exit(0)
ll = example['TorchLikelihood']().log_likelihood({'x': x[0]})
assert isinstance(ll, torch.Tensor) and ll.ndim == 0 and ll.requires_grad
assert torch.isfinite(torch.autograd.grad(ll, x)[0]).all()
value = target.log_prob(x, context)
assert value.ndim == 0 and value.requires_grad
expected_gradient = -x / 4 - 0.4 * 2 * (2*x-1)
torch.testing.assert_close(torch.autograd.grad(value, x)[0], expected_gradient)
with torch.no_grad():
    result, _, output = mutation.apply(state, context, 42)
assert not torch.equal(result.particles, state.particles)
assert torch.equal(result.log_weights, state.log_weights)
assert result.particles.dtype == state.particles.dtype
assert result.particles.device == state.particles.device
fresh = target.prepare_evaluation(result.particles, context)
for name in ('log_prior', 'log_likelihood'):
    torch.testing.assert_close(result.evaluation[name], fresh[name])
example['main']()
"""
    subprocess.run(
        [sys.executable, "-c", script, str(checkout), str(example), backend],
        env=env,
        check=True,
    )
