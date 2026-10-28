"""Core sampling must work without optional mutation packages."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


def test_minipcn_is_an_optional_dependency() -> None:
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )["project"]
    assert not any(
        dep.startswith("minipcn") for dep in metadata["dependencies"]
    )
    assert metadata["optional-dependencies"]["minipcn"] == ["minipcn>=0.3.0a1"]


def test_core_sampling_without_minipcn() -> None:
    # A fresh interpreter ensures no previously imported MiniPCN can mask a
    # missing dependency. The finder also blocks any MiniPCN submodule import.
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class MissingMiniPCN(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "minipcn" or fullname.startswith("minipcn."):
                    raise ModuleNotFoundError(
                        "No module named 'minipcn'", name="minipcn"
                    )

        sys.meta_path.insert(0, MissingMiniPCN())

        import numpy as np
        from sequince import ParticleState, make_mcmc, make_standard_bayesian_smc
        from sequince.steps import (
            AdaptiveMiniPCNMutation, MiniPCNMutation, StepContext,
        )

        def log_prob(x, context=None):
            return -0.5 * np.sum(x**2, axis=-1)

        particles = np.random.default_rng(42).normal(size=(32, 2))
        result = make_mcmc(log_prob, n_steps=3).run(
            ParticleState(particles.copy()), rng=43
        )
        assert result.chain.shape == (3, 32, 2)
        assert np.isfinite(result.chain).all()

        sampler, _, metadata = make_standard_bayesian_smc(
            log_prob, log_prob, logging_frequency=0
        )
        result = sampler.run(
            ParticleState(particles.copy()), rng=44, metadata=metadata
        )
        assert result.context.metadata["beta"] == 1
        assert np.isfinite(result.final_state.particles).all()
        assert not any(
            name == "minipcn" or name.startswith("minipcn.")
            for name in sys.modules
        )

        mutation = MiniPCNMutation(log_prob)
        for use_minipcn in (
            lambda: mutation(ParticleState(particles), StepContext(t=0), 45),
            lambda: AdaptiveMiniPCNMutation(mutation),
        ):
            try:
                use_minipcn()
            except ImportError as exc:
                assert "pip install 'sequince[minipcn]'" in str(exc), str(exc)
            else:
                raise AssertionError("MiniPCN must require the optional extra")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
