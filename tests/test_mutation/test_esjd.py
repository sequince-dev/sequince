"""Kernel-independent ESJD, rejection accounting, and segment continuity."""

from typing import Any

import numpy as np
import pytest

from sequince.array_api import asarray, get_device, namespace, to_numpy
from sequince.steps import ESJDMetric
from sequince.steps.context import StepContext
from sequince.steps.mutation import (
    DifferentialEvolutionMutation,
    GaussianRandomWalk,
    MetropolisHastingsMutation,
    MutationMetrics,
    MutationTransition,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("explicit_transition", [False, True])
def test_esjd_rejections_and_return_jumps(
    backend: str, explicit_transition: bool, backend_factory: Any
) -> None:
    case = backend_factory(backend)
    particles = case.particles
    metric = ESJDMetric()
    context = StepContext(t=0)
    state = metric.initialize(particles, case.log_weights, context)
    assert float(to_numpy(metric.result(state, context))) == 0
    jump = asarray([[3, 4], [0, 0], [0, 2], [0, 0]], like=particles)
    accepted = asarray(
        [True, False, True, False], like=particles, dtype=case.xp.bool
    )
    for previous, current in [
        (particles, particles + jump),
        (particles + jump, particles),
    ]:
        state = metric.update(
            state,
            current,
            accepted,
            context,
            transition=MutationTransition(current - previous)
            if explicit_transition
            else None,
        )
    result = metric.result(state, context)
    # Both outward and return transitions contribute, even though endpoint RJD=0.
    assert float(to_numpy(result)) == pytest.approx(29 / 4)
    assert namespace(result) is case.xp
    assert result.dtype == particles.dtype
    assert get_device(result) == get_device(particles)


def test_esjd_segments_count_attempts_and_exclude_external_moves() -> None:
    metric = ESJDMetric()
    context = StepContext(t=0)
    state = metric.initialize(np.zeros((2, 1)), np.zeros(2), context)
    state = metric.update(state, np.ones((2, 1)), np.ones(2, bool), context)
    state = metric.prepare(state, np.full((3, 1), 100.0), context)
    state = metric.update(
        state, np.full((3, 1), 102.0), np.ones(3, bool), context
    )
    assert metric.result(state, context) == pytest.approx((2 + 12) / 5)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("de", [False, True])
def test_esjd_sampling_rng_and_de_totals(
    backend: str, de: bool, backend_factory: Any
) -> None:
    case = backend_factory(backend)
    outputs = []
    for metrics in ([], [ESJDMetric()]):

        def target(x: Any, context: StepContext) -> Any:
            return -0.5 * case.xp.sum(x**2, axis=1)

        mutation = (
            DifferentialEvolutionMutation(target, n_steps=4, metrics=metrics)
            if de
            else MetropolisHastingsMutation(
                target,
                GaussianRandomWalk(scale=0.5),
                n_steps=4,
                metrics=metrics,
            )
        )
        rng = case.rng(23)
        final, _, output = mutation(case.state(), StepContext(t=0), rng)
        outputs.append((final.particles, rng.random((5,))))
        if metrics:
            result = output["metrics"]
            assert float(to_numpy(result.values["esjd"])) >= 0
            if de:
                totals = result.values["de_scales"]
                expected = (
                    to_numpy(totals["squared_jump_distance"]).sum()
                    / to_numpy(totals["n_attempted"]).sum()
                )
                assert float(to_numpy(result.values["esjd"])) == pytest.approx(
                    expected, rel=1e-6
                )
    for a, b in zip(*outputs, strict=True):
        np.testing.assert_array_equal(to_numpy(a), to_numpy(b))


def test_esjd_jax_compiled_loop() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    metrics = MutationMetrics([ESJDMetric()])
    context = StepContext(t=0)
    particles = jnp.zeros((3, 2))
    initial = metrics.initialize(particles, jnp.zeros(3), context)

    @jax.jit
    def run(state: Any) -> Any:
        state = jax.lax.fori_loop(
            0,
            4,
            lambda i, s: metrics.update(
                s, particles + i + 1, jnp.ones(3, dtype=bool), context
            ),
            state,
        )
        return metrics.result(state, context).values["esjd"]

    assert float(run(initial)) == 2
