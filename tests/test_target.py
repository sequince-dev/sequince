from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pytest

from sequince.posterior import TemperedPosterior
from sequince.target import (
    EvaluationCounts,
    SequentialTarget,
    Target,
    TargetEvaluation,
    TemperedTarget,
    _evaluate_component,
)


class DiscreteTarget(Target):
    """Small stage-dependent target without sequential reweighting."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluation_key(self, context) -> int:
        return int(context.metadata["stage"])

    def evaluate(
        self,
        particles,
        context,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
    ) -> TargetEvaluation:
        del components, evaluation
        self.calls += 1
        return self.bind_evaluation(
            TargetEvaluation(
                components={
                    "log_prob": np.full(
                        particles.shape[0], -self.evaluation_key(context)
                    )
                }
            ),
            context,
        )

    def log_prob(self, particles, context, *, evaluation=None):
        evaluation = evaluation or self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        return evaluation["log_prob"]


class DiscreteSequentialTarget(SequentialTarget):
    """Small beta-independent target used to exercise sequence stages."""

    def __init__(self) -> None:
        self.calls = {"log_prob": 0, "log_increment": 0}

    def evaluation_key(self, context) -> int:
        return int(context.metadata["stage"])

    def evaluate(
        self,
        particles,
        context,
        *,
        components: Collection[str] | None = None,
        evaluation: TargetEvaluation | None = None,
    ) -> TargetEvaluation:
        if evaluation is not None:
            self.validate_evaluation(evaluation, particles, context)
        allowed = {"log_prob", "log_increment"}
        requested = allowed if components is None else set(components)
        values = {} if evaluation is None else dict(evaluation.components)
        stage = int(context.metadata["stage"])
        if "log_prob" in requested:
            self.calls["log_prob"] += 1
            values["log_prob"] = np.full(particles.shape[0], -stage)
        if "log_increment" in requested:
            self.calls["log_increment"] += 1
            values["log_increment"] = np.full(particles.shape[0], stage + 1)
        return self.bind_evaluation(
            TargetEvaluation(components=values), context
        )

    def log_prob(
        self,
        particles,
        context,
        *,
        evaluation=None,
    ):
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        return evaluation["log_prob"]

    def log_increment(
        self,
        particles,
        context,
        *,
        evaluation=None,
    ):
        if evaluation is None:
            evaluation = self.evaluate(particles, context)
        self.validate_evaluation(evaluation, particles, context)
        return evaluation["log_increment"]


def test_evaluate_component_normalizes_particle_values(
    context_factory,
) -> None:
    particles = np.zeros((3, 2))
    context = context_factory(metadata={"value": 2.5})

    scalar = _evaluate_component(
        lambda values, ctx: ctx.metadata["value"],
        particles,
        context,
    )
    column = _evaluate_component(
        lambda values: np.arange(3.0)[:, np.newaxis],
        particles,
        context,
    )

    np.testing.assert_allclose(scalar, [2.5, 2.5, 2.5])
    np.testing.assert_allclose(column, [0.0, 1.0, 2.0])


def test_annealed_target_is_a_sequential_target() -> None:
    assert issubclass(TemperedTarget, SequentialTarget)


def test_sequential_target_is_a_target() -> None:
    assert issubclass(SequentialTarget, Target)


def test_target_instances_have_unique_evaluation_ids() -> None:
    assert DiscreteTarget().evaluation_id != DiscreteTarget().evaluation_id


def test_target_owns_and_reuses_cached_evaluations(context_factory) -> None:
    particles = np.zeros((3, 1))
    context = context_factory(metadata={"stage": 2})
    target = DiscreteTarget()
    other_target = DiscreteTarget()

    evaluation = target.prepare_evaluation(particles, context)
    reused = target.prepare_evaluation(
        particles, context, evaluation=evaluation
    )
    replaced = other_target.prepare_evaluation(
        particles, context, evaluation=evaluation
    )

    assert reused is evaluation
    assert target.calls == 1
    assert target.accepts_evaluation(evaluation, context)
    assert not other_target.accepts_evaluation(evaluation, context)
    assert other_target.calls == 1
    assert replaced.target_id == other_target.evaluation_id


def test_custom_target_counts_log_target_evaluations_by_default(
    context_factory,
) -> None:
    particles = np.zeros((3, 1))
    context = context_factory(metadata={"stage": 0})
    target = DiscreteTarget()

    evaluation, counts = target.prepare_evaluation(
        particles,
        context,
        return_counts=True,
    )
    cached, cached_counts = target.prepare_evaluation(
        particles,
        context,
        evaluation=evaluation,
        return_counts=True,
    )
    _, refreshed_counts = target.prepare_evaluation(
        particles,
        context,
        evaluation=evaluation,
        refresh_components={"log_prob"},
        return_counts=True,
    )

    assert cached is evaluation
    assert counts.log_target_evaluations == 3
    assert cached_counts == EvaluationCounts.zeros(xp=np)
    assert refreshed_counts.log_target_evaluations == 3


def test_sequential_target_does_not_assume_beta(context_factory) -> None:
    target = DiscreteSequentialTarget()
    particles = np.zeros((3, 1))
    context = context_factory(metadata={"stage": 2, "beta": 0.75})

    evaluation = target.prepare_evaluation(particles, context)

    np.testing.assert_allclose(
        target.log_prob(particles, context, evaluation=evaluation), -2.0
    )
    np.testing.assert_allclose(
        target.log_increment(particles, context, evaluation=evaluation), 3.0
    )
    assert evaluation.evaluation_key == 2


def test_sequential_target_reuses_evaluation_within_stage(
    context_factory,
) -> None:
    target = DiscreteSequentialTarget()
    particles = np.zeros((4, 1))
    context = context_factory(metadata={"stage": 0})

    evaluation = target.prepare_evaluation(particles, context)
    reused = target.prepare_evaluation(
        particles,
        context,
        evaluation=evaluation,
    )

    assert reused is evaluation
    assert target.calls == {"log_prob": 1, "log_increment": 1}


def test_sequential_target_replaces_evaluation_when_stage_changes(
    context_factory,
    caplog,
) -> None:
    target = DiscreteSequentialTarget()
    particles = np.zeros((3, 1))
    context = context_factory(metadata={"stage": 0})
    first = target.prepare_evaluation(particles, context)

    context = context.with_metadata(stage=1)
    with caplog.at_level("DEBUG", logger="sequince.target"):
        second = target.prepare_evaluation(
            particles,
            context,
            evaluation=first,
        )

    assert target.calls == {"log_prob": 2, "log_increment": 2}
    assert first.evaluation_key == 0
    assert second.evaluation_key == 1
    assert "at key 0" in caplog.text
    assert "at key 1" in caplog.text


def test_tempered_posterior_exposes_annealed_endpoints(
    context_factory,
) -> None:
    target = TemperedPosterior(
        log_prior=lambda particles: np.full(particles.shape[0], -1.0),
        log_likelihood=lambda particles: np.full(particles.shape[0], -2.0),
    )
    particles = np.zeros((2, 1))
    context = context_factory(metadata={"beta": 0.25})
    evaluation = target.evaluate(particles, context)

    np.testing.assert_allclose(
        target.log_initial(particles, context, evaluation=evaluation), -1.0
    )
    np.testing.assert_allclose(
        target.log_terminal(particles, context, evaluation=evaluation), -3.0
    )
    np.testing.assert_allclose(
        target.log_increment(particles, context, evaluation=evaluation), -2.0
    )
    np.testing.assert_allclose(
        target.log_prob(particles, context, evaluation=evaluation), -1.5
    )


@pytest.mark.parametrize("beta", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("array_beta", [False, True])
@pytest.mark.parametrize("backend_name", ["numpy", "jax", "torch", "cupy"])
def test_tempered_density_preserves_zero_support(
    beta, array_beta, backend_name, backend_factory, context_factory
):
    from sequince.array_api import to_numpy

    xp = backend_factory(backend_name).xp
    particles = xp.asarray(
        [
            [-float("inf"), -float("inf")],
            [-float("inf"), -2.0],
            [-3.0, -float("inf")],
            [-3.0, -2.0],
        ]
    )
    target = TemperedPosterior(
        log_proposal=lambda x: x[:, 0],
        log_prior=lambda x: x[:, 1],
        log_likelihood=lambda x: xp.zeros_like(x[:, 1]),
    )
    context = context_factory(
        metadata={"beta": xp.asarray(beta) if array_beta else beta}
    )
    with np.errstate(invalid="raise"):
        result = target.log_prob(particles, context)
    if beta == 0:
        expected = [-np.inf, -np.inf, -3.0, -3.0]
    elif beta == 1:
        expected = [-np.inf, -2.0, -np.inf, -2.0]
    else:
        expected = [-np.inf, -np.inf, -np.inf, -2.75]
    np.testing.assert_allclose(to_numpy(result), expected)


def test_tempered_density_with_traced_beta(context_factory):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    particles = jnp.array([[-jnp.inf, -2.0], [-3.0, -jnp.inf]])
    target = TemperedPosterior(
        log_proposal=lambda x: x[:, 0],
        log_prior=lambda x: x[:, 1],
        log_likelihood=lambda x: jnp.zeros_like(x[:, 1]),
    )

    @jax.jit
    def log_prob(beta):
        return target.log_prob(
            particles, context_factory(metadata={"beta": beta})
        )

    np.testing.assert_array_equal(log_prob(0.0), [-np.inf, -3.0])
    np.testing.assert_array_equal(log_prob(0.5), [-np.inf, -np.inf])
    np.testing.assert_array_equal(log_prob(1.0), [-2.0, -np.inf])
