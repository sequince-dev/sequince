from dataclasses import replace

import numpy as np
import pytest

from sequince.steps.context import StepContext
from sequince.steps.mutation.adaptive.policies.jump import (
    JumpDistancePerEvaluation,
)
from sequince.steps.mutation.metrics import MutationMetricsOutput
from sequince.steps.mutation.mixture import (
    MixtureMutationOutput,
    MoveDiagnostics,
)
from sequince.target import EvaluationCounts


def output(jumps=(2.0, 8.0), costs=(2.0, 4.0), selections=(1, 1)):
    metrics = MutationMetricsOutput(2, 2, 2)
    return MixtureMutationOutput(
        metrics,
        {
            name: MoveDiagnostics(
                n, metrics, EvaluationCounts(log_target_evaluations=cost), jump
            )
            for name, n, cost, jump in zip(
                ("a", "b"), selections, costs, jumps
            )
        },
        (0.5, 0.5),
        EvaluationCounts(),
        np.ones(1),
    )


def update(policy, value, weights=(0.5, 0.5)):
    return policy.update(
        current_weights=weights, output=value, context=StepContext(0)
    )


def test_scores_smoothing_floor_and_checkpoint():
    policy = JumpDistancePerEvaluation(
        score_smoothing=0.5, probability_smoothing=1, min_probability=0.1
    )
    np.testing.assert_allclose(
        update(policy, output()), [0.1 + 0.8 / 3, 0.1 + 1.6 / 3]
    )
    np.testing.assert_allclose(
        update(policy, output(jumps=(6, 0))), [0.1 + 1.6 / 3, 0.1 + 0.8 / 3]
    )
    other = JumpDistancePerEvaluation(
        score_smoothing=0.5, probability_smoothing=1, min_probability=0.1
    )
    other.load_state_dict(policy.state_dict())
    assert update(policy, output()) == update(other, output())


@pytest.mark.parametrize(
    "value",
    [
        output(costs=(0, 4)),
        output(jumps=(np.nan, 8)),
        output(selections=(0, 1)),
        output(jumps=(0, 0)),
        output(costs=(np.inf, 4)),
    ],
)
def test_missing_or_zero_information_retains_weights(value):
    assert update(JumpDistancePerEvaluation(), value) == (0.5, 0.5)


def test_invalid_observation_retains_old_score():
    policy = JumpDistancePerEvaluation()
    update(policy, output())
    before = policy.state_dict()
    update(policy, output(costs=(0, 0)))
    assert policy.state_dict() == before


def test_configurable_evaluation_costs():
    policy = JumpDistancePerEvaluation(
        evaluation_costs={"log_likelihood_evaluations": 3},
        min_probability=0,
        probability_smoothing=1,
    )
    value = output()
    value = replace(
        value,
        moves={
            name: replace(
                move,
                evaluation_counts=EvaluationCounts(
                    log_likelihood_evaluations=1
                ),
            )
            for name, move in value.moves.items()
        },
    )
    np.testing.assert_allclose(update(policy, value), [0.2, 0.8])


def test_infeasible_floor():
    with pytest.raises(ValueError, match="infeasible"):
        update(JumpDistancePerEvaluation(min_probability=0.6), output())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_smoothing": 0},
        {"probability_smoothing": np.nan},
        {"min_probability": -1},
        {"evaluation_costs": {}},
        {"evaluation_costs": {"gradient": 1}},
        {"min_selections": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        JumpDistancePerEvaluation(**kwargs)
