"""An opt-in displacement per evaluation heuristic for mixture probabilities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from .....array_api import to_numpy
from .....target import EvaluationCounts
from ....context import StepContext
from ...mixture import MixtureMutationOutput, normalize_move_weights


class JumpDistancePerEvaluation:
    """Smooth observed scaled squared jumps per weighted evaluation cost.

    Parameters
    ----------
    score_smoothing : float, optional
        Exponential update fraction in (0, 1], default 0.3.
    probability_smoothing : float, optional
        Fraction in (0, 1] moving old probabilities toward normalized scores.
    min_probability : float, optional
        Exploration floor, default 0.05. Must satisfy ``n_moves * floor <= 1``.
    evaluation_costs : mapping of str to float, optional
        Nonnegative costs keyed by EvaluationCounts field names. Unspecified
        fields have zero cost. Default: log_target_evaluations costs one, all
        others zero. Target and likelihood counters can overlap; the caller
        must choose coefficients that represent their actual work costs.
    min_selections : int, optional
        Minimum selections in a block needed to update a score, default one.

    Notes
    -----
    Unselected, insufficiently measured, zero-cost, or nonfinite observations
    retain the old score. Until every move has a valid estimate, probabilities
    are retained (subject to the floor). All-zero scores also retain weights.
    A valid zero jump updates the score toward zero. Score smoothing precedes
    normalization; probability smoothing precedes floor enforcement using
    ``floor + (1 - n_moves * floor) * probability``. This heuristic is not a
    universal measure of mixing efficiency and offers no improvement guarantee.
    No extra target evaluations or full trajectories are required.
    """

    def __init__(
        self,
        *,
        score_smoothing: float = 0.3,
        probability_smoothing: float = 0.3,
        min_probability: float = 0.05,
        evaluation_costs: Mapping[str, float] | None = None,
        min_selections: int = 1,
    ) -> None:
        for name, value in (
            ("score_smoothing", score_smoothing),
            ("probability_smoothing", probability_smoothing),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        if not math.isfinite(min_probability) or not 0 <= min_probability <= 1:
            raise ValueError("min_probability must be in [0, 1].")
        if isinstance(min_selections, bool) or not isinstance(
            min_selections, int
        ):
            raise TypeError("min_selections must be an integer.")
        if min_selections < 1:
            raise ValueError("min_selections must be positive.")
        costs = dict(
            {"log_target_evaluations": 1.0}
            if evaluation_costs is None
            else evaluation_costs
        )
        valid_names = {field.name for field in fields(EvaluationCounts)}
        if set(costs) - valid_names:
            raise ValueError("Unknown evaluation counter in evaluation_costs.")
        if (
            not costs
            or any(
                not math.isfinite(value) or value < 0
                for value in costs.values()
            )
            or not any(costs.values())
        ):
            raise ValueError(
                "Evaluation costs must be nonnegative with positive sum."
            )
        self.score_smoothing = float(score_smoothing)
        self.probability_smoothing = float(probability_smoothing)
        self.min_probability = float(min_probability)
        self.evaluation_costs = costs
        self.min_selections = min_selections
        self._scores: dict[str, float] = {}

    def _floor(self, weights: tuple[float, ...]) -> tuple[float, ...]:
        count = len(weights)
        if count == 0 or count * self.min_probability > 1:
            raise ValueError("min_probability is infeasible for this mixture.")
        remainder = 1 - count * self.min_probability
        return tuple(
            self.min_probability + remainder * value for value in weights
        )

    def initialise(
        self, *, initial_weights: tuple[float, ...], context: StepContext
    ) -> tuple[float, ...]:
        return self._floor(initial_weights)

    def update(
        self,
        *,
        current_weights: tuple[float, ...],
        output: MixtureMutationOutput,
        context: StepContext,
    ) -> tuple[float, ...]:
        names = tuple(output.moves)
        weights = normalize_move_weights(current_weights, names)
        # Validate feasibility even when there is no measurement this block.
        self._floor(weights)
        for name, move in output.moves.items():
            if move.selections < self.min_selections:
                continue
            cost = sum(
                coefficient
                * float(to_numpy(getattr(move.evaluation_counts, key)))
                for key, coefficient in self.evaluation_costs.items()
                if coefficient > 0
            )
            jump = move.scaled_squared_displacement
            if (
                not math.isfinite(cost)
                or cost <= 0
                or not math.isfinite(jump)
                or jump < 0
            ):
                continue
            estimate = jump / cost
            if not math.isfinite(estimate):
                continue
            previous = self._scores.get(name, estimate)
            alpha = self.score_smoothing
            self._scores[name] = (1 - alpha) * previous + alpha * estimate
        if not all(name in self._scores for name in names):
            return weights
        scores = tuple(self._scores[name] for name in names)
        if not any(scores):
            return weights
        proposed = normalize_move_weights(scores, names)
        alpha = self.probability_smoothing
        return self._floor(
            tuple(
                (1 - alpha) * old + alpha * new
                for old, new in zip(weights, proposed, strict=True)
            )
        )

    def state_dict(self) -> dict[str, Any]:
        """Return smoothed scores by move name."""
        return {"scores": dict(self._scores)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore finite nonnegative smoothed scores."""
        scores = state.get("scores")
        if not isinstance(scores, Mapping):
            raise TypeError("Policy scores must be a mapping.")
        parsed = {}
        for name, value in scores.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Score keys must be nonempty move names.")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Policy scores must be finite and nonnegative."
                )
            parsed[name] = value
        self._scores = parsed


__all__ = ["JumpDistancePerEvaluation"]
