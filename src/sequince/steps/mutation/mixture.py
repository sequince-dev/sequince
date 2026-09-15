"""Randomly interleave population-preserving mutation transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from orng import RandomGenerator

from ..._typing import Array
from ...array_api import namespace, to_numpy
from ...state import ParticleState
from ...target import EvaluationCounts
from ..base import SMCStep, validate_n_steps
from ..context import StepContext
from .metrics import MutationMetricsOutput, MutationStepState
from .metropolis import (
    DifferentialEvolutionMutation,
    MetropolisHastingsMutation,
    RedBlueMetropolisHastingsMutation,
)
from .minipcn import MiniPCNMutation


def normalize_move_weights(
    weights: Sequence[float], names: Sequence[str]
) -> tuple[float, ...]:
    """Validate relative weights and return probabilities in move order."""
    if len(weights) != len(names):
        raise ValueError("Weights must match the number of moves.")
    values = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Weights must be finite and nonnegative.")
    largest = max(values, default=0.0)
    if largest == 0:
        raise ValueError("At least one weight must be positive.")
    scaled = tuple(value / largest for value in values)
    total = sum(scaled)
    return tuple(value / total for value in scaled)


@dataclass(frozen=True)
class MoveDiagnostics:
    """Completed metrics for one named move; selections count whole sweeps."""

    selections: int
    metrics: MutationMetricsOutput
    evaluation_counts: EvaluationCounts
    scaled_squared_displacement: float = 0.0


@dataclass(frozen=True)
class MixtureMutationOutput:
    """Aggregate bookkeeping and separate named child metrics."""

    metrics: MutationMetricsOutput
    moves: Mapping[str, MoveDiagnostics]
    weights: tuple[float, ...]
    evaluation_counts: EvaluationCounts
    coordinate_scale: Array


class MixtureMutation(SMCStep):
    """Select one move for the whole population at each transition.

    Parameters
    ----------
    moves : mapping of str to mutation
        Ordered, nonempty named children sharing the identical target object
        or log-density callable. Supported children are untransformed
        MetropolisHastingsMutation with a persistent kernel,
        RedBlueMetropolisHastingsMutation, DifferentialEvolutionMutation, and
        MiniPCNMutation with pcn/tpcn and adaptive=False.
        Kernels must preserve particle correspondence and the current target;
        state-dependent proposal adaptation inside a kernel is unsupported.
    weights : sequence of float, optional
        Finite nonnegative relative weights in mapping order. Defaults to equal
        probabilities. At least one weight must be positive.
    n_steps : int, optional
        Total transitions (complete sweeps for red-blue children), including
        zero. Probabilities and length stay fixed throughout each block.

    Notes
    -----
    Eager execution only. MiniPCN proposals are fitted once at block entry
    for every positive-weight child, then frozen for the complete block.
    Transforms, kernel factories, other external samplers, and mutation
    wrappers are rejected: their per-call preparation may refit,
    warm up, or rebuild a sampler between individual transitions. Custom
    metrics have independent per-child state, finalized once per block.
    """

    def __init__(
        self,
        moves: Mapping[str, MetropolisHastingsMutation | MiniPCNMutation],
        weights: Sequence[float] | None = None,
        *,
        n_steps: int = 1,
    ) -> None:
        if not isinstance(moves, Mapping) or not moves:
            raise ValueError("moves must be a nonempty ordered mapping.")
        if any(
            not isinstance(name, str) or not name.strip() for name in moves
        ):
            raise ValueError("Move names must be nonempty strings.")
        self.moves = dict(moves)
        self.names = tuple(moves)
        if len({id(move) for move in moves.values()}) != len(moves):
            raise ValueError("Each move must be a distinct mutation instance.")
        target: object | None = None
        for move in moves.values():
            if type(move) not in (
                MetropolisHastingsMutation,
                RedBlueMetropolisHastingsMutation,
                DifferentialEvolutionMutation,
                MiniPCNMutation,
            ):
                raise TypeError("Unsupported mixture child mutation.")
            if (
                move._transform is not None
                or getattr(move, "_kernel_factory", None) is not None
            ):
                raise ValueError(
                    "Mixture children require no transform and a persistent kernel."
                )
            if isinstance(move, MiniPCNMutation):
                move._validate_mixture_support()
            move_target = (
                move._target if move._target is not None else move._log_prob_fn
            )
            if target is not None and move_target is not target:
                raise ValueError(
                    "Mixture children must share the same target."
                )
            target = move_target
        self.weights = normalize_move_weights(
            weights if weights is not None else [1.0] * len(moves), self.names
        )
        self.n_steps = validate_n_steps(n_steps)

    def apply(
        self, state: ParticleState, context: StepContext, rng: RandomGenerator
    ) -> tuple[ParticleState, StepContext, MixtureMutationOutput]:
        return self.apply_with_n_steps(
            state, context, rng, n_steps=self.n_steps
        )

    def apply_with_n_steps(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
        *,
        n_steps: int,
        weights: Sequence[float] | None = None,
    ) -> tuple[ParticleState, StepContext, MixtureMutationOutput]:
        """Execute a block with optional per-call relative weight overrides."""
        n_steps = validate_n_steps(n_steps)
        probabilities = normalize_move_weights(
            self.weights if weights is None else weights, self.names
        )
        states: dict[str, MutationStepState | None] = dict.fromkeys(self.names)
        selections = dict.fromkeys(self.names, 0)
        costs = {name: EvaluationCounts() for name in self.names}
        jumps = dict.fromkeys(self.names, 0.0)
        xp = namespace(state.particles)
        if state.particles.shape[0] == 0:
            raise ValueError(
                "Mixture mutation requires a nonempty population."
            )
        # Use common original coordinates, frozen before any move executes.
        variance = xp.var(state.particles, axis=0)
        valid = xp.isfinite(variance) & (variance > 0)
        scale = xp.sqrt(xp.where(valid, variance, xp.ones_like(variance)))
        prepared: dict[str, MetropolisHastingsMutation] = {}
        if n_steps:
            for name, probability in zip(
                self.names, probabilities, strict=True
            ):
                move = self.moves[name]
                if probability == 0:
                    continue
                prepared[name] = (
                    move._prepare_mixture_mutation(state, context, rng)
                    if isinstance(move, MiniPCNMutation)
                    else move
                )
        for _ in range(n_steps):
            index = int(
                rng.choice(len(self.names), size=(1,), p=probabilities)[0]
            )
            name = self.names[index]
            before_counts = context.evaluation_counts
            before = xp.asarray(state.particles, copy=True)
            state, context, _, states[name] = prepared[
                name
            ].apply_with_mutation_state(
                state, context, rng, n_steps=1, mutation_state=states[name]
            )
            displacement = (state.particles - before) / scale
            jumps[name] += float(to_numpy(xp.sum(displacement**2)))
            costs[name] = costs[name] + EvaluationCounts(
                **{
                    field.name: getattr(context.evaluation_counts, field.name)
                    - getattr(before_counts, field.name)
                    for field in fields(EvaluationCounts)
                }
            )
            selections[name] += 1
        diagnostics = {
            name: MoveDiagnostics(
                selections[name],
                self.moves[name].metrics.result(states[name], context)
                or MutationMetricsOutput(0, 0, 0),
                costs[name],
                jumps[name],
            )
            for name in self.names
        }
        metrics = MutationMetricsOutput(
            n_steps,
            sum(item.metrics.n_attempted for item in diagnostics.values()),
            sum(item.metrics.n_accepted for item in diagnostics.values()),
        )
        total_cost = EvaluationCounts()
        for cost in costs.values():
            total_cost = total_cost + cost
        return (
            state,
            context,
            MixtureMutationOutput(
                metrics, diagnostics, probabilities, total_cost, scale
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        """Save child state with names and order for compatibility checking."""
        return {
            "names": self.names,
            "children": {
                name: child.state_dict() for name, child in self.moves.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore children, rejecting a different ordered composition."""
        if tuple(state.get("names", ())) != self.names:
            raise ValueError("Checkpoint move names/order do not match.")
        children = state.get("children")
        if not isinstance(children, Mapping) or set(children) != set(
            self.names
        ):
            raise ValueError("Checkpoint must contain all named children.")
        if any(not isinstance(value, Mapping) for value in children.values()):
            raise TypeError("Child checkpoint states must be mappings.")
        for name, child in self.moves.items():
            child.load_state_dict(children[name])

    def apply_functional(
        self, state: ParticleState, context: StepContext, random_state: Any
    ) -> tuple[ParticleState, StepContext, Any, MixtureMutationOutput]:
        raise NotImplementedError(
            "MixtureMutation supports eager execution only."
        )


__all__ = ["MixtureMutation", "MixtureMutationOutput", "MoveDiagnostics"]
