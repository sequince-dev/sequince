from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from sequince import ParticleState
from sequince.steps.base import SMCStep
from sequince.steps.control import RepeatStep
from sequince.steps.mutation import (
    AdaptMutationSteps,
    CallableMutationKernel,
    ESJDPolicy,
    FixedMutationStepPolicy,
    MetropolisHastingsMutation,
    MutationDiagnostics,
    RepeatedMutationStep,
)


class RecordingMutation(SMCStep):
    def __init__(self, events: list[tuple[str, int]] | None = None):
        self.lengths: list[int] = []
        self.events = events

    def apply(self, state, context, rng):
        return self.apply_with_n_steps(state, context, rng, n_steps=1)

    def apply_with_n_steps(self, state, context, rng, *, n_steps):
        del rng
        self.lengths.append(n_steps)
        if self.events is not None:
            self.events.append(("mutation", n_steps))
        diagnostics = MutationDiagnostics(
            n_steps=n_steps,
            n_attempted=n_steps * state.particles.shape[0],
            n_accepted=np.asarray(0),
            metrics={"esjd": np.asarray(0.0)},
        )
        return state, context, {"steps": n_steps, "diagnostics": diagnostics}


class SequencePolicy:
    def __init__(
        self,
        decisions: list[int],
        events: list[tuple[str, int]] | None = None,
    ):
        self.decisions = decisions
        self.index = 0
        self.initialise_calls = 0
        self.updates: list[tuple[int, int, int]] = []
        self.events = events

    def initialise(self, *, initial_steps, context):
        self.initialise_calls += 1
        return initial_steps

    def update(self, *, current_steps, diagnostics, context):
        self.updates.append(
            (current_steps, diagnostics.n_steps, context.iteration)
        )
        if self.events is not None:
            self.events.append(("policy", current_steps))
        decision = self.decisions[self.index]
        self.index += 1
        return decision

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state: Mapping[str, object]):
        self.index = int(state["index"])


class DiagnosticSequenceMutation(RecordingMutation):
    def __init__(self, esjd: list[float]):
        super().__init__()
        self.esjd = esjd
        self.proposal_scale = 0.125

    def apply_with_n_steps(self, state, context, rng, *, n_steps):
        del rng
        self.lengths.append(n_steps)
        value = self.esjd[len(self.lengths) - 1]
        diagnostics = MutationDiagnostics(
            n_steps=n_steps,
            n_attempted=n_steps * state.particles.shape[0],
            n_accepted=np.asarray(0),
            metrics={"esjd": np.asarray(value)},
        )
        return state, context, {"steps": n_steps, "diagnostics": diagnostics}


def _diagnostics(esjd) -> MutationDiagnostics:
    return MutationDiagnostics(
        n_steps=8,
        n_attempted=32,
        n_accepted=np.asarray(0),
        metrics={"esjd": esjd},
    )


def _esjd_update(policy: ESJDPolicy, esjd, *, current=8) -> int:
    return policy.update(
        current_steps=current,
        diagnostics=_diagnostics(esjd),
        context=None,
    )


def test_esjd_at_target_keeps_step_count() -> None:
    assert _esjd_update(ESJDPolicy(target_esjd=1.0), 1.0) == 8


def test_esjd_below_target_increases_step_count() -> None:
    assert (
        _esjd_update(
            ESJDPolicy(target_esjd=1.0, damping=1.0, max_change_factor=100.0),
            0.25,
        )
        == 32
    )


def test_esjd_above_target_decreases_step_count() -> None:
    assert (
        _esjd_update(
            ESJDPolicy(target_esjd=1.0, damping=1.0, max_change_factor=100.0),
            4.0,
        )
        == 2
    )


def test_esjd_update_follows_damped_power_law() -> None:
    policy = ESJDPolicy(target_esjd=1.0, damping=0.5, max_change_factor=100.0)

    assert _esjd_update(policy, 0.25) == 16
    assert _esjd_update(policy, 4.0) == 4


def test_smaller_damping_is_less_aggressive() -> None:
    undamped = ESJDPolicy(
        target_esjd=1.0, damping=1.0, max_change_factor=100.0
    )
    damped = ESJDPolicy(target_esjd=1.0, damping=0.5, max_change_factor=100.0)

    assert _esjd_update(undamped, 1.0 / 16.0) == 128
    assert _esjd_update(damped, 1.0 / 16.0) == 32


def test_zero_esjd_uses_maximum_increase() -> None:
    policy = ESJDPolicy(target_esjd=1.0, max_change_factor=2.0)

    assert _esjd_update(policy, 0.0) == 16
    assert _esjd_update(policy, 0.0, current=0) == 1


@pytest.mark.parametrize("esjd", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_esjd_raises_clear_error(esjd) -> None:
    with pytest.raises(ValueError, match="esjd"):
        _esjd_update(ESJDPolicy(target_esjd=1.0), esjd)


def test_missing_or_nonscalar_esjd_raises_clear_error() -> None:
    policy = ESJDPolicy(target_esjd=1.0)
    with pytest.raises(TypeError, match="completed MutationDiagnostics"):
        policy.update(current_steps=8, diagnostics=None, context=None)
    with pytest.raises(ValueError, match="missing"):
        _esjd_update(policy, None)
    with pytest.raises(TypeError, match="scalar"):
        _esjd_update(policy, np.asarray([1.0, 2.0]))


def test_esjd_maximum_change_factor_is_respected() -> None:
    policy = ESJDPolicy(target_esjd=1.0, damping=1.0, max_change_factor=2.0)

    assert _esjd_update(policy, 0.0001, current=10) == 20
    assert _esjd_update(policy, 10000.0, current=10) == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_esjd": 0.0}, "target_esjd"),
        ({"target_esjd": np.inf}, "target_esjd"),
        ({"target_esjd": 1.0, "damping": 0.0}, "damping"),
        ({"target_esjd": 1.0, "damping": 1.1}, "damping"),
        (
            {"target_esjd": 1.0, "max_change_factor": 0.5},
            "max_change_factor",
        ),
        (
            {"target_esjd": 1.0, "max_change_factor": np.nan},
            "max_change_factor",
        ),
    ],
)
def test_invalid_esjd_policy_configuration_raises(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ESJDPolicy(**kwargs)


def _adaptive(mutation, policy=None, *, initial=4, minimum=0, maximum=20):
    return AdaptMutationSteps(
        mutation,
        policy,
        initial_steps=initial,
        min_steps=minimum,
        max_steps=maximum,
    )


def test_esjd_policy_uses_each_completed_block_for_next_iteration(
    particle_state, context_factory, rng
) -> None:
    mutation = DiagnosticSequenceMutation([0.25, 2.0, 1.0])
    adaptive = _adaptive(
        mutation,
        ESJDPolicy(
            target_esjd=1.0,
            damping=0.5,
            max_change_factor=2.0,
        ),
        initial=8,
        maximum=32,
    )

    adaptive(particle_state, context_factory(t=0), rng)
    assert mutation.lengths == [8]
    adaptive(particle_state, context_factory(t=1), rng)
    assert mutation.lengths == [8, 16]
    adaptive(particle_state, context_factory(t=2), rng)

    assert mutation.lengths == [8, 16, 12]
    assert mutation.proposal_scale == 0.125


def test_wrapper_applies_absolute_bounds_to_esjd_policy(
    particle_state, context_factory, rng
) -> None:
    mutation = DiagnosticSequenceMutation([0.01, 100.0, 1.0])
    adaptive = _adaptive(
        mutation,
        ESJDPolicy(
            target_esjd=1.0,
            damping=1.0,
            max_change_factor=100.0,
        ),
        initial=8,
        minimum=3,
        maximum=20,
    )

    adaptive(particle_state, context_factory(t=0), rng)
    adaptive(particle_state, context_factory(t=1), rng)
    adaptive(particle_state, context_factory(t=2), rng)

    assert mutation.lengths == [8, 20, 3]


def test_budget_is_frozen_per_outer_iteration_and_updates_once(
    particle_state, context_factory, rng
) -> None:
    mutation = RecordingMutation()
    policy = SequencePolicy([8, 2, 6])
    adaptive = _adaptive(mutation, policy)

    adaptive(particle_state, context_factory(t=0), rng)
    adaptive(particle_state, context_factory(t=0), rng)
    adaptive(particle_state, context_factory(t=1), rng)
    adaptive(particle_state, context_factory(t=2), rng)

    assert mutation.lengths == [4, 4, 8, 2]
    assert policy.initialise_calls == 1
    assert policy.updates == [(4, 4, 0), (8, 8, 1), (2, 2, 2)]
    assert adaptive.current_n_steps == 2
    assert adaptive.next_n_steps == 6


def test_policy_is_only_consulted_after_complete_mutation_block(
    particle_state, context_factory, rng
) -> None:
    events: list[tuple[str, int]] = []
    adaptive = _adaptive(
        RecordingMutation(events), SequencePolicy([7], events)
    )

    adaptive(particle_state, context_factory(t=3), rng)

    assert events == [("mutation", 4), ("policy", 4)]
    assert adaptive.next_n_steps == 7


def test_policy_decisions_are_clamped_to_configured_bounds(
    particle_state, context_factory, rng
) -> None:
    mutation = RecordingMutation()
    adaptive = _adaptive(
        mutation,
        SequencePolicy([-100, 100, 9]),
        initial=4,
        minimum=2,
        maximum=9,
    )

    adaptive(particle_state, context_factory(t=0), rng)
    adaptive(particle_state, context_factory(t=1), rng)
    adaptive(particle_state, context_factory(t=2), rng)

    assert mutation.lengths == [4, 2, 9]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"initial_steps": 1, "min_steps": -1, "max_steps": 2},
            ValueError,
            ">= 0",
        ),
        (
            {"initial_steps": 1, "min_steps": 2, "max_steps": 1},
            ValueError,
            ">= min",
        ),
        (
            {"initial_steps": 3, "min_steps": 0, "max_steps": 2},
            ValueError,
            "within",
        ),
        (
            {"initial_steps": True, "min_steps": 0, "max_steps": 2},
            TypeError,
            "integer",
        ),
    ],
)
def test_invalid_bounds_raise_clear_errors(kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        AdaptMutationSteps(RecordingMutation(), **kwargs)


def test_non_integer_policy_decision_is_rejected_after_block(
    particle_state, context_factory, rng
) -> None:
    adaptive = _adaptive(RecordingMutation(), SequencePolicy([1.5]))

    with pytest.raises(TypeError, match="decisions must be integers"):
        adaptive(particle_state, context_factory(), rng)


def test_fixed_policy_keeps_initial_budget(
    particle_state, context_factory, rng
) -> None:
    mutation = RecordingMutation()
    adaptive = _adaptive(mutation, FixedMutationStepPolicy(), initial=3)

    adaptive(particle_state, context_factory(t=0), rng)
    adaptive(particle_state, context_factory(t=1), rng)

    assert mutation.lengths == [3, 3]


def _flat_log_prob(particles, context):
    del context
    return np.zeros(particles.shape[0])


def _increment_kernel():
    return CallableMutationKernel(
        lambda particles, rng: particles + 1.0,
        symmetric=True,
    )


class NegateStep(SMCStep):
    def apply(self, state, context, rng):
        del rng
        return state.replace_particles(-state.particles), context, None


def test_repeat_step_receives_total_budget_and_keeps_round_schedule(
    context_factory, rng
) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    repeated = RepeatedMutationStep(
        MetropolisHastingsMutation(_flat_log_prob, _increment_kernel()),
        n_rounds=3,
    )
    adaptive = _adaptive(repeated, initial=5)

    updated, _, outputs = adaptive(state, context_factory(), rng)

    assert [output["steps"] for output in outputs.round_outputs] == [2, 2, 1]
    assert outputs.diagnostics.n_steps == 5
    np.testing.assert_array_equal(updated.particles, state.particles + 5.0)


def test_repeated_symmetry_schedule_is_unchanged(context_factory, rng) -> None:
    state = ParticleState(np.array([[-1.0], [1.0]]))
    mutation = MetropolisHastingsMutation(_flat_log_prob, _increment_kernel())
    repeated = RepeatedMutationStep(
        mutation, n_rounds=2, after_round=NegateStep()
    )
    adaptive = _adaptive(repeated, initial=3)

    updated, _, outputs = adaptive(state, context_factory(), rng)

    assert [
        round_output["steps"] for round_output in outputs.round_outputs
    ] == [2, 1]
    assert len(outputs.round_outputs) == 2
    assert outputs.diagnostics.n_steps == 3
    np.testing.assert_array_equal(updated.particles, state.particles + 1.0)


def test_checkpoint_preserves_next_decision_and_policy_state(
    particle_state, context_factory, rng
) -> None:
    original_mutation = RecordingMutation()
    original_policy = SequencePolicy([8, 2])
    original = _adaptive(original_mutation, original_policy)
    original(particle_state, context_factory(t=0), rng)
    checkpoint = original.state_dict()

    original(particle_state, context_factory(t=1), rng)

    restored_mutation = RecordingMutation()
    restored_policy = SequencePolicy([8, 2])
    restored = _adaptive(restored_mutation, restored_policy)
    restored.load_state_dict(checkpoint)
    # Replaying the checkpointed iteration uses its frozen decision but does not
    # update the policy a second time.
    restored(particle_state, context_factory(t=0), rng)
    restored(particle_state, context_factory(t=1), rng)

    assert original_mutation.lengths[-1] == 8
    assert restored_mutation.lengths == [4, 8]
    assert restored_policy.index == original_policy.index == 2
    assert restored.next_n_steps == original.next_n_steps == 2


def test_repeat_wrapper_preserves_nested_adaptive_state(
    particle_state, context_factory, rng
) -> None:
    wrapped = RepeatStep(
        _adaptive(RecordingMutation(), SequencePolicy([8]), initial=4), n=1
    )
    wrapped(particle_state, context_factory(t=0), rng)

    restored = RepeatStep(
        _adaptive(RecordingMutation(), SequencePolicy([8]), initial=4), n=1
    )
    restored.load_state_dict(wrapped.state_dict())

    assert restored.step.next_n_steps == 8


def test_unwrapped_mutation_keeps_its_configured_length(
    particle_state, context_factory, rng
) -> None:
    mutation = RecordingMutation()

    mutation(particle_state, context_factory(t=0), rng)
    mutation(particle_state, context_factory(t=1), rng)

    assert mutation.lengths == [1, 1]


def test_shared_adaptation_lifecycle(context_factory, rng):
    from sequince.steps import AdaptiveMutationStep

    events = []
    output = {"diagnostics": MutationDiagnostics(1, 2, 1)}

    class Child(SMCStep):
        def apply(self, state, context, rng):
            events.append("mutation")
            assert context.metadata["prepared"]
            return state, context.with_metadata(child=True), output

    class Adapter(AdaptiveMutationStep):
        def _prepare_mutation(self, context, n_steps):
            events.append("prepare")
            return context.with_metadata(prepared=True), n_steps

        def _adapt(self, diagnostics, context, *, initial_context, n_steps):
            events.append("adapt")
            assert diagnostics is output["diagnostics"]
            assert context.metadata["child"]
            assert "child" not in initial_context.metadata
            return context.with_metadata(adapted=True)

    state = ParticleState(np.array([[-1.0], [1.0]]))
    updated, context, result = Adapter(Child())(state, context_factory(), rng)
    assert updated is state
    assert result is output
    assert context.metadata["adapted"]
    assert events == ["prepare", "mutation", "adapt"]


def test_budget_and_scale_adaptation_compose_and_restore(context_factory, rng):
    from sequince.steps import AdaptGaussianProposal, AdaptiveMutationStep

    child = RecordingMutation()
    scale = AdaptGaussianProposal(child, initial_scale=0.5)
    policy = SequencePolicy([4, 6])
    budget = AdaptMutationSteps(
        scale,
        policy,
        initial_steps=2,
        min_steps=1,
        max_steps=10,
    )
    assert isinstance(scale, AdaptiveMutationStep)
    assert isinstance(budget, AdaptiveMutationStep)
    state = ParticleState(np.array([[-1.0], [1.0]]))
    _, context, output = budget(state, context_factory(t=0), rng)
    assert output["diagnostics"].n_steps == 2
    assert context.metadata["proposal_scale"] == pytest.approx(
        0.5 * np.exp(-0.03)
    )
    checkpoint = budget.state_dict()

    restored_child = RecordingMutation()
    restored = AdaptMutationSteps(
        AdaptGaussianProposal(restored_child, initial_scale=0.5),
        SequencePolicy([4, 6]),
        initial_steps=2,
        min_steps=1,
        max_steps=10,
    )
    restored.load_state_dict(checkpoint)
    next_context = context_factory(t=1, metadata=context.metadata)
    _, next_context, output = restored(state, next_context, rng)
    assert restored_child.lengths == [4]
    assert output["diagnostics"].n_steps == 4
    assert next_context.metadata["proposal_scale"] == pytest.approx(
        0.5 * np.exp(-0.06)
    )
    assert restored.next_n_steps == 6
