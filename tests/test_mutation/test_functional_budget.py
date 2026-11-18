"""Pure budget transitions, independent of mutation execution."""

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sequince.array_api import asarray, to_numpy
from sequince.checkpoint import SMCCheckpoint
from sequince.state import ParticleState
from sequince.steps import AdaptMutationSteps, MutationBudgetState, StepContext
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)
from sequince.steps.mutation.adaptive.policies.fixed import (
    FixedMutationStepPolicy,
)
from sequince.steps.mutation.metrics import MutationMetricsOutput
from sequince.typing import Array


class UnusedMutation:
    def apply_with_n_steps(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Budget operations must not execute mutation.")


def adapter(policy: Any = None, **kwargs: Any) -> AdaptMutationSteps:
    return AdaptMutationSteps(
        UnusedMutation(),
        policy,
        initial_steps=4,
        min_steps=0,
        max_steps=20,
        **kwargs,
    )


def initialize(step: AdaptMutationSteps, state: ParticleState) -> StepContext:
    return StepContext(
        t=0, metadata=step.initialize_functional_metadata(state, {})
    )


@pytest.mark.parametrize(
    "target,minimum,maximum,safeguard,attempted,accepted,current,expected",
    [
        (1.5, 0, 20, 1e-6, 10, 2.5, 4, 6),
        (2.0, 0, 20, 0.25, 10, 0, 4, 8),
        (2.0, 0, 20, 1e-6, 10, 0, 4, 20),
        (2.0, 0, 20, 1e-6, 0, 0, 0, 1),
        (2.0, 0, 20, 1e-6, 0, 0, 7, 7),
        (2.0, 5, 20, 1e-6, 10, 10, 7, 5),
        (1e300, 0, 20, 1e-6, 10, 10, 4, 20),
        (1e300, 0, 20, 1e-300, 10, 0, 4, 20),
        (1e-300, 0, 20, 1e-6, 10, 10, 4, 1),
        (1e-300, 0, 20, 1e-300, 10, 0, 4, 1),
        (2.0, 0, 0, 1e-6, 10, 0, 0, 0),
        (2.0, 0, 2**24, 1e-300, 10, 0, 4, 2**24),
    ],
)
def test_policy_agreement(
    backend,
    backend_factory,
    target,
    minimum,
    maximum,
    safeguard,
    attempted,
    accepted,
    current,
    expected,
) -> None:
    case = backend_factory(backend)
    policy = AcceptanceRatePolicy(
        target,
        min_steps=minimum,
        max_steps=maximum,
        min_acceptance=safeguard,
    )
    policy.validate_functional_configuration()
    budget = asarray(current, like=case.log_weights, dtype="int32")
    metrics = MutationMetricsOutput(
        budget,
        asarray(attempted, like=case.log_weights),
        asarray(accepted, like=case.log_weights),
    )
    context = StepContext(t=0)
    initial = policy.initialise_functional(
        initial_steps=budget, context=context
    )
    assert int(to_numpy(initial)) == policy.initialise(
        initial_steps=current, context=context
    )
    result = policy.update_functional(
        current_steps=budget, metrics=metrics, context=context
    )
    if backend == "jax":
        import jax

        compiled = jax.jit(
            lambda steps, output: policy.update_functional(
                current_steps=steps, metrics=output, context=context
            )
        )(budget, metrics)
        assert int(to_numpy(compiled)) == expected
    assert result.dtype == budget.dtype
    assert result.shape == ()
    assert case.matches_namespace(result)
    assert (
        int(to_numpy(result))
        == expected
        == policy.update(
            current_steps=current, metrics=metrics, context=context
        )
    )


@pytest.mark.parametrize("policy", [FixedMutationStepPolicy(), None])
def test_fixed_decisions_and_initialization(
    policy, backend, backend_factory
) -> None:
    case = backend_factory(backend)
    step = adapter(policy)
    context = initialize(step, case.state())
    prepared, budget = step.prepare_budget_functional(context)
    assert int(to_numpy(budget)) == 4
    updated = step.update_budget_functional(
        MutationMetricsOutput(budget, budget * 0, budget * 0), prepared
    )
    promoted, budget = step.prepare_budget_functional(replace(updated, t=1))
    assert int(to_numpy(budget)) == 4
    carry = promoted.metadata[step.metadata_key]
    for field in fields(carry):
        leaf = getattr(carry, field.name)
        assert leaf.shape == ()
        assert leaf.dtype == case.xp.int32


def test_functional_runs_leave_shared_adapter_unchanged(
    particle_state,
) -> None:
    step = adapter(AcceptanceRatePolicy(2, min_steps=0, max_steps=20))
    attributes = step.__dict__.copy()
    policy_attributes = step.policy.__dict__.copy()
    initial = initialize(step, particle_state)
    prepared, budget = step.prepare_budget_functional(initial)
    updated = step.update_budget_functional(
        MutationMetricsOutput(budget, np.array(10), np.array(2.5)), prepared
    )
    independent = initialize(step, particle_state)
    assert int(updated.metadata[step.metadata_key].next_n_steps) == 8
    assert int(independent.metadata[step.metadata_key].next_n_steps) == 4
    assert int(initial.metadata[step.metadata_key].current_iteration) == -1
    assert step.__dict__ == attributes
    assert step.policy.__dict__ == policy_attributes


@pytest.mark.parametrize("format", ["pickle", "hdf5"])
def test_resume_and_multiple_adapters(particle_state, tmp_path: Path, format):
    one = adapter(
        AcceptanceRatePolicy(2, min_steps=0, max_steps=20), metadata_key="one"
    )
    two = adapter(metadata_key="two")
    metadata = one.initialize_functional_metadata(
        particle_state, {"other": 12}
    )
    metadata = two.initialize_functional_metadata(particle_state, metadata)
    context, budget = one.prepare_budget_functional(
        StepContext(t=0, metadata=metadata)
    )
    context = one.update_budget_functional(
        MutationMetricsOutput(budget, np.array(10), np.array(2.5)), context
    )
    context, _ = two.prepare_budget_functional(context)
    checkpoint = SMCCheckpoint(
        states=(particle_state,),
        next_iteration=1,
        next_observation_index=0,
        metadata=context.metadata,
        outputs=(),
        step_states=(),
    )
    path = tmp_path / f"checkpoint.{format}"
    if format == "hdf5":
        h5py = pytest.importorskip("h5py")
        with h5py.File(path, "w") as handle:
            checkpoint.save_hdf5(handle)
        with h5py.File(path, "r") as handle:
            restored = SMCCheckpoint.load_hdf5(handle)
    else:
        checkpoint.save_pickle(path)
        restored = SMCCheckpoint.load_pickle(path)
    reconstructed = adapter(
        AcceptanceRatePolicy(2, min_steps=0, max_steps=20), metadata_key="one"
    )
    metadata = reconstructed.initialize_functional_metadata(
        particle_state, restored.metadata
    )
    assert metadata is restored.metadata
    replayed, budget = reconstructed.prepare_budget_functional(
        StepContext(t=0, metadata=metadata)
    )
    assert int(budget) == 4
    replayed = reconstructed.update_budget_functional(
        MutationMetricsOutput(budget, np.array(10), np.array(0)), replayed
    )
    promoted, budget = reconstructed.prepare_budget_functional(
        replace(replayed, t=1)
    )
    assert int(budget) == 8
    assert int(promoted.metadata["two"].current_n_steps) == 4
    assert promoted.metadata["other"] == 12


def test_custom_functional_opt_in_and_eager_only(particle_state) -> None:
    class Custom(FixedMutationStepPolicy):
        def initialise_functional(self, *, initial_steps, context):
            return initial_steps + 1

        def update_functional(self, *, current_steps, metrics, context):
            return current_steps + 100

    step = adapter(Custom())
    context, budget = step.prepare_budget_functional(
        initialize(step, particle_state)
    )
    assert int(budget) == 5
    context = step.update_budget_functional(
        MutationMetricsOutput(5, 10, 5), context
    )
    assert int(context.metadata[step.metadata_key].next_n_steps) == 20

    class EagerOnly:
        def initialise(self, *, initial_steps, context):
            return initial_steps

        def update(self, *, current_steps, metrics, context):
            return current_steps

    eager = adapter(EagerOnly())
    assert eager._freeze_budget(StepContext(t=0)) == 4
    with pytest.raises(
        NotImplementedError, match="FunctionalMutationStepPolicy"
    ):
        initialize(eager, particle_state)
    with pytest.raises(NotImplementedError, match="eager execution only"):
        step.apply_functional(particle_state, context, None)


def test_invalid_static_configuration_and_missing_carry(
    particle_state,
) -> None:
    with pytest.raises(ValueError, match="metadata_key"):
        adapter(metadata_key="")
    with pytest.raises(TypeError, match="Initialize functional"):
        adapter().prepare_budget_functional(StepContext(t=0))
    with pytest.raises(TypeError, match="MutationBudgetState"):
        adapter().initialize_functional_metadata(
            particle_state, {"mutation_budget": 3}
        )
    large = AdaptMutationSteps(
        UnusedMutation(), initial_steps=1, min_steps=0, max_steps=2**24 + 1
    )
    with pytest.raises(ValueError, match="max_steps"):
        initialize(large, particle_state)
    with pytest.raises(ValueError, match="max_steps"):
        AcceptanceRatePolicy(2, min_steps=0, max_steps=2**24 + 1)
    for carry in (
        MutationBudgetState(*(np.array(0, dtype=np.int64) for _ in range(4))),
        MutationBudgetState[int].initial(4, -1),
    ):
        with pytest.raises(TypeError, match="scalar int32"):
            adapter().initialize_functional_metadata(
                particle_state, {"mutation_budget": carry}
            )


@pytest.mark.parametrize("accepted", [0.0, 1.0])
def test_shared_ceiling_rounding(accepted: float) -> None:
    policy = AcceptanceRatePolicy(
        1.1, min_steps=0, max_steps=20, min_acceptance=0.1
    )
    budget = np.array(4, dtype=np.int32)
    metrics = MutationMetricsOutput(budget, np.array(10.0), np.array(accepted))
    context = StepContext(t=0)
    # Both paths divide before the ceiling, using the same stored binary floats.
    assert (
        policy.update(current_steps=4, metrics=metrics, context=context) == 11
    )
    assert (
        policy.update_functional(
            current_steps=budget, metrics=metrics, context=context
        )
        == 11
    )


def test_scalar_metrics_required_and_inputs_preserved() -> None:
    policy = AcceptanceRatePolicy(2, min_steps=0, max_steps=20)
    budget = np.array(4, dtype=np.int32)
    attempted, accepted = np.array(10.0), np.array(2.5)
    metrics = MutationMetricsOutput(budget, attempted, accepted)
    policy.update_functional(
        current_steps=budget, metrics=metrics, context=StepContext(t=0)
    )
    assert budget == 4
    assert attempted == 10
    assert accepted == 2.5
    with pytest.raises(TypeError, match="scalar"):
        policy.update_functional(
            current_steps=budget,
            metrics=MutationMetricsOutput(budget, np.ones(2), accepted),
            context=StepContext(t=0),
        )


def test_optree_carry_roundtrip(particle_state) -> None:
    import optree

    context = initialize(adapter(), particle_state)
    carry = context.metadata["mutation_budget"]
    leaves, structure = optree.tree_flatten(carry, namespace="sequince")
    assert len(leaves) == 4
    restored = optree.tree_unflatten(structure, leaves)
    assert isinstance(restored, MutationBudgetState)
    assert int(restored.current_iteration) == -1
    assert int(restored.next_n_steps) == 4


class LifecyclePolicy(FixedMutationStepPolicy):
    """Exercise initial decisions and clipping independently of acceptance."""

    def initialise(
        self, *, initial_steps: int | Array, context: StepContext
    ) -> int | Array:
        return initial_steps + 2

    def update(
        self,
        *,
        current_steps: int | Array,
        metrics: MutationMetricsOutput,
        context: StepContext,
    ) -> int | Array:
        return metrics.values["proposal"]

    initialise_functional = initialise
    update_functional = update


@pytest.mark.parametrize("resume_after", [0, 1, 4, 7, 9])
def test_eager_and_functional_lifecycle(
    backend, backend_factory, resume_after: int
) -> None:
    import pickle

    case = backend_factory(backend)
    eager, functional = adapter(LifecyclePolicy()), adapter(LifecyclePolicy())
    context = initialize(functional, case.state())
    assert eager.current_n_steps == eager.next_n_steps == 4
    carry = context.metadata[functional.metadata_key]
    assert int(to_numpy(carry.current_iteration)) == -1
    assert int(to_numpy(carry.last_updated_iteration)) == -1
    assert int(to_numpy(carry.current_n_steps)) == 4
    assert int(to_numpy(carry.next_n_steps)) == 4

    prepare = functional.prepare_budget_functional
    update = functional.update_budget_functional
    if backend == "jax":
        import jax

        prepare, update = jax.jit(prepare), jax.jit(update)

    operations = [
        ("prepare", 3, None, 6, 6),
        ("prepare", 3, None, 6, 6),
        # Advancing without completed metrics retains the frozen budget.
        ("prepare", 4, None, 6, 6),
        ("update", 4, 99, 6, 20),
        ("update", 4, 0, 6, 20),
        ("prepare", 4, None, 6, 20),
        ("prepare", 5, None, 20, 20),
        ("update", 5, -7, 20, 0),
        ("update", 5, 8, 20, 0),
        ("prepare", 6, None, 0, 0),
        ("update", 6, 5, 0, 5),
        ("prepare", 7, None, 5, 5),
    ]
    for index, (operation, iteration, proposal, frozen, pending) in enumerate(
        operations
    ):
        if index == resume_after:
            eager_state, metadata = pickle.loads(
                pickle.dumps((eager.state_dict(), context.metadata))
            )
            eager, functional = (
                adapter(LifecyclePolicy()),
                adapter(LifecyclePolicy()),
            )
            eager.load_state_dict(eager_state)
            initialized = functional.initialize_functional_metadata(
                case.state(), metadata
            )
            assert initialized is metadata
            context = replace(context, metadata=initialized)
            prepare = functional.prepare_budget_functional
            update = functional.update_budget_functional
            if backend == "jax":
                prepare, update = jax.jit(prepare), jax.jit(update)

        eager_context = StepContext(t=iteration)
        context = replace(
            context, t=asarray(iteration, like=case.log_weights, dtype="int32")
        )
        if operation == "prepare":
            assert eager._freeze_budget(eager_context) == frozen
            context, budget = prepare(context)
            assert int(to_numpy(budget)) == frozen
        else:
            metrics = MutationMetricsOutput(
                asarray(frozen, like=case.log_weights, dtype="int32"),
                asarray(0, like=case.log_weights, dtype="int32"),
                asarray(0, like=case.log_weights, dtype="int32"),
                values={
                    "proposal": asarray(
                        proposal, like=case.log_weights, dtype="int32"
                    )
                },
            )
            # This custom diagnostic supplies intentionally out-of-bound
            # budgets while keeping the core mutation counts valid.
            eager_metrics = replace(metrics, values={"proposal": proposal})
            eager._adapt(
                eager_metrics,
                eager_context,
                initial_context=eager_context,
                n_steps=eager.current_n_steps,
            )
            context = update(metrics, context)
        carry = context.metadata[functional.metadata_key]
        assert (
            eager.current_n_steps
            == int(to_numpy(carry.current_n_steps))
            == frozen
        )
        assert eager.next_n_steps == pending
        assert int(to_numpy(carry.next_n_steps)) == pending
        for name, value in eager.state_dict()["budget"].items():
            assert value == int(to_numpy(getattr(carry, name)))


def test_shared_lifecycle_preserves_stateful_eager_only_policy() -> None:
    class EagerPolicy:
        def __init__(self) -> None:
            self.initialisations = 0
            self.updates = 0

        def initialise(
            self, *, initial_steps: int, context: StepContext
        ) -> int:
            self.initialisations += 1
            return initial_steps

        def update(
            self,
            *,
            current_steps: int,
            metrics: MutationMetricsOutput,
            context: StepContext,
        ) -> int:
            self.updates += 1
            return current_steps + 7

        def state_dict(self) -> dict[str, int]:
            return dict(
                initialisations=self.initialisations, updates=self.updates
            )

        def load_state_dict(self, state: dict[str, int]) -> None:
            self.initialisations = state["initialisations"]
            self.updates = state["updates"]

    def build() -> AdaptMutationSteps:
        return AdaptMutationSteps(
            UnusedMutation(),
            EagerPolicy(),
            initial_steps=2**80,
            min_steps=0,
            max_steps=2**81,
        )

    step = build()
    context = StepContext(t=0)
    for _ in range(2):
        assert step._freeze_budget(context) == 2**80
        step._adapt(
            MutationMetricsOutput(2**80, 0, 0),
            context,
            initial_context=context,
            n_steps=2**80,
        )
    saved = step.state_dict()
    assert saved["policy"] == {"initialisations": 1, "updates": 1}
    assert saved["budget"]["next_n_steps"] == 2**80 + 7
    resumed = build()
    resumed.load_state_dict(saved)
    assert resumed._freeze_budget(context) == 2**80
    resumed._adapt(
        MutationMetricsOutput(2**80, 0, 0),
        context,
        initial_context=context,
        n_steps=2**80,
    )
    assert resumed.state_dict()["policy"] == saved["policy"]
    assert resumed._freeze_budget(replace(context, t=1)) == 2**80 + 7


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"current_n_steps": None}, TypeError),
        ({"next_n_steps": True}, TypeError),
        ({"next_n_steps": 1.5}, TypeError),
        ({"next_n_steps": -1}, ValueError),
        ({"next_n_steps": 21}, ValueError),
        ({"current_iteration": -2}, ValueError),
        ({"last_updated_iteration": 0}, ValueError),
    ],
)
def test_complete_budget_checkpoint_validation(changes, error) -> None:
    step = adapter()
    saved = step.state_dict()
    saved["budget"].update(changes)
    before = step.state_dict()
    with pytest.raises(error):
        step.load_state_dict(saved)
    assert step.state_dict() == before


def test_incomplete_and_legacy_checkpoints_rejected() -> None:
    step = adapter()
    before = step.state_dict()
    assert before["budget"] == dict(
        current_iteration=-1,
        current_n_steps=4,
        next_n_steps=4,
        last_updated_iteration=-1,
    )
    with pytest.raises(ValueError, match="budget"):
        step.load_state_dict(
            {"mutation": {}, "policy": {}, "policy_initialised": False}
        )
    incomplete = step.state_dict()
    del incomplete["budget"]["next_n_steps"]
    with pytest.raises(ValueError, match="four carry fields"):
        step.load_state_dict(incomplete)
