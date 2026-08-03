from __future__ import annotations

import copy
from collections.abc import Sequence

import h5py
import numpy as np
import pytest

from sequince.checkpoint import (
    CheckpointWriter,
    HDF5CheckpointWriter,
    PickleCheckpointWriter,
    SMCCheckpoint,
)
from sequince.result import SMCResult
from sequince.smc import SequentialMonteCarlo
from sequince.state import ParticleState
from sequince.steps.base import SMCStep, StepContext
from sequince.utils import resolve_rng


class StochasticStep(SMCStep):
    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng,
    ) -> ParticleState:
        observation_value = (
            0.0 if context.observation is None else float(context.observation)
        )
        particles = np.asarray(state.particles)
        log_weights = np.asarray(state.log_weights)
        noise = np.asarray(rng.normal(size=particles.shape))
        weight_noise = np.asarray(rng.normal(size=log_weights.shape))

        metadata = copy.deepcopy(state.metadata)
        metadata.setdefault("noise_sums", []).append(float(noise.sum()))
        metadata["iteration"] = context.t

        context.metadata.setdefault("seen_observations", []).append(
            context.observation
        )
        context.metadata["last_iteration"] = context.t
        context.record_output("noise_sums", float(noise.sum()))
        context.record_output("observations", context.observation)

        return ParticleState(
            particles=particles + noise + observation_value,
            log_weights=log_weights + weight_noise + observation_value,
            log_evidence=state.log_evidence,
            metadata=metadata,
        )


class DeterministicStep(SMCStep):
    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng,
    ) -> ParticleState:
        metadata = copy.deepcopy(state.metadata)
        metadata["iteration"] = context.t
        context.record_output("iterations", context.t)
        return ParticleState(
            particles=np.asarray(state.particles) + 1.0,
            log_weights=np.asarray(state.log_weights),
            log_evidence=state.log_evidence,
            metadata=metadata,
        )


class StatefulStep(SMCStep):
    def __init__(self) -> None:
        self.counter = 0

    def apply(self, state, context, rng):
        particles = np.asarray(state.particles) + self.counter
        self.counter += 1
        return state.with_updates(particles=particles)

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state):
        self.counter = int(state["counter"])


class OutputCallback:
    def __call__(
        self,
        state: ParticleState,
        context: StepContext,
        result: SMCResult,
    ) -> None:
        context.metadata.setdefault("callback_iterations", []).append(
            context.t
        )
        context.record_output("callback_iterations", context.t)
        context.metadata["callback_result_length"] = len(result)


class CheckpointRecorder(CheckpointWriter):
    def __init__(self, *, save_rng: bool = True) -> None:
        super().__init__(save_rng=save_rng)
        self.checkpoints: list[SMCCheckpoint] = []

    def _write(self, checkpoint: SMCCheckpoint) -> None:
        self.checkpoints.append(checkpoint)


def _assert_states_equal(left: ParticleState, right: ParticleState) -> None:
    np.testing.assert_allclose(left.particles, right.particles)
    np.testing.assert_allclose(left.log_weights, right.log_weights)
    assert left.log_evidence == pytest.approx(right.log_evidence)
    assert (left.evaluation is None) == (right.evaluation is None)
    if left.evaluation is not None and right.evaluation is not None:
        np.testing.assert_allclose(
            left.evaluation.log_prior,
            right.evaluation.log_prior,
        )
        np.testing.assert_allclose(
            left.evaluation.log_likelihood,
            right.evaluation.log_likelihood,
        )
        assert (left.evaluation.log_proposal is None) == (
            right.evaluation.log_proposal is None
        )
        if (
            left.evaluation.log_proposal is not None
            and right.evaluation.log_proposal is not None
        ):
            np.testing.assert_allclose(
                left.evaluation.log_proposal,
                right.evaluation.log_proposal,
            )
    assert left.metadata == right.metadata


def _assert_results_equal(
    left: SMCResult,
    right: SMCResult,
) -> None:
    assert len(left.states) == len(right.states)
    for left_state, right_state in zip(left.states, right.states):
        _assert_states_equal(left_state, right_state)
    assert (left.initial_state is None) == (right.initial_state is None)
    if left.initial_state is not None and right.initial_state is not None:
        _assert_states_equal(left.initial_state, right.initial_state)
    assert left.context.t == right.context.t
    assert (
        left.context.next_observation_index
        == right.context.next_observation_index
    )
    assert left.context.observation == right.context.observation
    assert dict(left.context.shared) == dict(right.context.shared)
    assert left.context.metadata == right.context.metadata
    assert left.context.outputs == right.context.outputs


def _make_smc(
    *checkpoint_writers, step: SMCStep | None = None
) -> SequentialMonteCarlo:
    return SequentialMonteCarlo(
        steps=[step or StochasticStep()],
        callbacks=[OutputCallback()],
        checkpoint_writers=checkpoint_writers,
    )


def _run_with_checkpoint_split(
    particle_state: ParticleState,
    observations: Sequence[float] | None,
    *,
    total_steps: int,
    split_after: int,
) -> tuple[SMCResult, SMCResult, SMCCheckpoint]:
    full_result = _make_smc().run(
        particle_state,
        observations=observations,
        num_steps=total_steps,
        rng=resolve_rng(2024),
        shared_context={"alpha": 1.5},
        metadata={"tag": "run"},
        outputs={"seed_output": ["start"]},
    )

    recorder = CheckpointRecorder()
    _make_smc(recorder).run(
        particle_state,
        observations=observations,
        num_steps=split_after,
        rng=resolve_rng(2024),
        shared_context={"alpha": 1.5},
        metadata={"tag": "run"},
        outputs={"seed_output": ["start"]},
    )

    runtime = _make_smc().resume_from(
        recorder.checkpoints[-1].to_dict(),
        rng=resolve_rng(999),
        shared_context={"alpha": 1.5},
    )
    resumed_result = _make_smc().run(
        runtime=runtime,
        observations=observations,
        num_steps=total_steps - split_after,
    )

    return full_result, resumed_result, recorder.checkpoints[-1]


def test_checkpoint_round_trip_preserves_runtime_objects(
    particle_state: ParticleState,
) -> None:
    context = StepContext(
        t=2,
        next_observation_index=3,
        observation={"batch": 2},
        shared={"alpha": 1.5},
        metadata={"notes": ["x"]},
        outputs={"metric": [1.0, 2.0]},
    )
    state = particle_state.with_updates(
        metadata={"temperature": 0.5, "nested": {"value": [1, 2, 3]}}
    )
    result = SMCResult(
        states=[particle_state, state],
        context=context,
        initial_state=particle_state,
    )
    rng = np.random.default_rng(123)
    rng.normal(size=3)

    checkpoint = SMCCheckpoint.from_runtime(
        state=state,
        context=context,
        result=result,
        steps=[],
        rng=rng,
    )
    restored = SMCCheckpoint.from_dict(checkpoint.to_dict())

    _assert_states_equal(checkpoint.state, restored.state)
    _assert_results_equal(checkpoint.result(), restored.result())
    assert restored.initial_state is not None
    assert checkpoint.metadata == restored.metadata
    assert checkpoint.outputs == restored.outputs
    assert restored.observation == {"batch": 2}
    restored_rng = restored.restore_rng()
    assert restored_rng is not None
    np.testing.assert_allclose(
        rng.normal(size=4),
        restored_rng.normal(size=4),
    )


def test_checkpoint_version_one_payload_without_observation_still_loads(
    particle_state: ParticleState,
) -> None:
    context = StepContext(t=0, observation={"batch": 1})
    checkpoint = SMCCheckpoint.from_runtime(
        state=particle_state,
        context=context,
        result=SMCResult(states=[particle_state], context=context),
        steps=[],
    )
    payload = checkpoint.to_dict()
    payload.pop("observation")

    restored = SMCCheckpoint.from_dict(payload)

    assert restored.observation is None
    assert restored.result().context.observation is None


def test_checkpoint_round_trip_allows_missing_rng_state(
    particle_state: ParticleState,
) -> None:
    context = StepContext(t=1, next_observation_index=2)
    result = SMCResult(states=[particle_state], context=context)

    checkpoint = SMCCheckpoint.from_runtime(
        state=particle_state,
        context=context,
        result=result,
        steps=[],
        rng=None,
    )
    restored = SMCCheckpoint.from_dict(checkpoint.to_dict())

    assert restored.rng_state is None
    assert restored.restore_rng() is None


def test_checkpoint_writers_run_after_callbacks(
    particle_state: ParticleState,
) -> None:
    recorder = CheckpointRecorder()
    result = _make_smc(recorder).run(
        particle_state,
        observations=[0.0, 1.0, 2.0],
        num_steps=3,
        rng=resolve_rng(123),
    )

    assert len(recorder.checkpoints) == 3
    assert [
        checkpoint.next_iteration for checkpoint in recorder.checkpoints
    ] == [
        1,
        2,
        3,
    ]
    assert recorder.checkpoints[-1].next_observation_index == 3
    assert recorder.checkpoints[-1].outputs["callback_iterations"] == [
        0,
        1,
        2,
    ]
    assert recorder.checkpoints[-1].metadata["callback_iterations"] == [
        0,
        1,
        2,
    ]
    assert (
        recorder.checkpoints[-1].result().context.outputs
        == result.context.outputs
    )


def test_hdf5_checkpoint_writer_writes_latest_checkpoint(
    particle_state: ParticleState,
    tmp_path,
) -> None:
    file_path = tmp_path / "checkpoint.h5"
    writer = HDF5CheckpointWriter(file_path, path="checkpoint", every=1)
    result = _make_smc(writer).run(
        particle_state,
        observations=[0.0, 1.0, 2.0],
        num_steps=3,
        rng=resolve_rng(123),
        metadata={
            "nested": {
                "values": [1, 2, 3],
                "unsupported_leaf": range(2, 7, 2),
            }
        },
    )

    with h5py.File(file_path, "r") as h5_file:
        assert len(h5_file["checkpoint/history"]) == 3
        runtime = h5_file["checkpoint/runtime"]
        assert isinstance(runtime, h5py.Group)
        assert isinstance(runtime["state"], h5py.Group)
        assert isinstance(runtime["state/particles"], h5py.Dataset)
        assert isinstance(runtime["metadata/nested"], h5py.Group)
        assert isinstance(runtime["metadata/nested/values"], h5py.Group)
        assert (
            runtime["metadata/nested/unsupported_leaf"].attrs["_sequince_kind"]
            == "pickle"
        )
        assert isinstance(
            h5_file["checkpoint/history/00000000/particles"], h5py.Dataset
        )
        checkpoint = SMCCheckpoint.load_hdf5(h5_file, path="checkpoint")

    _assert_states_equal(checkpoint.state, result.final_state)
    _assert_results_equal(checkpoint.result(), result)
    assert checkpoint.next_iteration == 3
    assert checkpoint.next_observation_index == 3
    assert checkpoint.rng_state is not None


def test_hdf5_checkpoint_writer_respects_every_and_optional_rng(
    particle_state: ParticleState,
    tmp_path,
) -> None:
    file_path = tmp_path / "checkpoint_every.hdf5"
    writer = HDF5CheckpointWriter(
        file_path,
        path="checkpoint",
        every=2,
        save_rng=False,
    )
    result = _make_smc(writer).run(
        particle_state,
        observations=[0.0, 1.0, 2.0],
        num_steps=3,
        rng=resolve_rng(123),
    )

    with h5py.File(file_path, "r") as h5_file:
        checkpoint = SMCCheckpoint.load_hdf5(h5_file, path="checkpoint")

    assert checkpoint.next_iteration == 2
    assert checkpoint.next_observation_index == 2
    assert checkpoint.rng_state is None
    _assert_states_equal(checkpoint.state, result.states[1])


def test_hdf5_checkpoint_can_omit_result_history(
    particle_state: ParticleState,
    tmp_path,
) -> None:
    file_path = tmp_path / "checkpoint_without_history.h5"
    writer = HDF5CheckpointWriter(file_path, include_history=False)
    _make_smc(writer).run(
        particle_state,
        num_steps=2,
        rng=123,
    )

    with h5py.File(file_path, "r") as h5_file:
        assert "history" not in h5_file["checkpoint"]
        checkpoint = SMCCheckpoint.load_hdf5(h5_file)

    assert checkpoint.history is None
    assert len(checkpoint.result()) == 1


def test_hdf5_writer_replaces_history_when_reused_for_a_fresh_run(
    particle_state: ParticleState,
    tmp_path,
) -> None:
    file_path = tmp_path / "reused_writer.h5"
    writer = HDF5CheckpointWriter(file_path, every=2)
    smc = SequentialMonteCarlo(steps=[], checkpoint_writers=[writer])

    smc.run(particle_state, num_steps=2, rng=1)
    replacement_state = particle_state.with_updates(
        particles=np.full_like(particle_state.particles, 9.0)
    )
    replacement_result = smc.run(replacement_state, num_steps=2, rng=2)

    with h5py.File(file_path, "r") as h5_file:
        checkpoint = SMCCheckpoint.load_hdf5(h5_file)

    _assert_results_equal(checkpoint.result(), replacement_result)


def test_pickle_checkpoint_writer_writes_latest_checkpoint(
    particle_state: ParticleState,
    tmp_path,
) -> None:
    file_path = tmp_path / "checkpoint.pkl"
    writer = PickleCheckpointWriter(file_path, every=1)
    result = _make_smc(writer).run(
        particle_state,
        observations=[0.0, 1.0, 2.0],
        num_steps=3,
        rng=resolve_rng(123),
    )

    checkpoint = SMCCheckpoint.load_pickle(file_path)

    _assert_states_equal(checkpoint.state, result.final_state)
    _assert_results_equal(checkpoint.result(), result)
    assert checkpoint.next_iteration == 3
    assert checkpoint.next_observation_index == 3
    assert checkpoint.rng_state is not None
    assert not file_path.with_name(file_path.name + ".tmp").exists()


def test_resume_matches_uninterrupted_run_with_sequence_observations(
    particle_state: ParticleState,
) -> None:
    observations = [0.5, -1.0, 2.0, 0.25]
    full_result, resumed_result, checkpoint = _run_with_checkpoint_split(
        particle_state,
        observations,
        total_steps=4,
        split_after=2,
    )

    _assert_results_equal(full_result, resumed_result)
    assert checkpoint.next_observation_index == 2
    assert len(resumed_result.states) == len(full_result.states)
    _assert_states_equal(resumed_result.states[1], full_result.states[1])
    _assert_states_equal(resumed_result.states[2], full_result.states[2])


def test_resume_matches_uninterrupted_run_with_none_observations(
    particle_state: ParticleState,
) -> None:
    full_result, resumed_result, checkpoint = _run_with_checkpoint_split(
        particle_state,
        None,
        total_steps=4,
        split_after=2,
    )

    _assert_results_equal(full_result, resumed_result)
    assert checkpoint.next_observation_index == 0


def test_resume_without_saved_rng_works_for_deterministic_steps(
    particle_state: ParticleState,
) -> None:
    full_result = _make_smc(step=DeterministicStep()).run(
        particle_state,
        observations=[0.0, 1.0, 2.0, 3.0],
        num_steps=4,
        rng=resolve_rng(2024),
    )

    recorder = CheckpointRecorder(save_rng=False)
    _make_smc(recorder, step=DeterministicStep()).run(
        particle_state,
        observations=[0.0, 1.0, 2.0, 3.0],
        num_steps=2,
        rng=resolve_rng(2024),
    )

    resumed_result = _make_smc(step=DeterministicStep()).run(
        runtime=_make_smc(step=DeterministicStep()).resume_from(
            recorder.checkpoints[-1],
            rng=resolve_rng(999),
        ),
        observations=[0.0, 1.0, 2.0, 3.0],
        num_steps=2,
    )

    _assert_results_equal(full_result, resumed_result)
    assert recorder.checkpoints[-1].rng_state is None


@pytest.mark.parametrize("immediate", [False, True])
def test_resume_clears_transient_stop_controls(
    particle_state: ParticleState,
    immediate: bool,
) -> None:
    step = DeterministicStep()
    context = StepContext(t=0)
    if immediate:
        context.request_stop_immediately()
    else:
        context.request_stop()
    checkpoint = SMCCheckpoint.from_runtime(
        state=particle_state,
        context=context,
        result=SMCResult(states=[particle_state], context=context),
        steps=[step],
        rng=123,
    )
    resumed_smc = _make_smc(step=DeterministicStep())

    runtime = resumed_smc.resume_from(checkpoint)

    assert not runtime.metadata.get("stop_requested", False)
    assert not runtime.metadata.get("stop_immediately", False)
    result = resumed_smc.run(runtime=runtime, num_steps=2)
    assert len(result.states) == 3


def test_resume_restores_stateful_step(particle_state: ParticleState) -> None:
    full_step = StatefulStep()
    full_result = _make_smc(step=full_step).run(
        particle_state,
        num_steps=4,
        rng=123,
    )

    recorder = CheckpointRecorder()
    first_step = StatefulStep()
    _make_smc(recorder, step=first_step).run(
        particle_state,
        num_steps=2,
        rng=123,
    )

    resumed_step = StatefulStep()
    resumed_smc = _make_smc(step=resumed_step)
    runtime = resumed_smc.resume_from(recorder.checkpoints[-1])
    resumed_result = resumed_smc.run(runtime=runtime, num_steps=2)

    _assert_results_equal(full_result, resumed_result)
    assert resumed_step.counter == 4


def test_emcee_resume_uses_checkpointed_master_rng(
    particle_state: ParticleState,
) -> None:
    pytest.importorskip("emcee")
    from sequince.steps.mutation import EmceeMutation

    def log_prob(particles):
        values = np.asarray(particles)
        return -0.5 * np.sum(values**2, axis=-1)

    full_result = _make_smc(
        step=EmceeMutation(log_prob, n_steps=2, vectorize=True)
    ).run(particle_state, num_steps=3, rng=123)

    recorder = CheckpointRecorder()
    _make_smc(
        recorder,
        step=EmceeMutation(log_prob, n_steps=2, vectorize=True),
    ).run(particle_state, num_steps=1, rng=123)

    resumed_smc = _make_smc(
        step=EmceeMutation(log_prob, n_steps=2, vectorize=True)
    )
    runtime = resumed_smc.resume_from(recorder.checkpoints[-1])
    resumed_result = resumed_smc.run(runtime=runtime, num_steps=2)

    _assert_results_equal(full_result, resumed_result)


def test_resume_rejects_different_step_pipeline(
    particle_state: ParticleState,
) -> None:
    recorder = CheckpointRecorder()
    _make_smc(recorder, step=StatefulStep()).run(
        particle_state,
        num_steps=1,
        rng=123,
    )

    with pytest.raises(ValueError, match="configured step"):
        _make_smc(step=DeterministicStep()).resume_from(
            recorder.checkpoints[-1]
        )


def test_resume_with_non_sequence_observations_raises(
    particle_state: ParticleState,
) -> None:
    recorder = CheckpointRecorder()
    _make_smc(recorder).run(
        particle_state,
        observations=[0.0, 1.0],
        num_steps=1,
        rng=resolve_rng(123),
    )

    with pytest.raises(ValueError, match="indexable sequence"):
        runtime = _make_smc().resume_from(
            recorder.checkpoints[-1],
            rng=resolve_rng(123),
        )
        _make_smc().run(
            runtime=runtime,
            observations=iter([1.0, 2.0]),
            num_steps=1,
        )


def test_run_rejects_initial_state_and_runtime_together(
    particle_state: ParticleState,
) -> None:
    step = StochasticStep()
    smc = _make_smc(step=step)
    runtime = smc.resume_from(
        SMCCheckpoint.from_runtime(
            state=particle_state,
            context=StepContext(t=0, next_observation_index=0),
            result=SMCResult(
                states=[particle_state], context=StepContext(t=0)
            ),
            steps=[step],
        )
    )

    with pytest.raises(ValueError, match="either initial_state or runtime"):
        _make_smc().run(
            particle_state,
            runtime=runtime,
            num_steps=1,
        )


def test_malformed_checkpoint_payload_raises() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        SMCCheckpoint.from_dict({"state": {}})

    with pytest.raises(ValueError, match="Unsupported SMC checkpoint version"):
        SMCCheckpoint.from_dict(
            {
                "version": 99,
                "state": {},
                "next_iteration": 1,
                "next_observation_index": 0,
                "metadata": {},
                "outputs": {},
                "step_states": [],
            }
        )

    with pytest.raises(TypeError, match="rng_state"):
        SMCCheckpoint.from_dict(
            {
                "version": 1,
                "state": {
                    "particles": np.zeros((2, 1)),
                    "log_weights": np.zeros(2),
                    "log_evidence": 0.0,
                    "metadata": {},
                },
                "next_iteration": 1,
                "next_observation_index": 0,
                "metadata": {},
                "outputs": {},
                "step_states": [],
                "rng_state": "bad",
                "history": None,
            }
        )
