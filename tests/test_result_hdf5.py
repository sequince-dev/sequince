from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pytest

from sequince.checkpoint import SMCCheckpoint
from sequince.mcmc import MCMCResult
from sequince.result import MetaSMCResult, SMCResult
from sequince.state import ParticleState, PersistentParticleState
from sequince.steps.context import StepContext
from sequince.steps.control import ConditionalStepOutput
from sequince.steps.mutation.blackjax import BlackJAXOutput
from sequince.steps.mutation.metrics import MutationMetricsOutput
from sequince.steps.mutation.minipcn import MiniPCNOutput
from sequince.steps.resampling import ResamplingOutput
from sequince.steps.transformations import SMCTransformation
from sequince.target import EvaluationCounts, TargetEvaluation


def make_evaluation() -> TargetEvaluation:
    return TargetEvaluation(
        {"log_prior": np.array([-1.0, -2.0])},
        target_id="target",
        evaluation_key=("beta", np.float32(0.5), True),
    )


def make_state(*, persistent: bool = False):
    kwargs = {
        "particles": np.array([[1.0], [2.0]]),
        "log_weights": np.log([0.25, 0.75]),
        "log_normalizing_constant": 1.25,
        "initial_ancestor_indices": np.array([4, 5]),
        "evaluation": make_evaluation(),
        "metadata": {"label": "saved", "nested": [1, None]},
    }
    if not persistent:
        return ParticleState(**kwargs)
    return PersistentParticleState(
        **kwargs,
        betas=np.array([0.5]),
        log_normalizing_constants=np.array([1.0]),
        batch_size=2,
    )


def assert_state_equal(actual, expected) -> None:
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual.particles, expected.particles)
    np.testing.assert_array_equal(actual.log_weights, expected.log_weights)
    np.testing.assert_array_equal(
        actual.initial_ancestor_indices, expected.initial_ancestor_indices
    )
    assert actual.log_normalizing_constant == pytest.approx(
        expected.log_normalizing_constant
    )
    assert actual.metadata == expected.metadata
    assert actual.evaluation.target_id == expected.evaluation.target_id
    np.testing.assert_array_equal(
        actual.evaluation["log_prior"], expected.evaluation["log_prior"]
    )
    if isinstance(expected, PersistentParticleState):
        np.testing.assert_array_equal(actual.betas, expected.betas)
        np.testing.assert_array_equal(
            actual.log_normalizing_constants,
            expected.log_normalizing_constants,
        )
        assert actual.batch_size == expected.batch_size


@pytest.mark.parametrize("persistent", [False, True])
def test_state_round_trip_and_dispatch(tmp_path, persistent) -> None:
    expected = make_state(persistent=persistent)
    with h5py.File(tmp_path / "state.h5", "w") as h5_file:
        expected.save(h5_file)
    with h5py.File(tmp_path / "state.h5", "r") as h5_file:
        actual = ParticleState.load(h5_file)
        if persistent:
            strict = PersistentParticleState.load(h5_file)
            assert_state_equal(strict, expected)
    assert_state_equal(actual, expected)


def test_persistent_loader_rejects_normal_state(tmp_path) -> None:
    with h5py.File(tmp_path / "state.h5", "w") as h5_file:
        make_state().save(h5_file)
    with h5py.File(tmp_path / "state.h5", "r") as h5_file:
        with pytest.raises(ValueError, match="kind"):
            PersistentParticleState.load(h5_file)


def test_target_evaluation_round_trip_and_portable_keys(tmp_path) -> None:
    expected = make_evaluation()
    with h5py.File(tmp_path / "evaluation.h5", "w") as h5_file:
        expected.save(h5_file)
    with h5py.File(tmp_path / "evaluation.h5", "r") as h5_file:
        actual = TargetEvaluation.load(h5_file)
    assert actual.evaluation_key == expected.evaluation_key
    np.testing.assert_array_equal(actual["log_prior"], expected["log_prior"])

    for key in ({"x": 1}, [1], np.array([1]), {1}):
        with pytest.raises(TypeError, match="evaluation key"):
            invalid = TargetEvaluation({"x": np.ones(2)}, evaluation_key=key)
            invalid.encode_for_hdf5()


def test_smc_result_round_trip_all_fields_and_outputs(tmp_path) -> None:
    initial = make_state()
    final = make_state(persistent=True)
    context = StepContext(
        t=3,
        next_observation_index=4,
        observation=np.array([99]),
        shared={"live": object()},
        metadata={
            "beta": 0.5,
            "particle_transform": object(),
            "_functional_particle_transform": object(),
            "smc_transformation": object(),
            "log_proposal_override": lambda x: x,
            "nested": {"particle_transform": "user value"},
        },
        evaluation_counts=EvaluationCounts(
            log_target_evaluations=12,
            log_likelihood_evaluations=9,
            log_proposal_evaluations=7,
            log_increment_evaluations=5,
        ),
        stop_requested=True,
        stop_immediately=False,
    )
    outputs = (
        (
            ConditionalStepOutput(
                True,
                ResamplingOutput(1.5, True, np.array([1, 0]), "systematic"),
            ),
            MiniPCNOutput(
                acceptance_by_step=np.array([0.5]),
                step_fn="pcn",
                steps=2,
                chain=None,
            ),
            BlackJAXOutput(
                mean_acceptance_probability=np.array([0.75]),
                divergence=np.array([False]),
                algorithm="hmc",
                steps=3,
                chain=None,
                num_integration_steps=np.array([[2, 3]]),
                metrics=MutationMetricsOutput(
                    3, 6, 4, values={"custom_metric": 0.5}
                ),
            ),
            {"nested": [np.array([1, 2]), ("x", None)]},
        ),
    )
    expected = SMCResult(
        states=[initial, final],
        context=context,
        initial_state=initial,
        outputs=outputs,
    )

    with h5py.File(tmp_path / "result.h5", "w") as h5_file:
        expected.save(h5_file)
    with h5py.File(tmp_path / "result.h5", "r") as h5_file:
        actual = SMCResult.load(h5_file)

    assert len(actual.states) == 2
    assert_state_equal(actual.states[0], initial)
    assert_state_equal(actual.states[1], final)
    assert_state_equal(actual.initial_state, initial)
    assert actual.context.t == 3
    assert actual.context.next_observation_index == 4
    assert actual.context.stop_requested is True
    assert actual.context.stop_immediately is False
    assert actual.context.shared == {}
    assert actual.context.observation is None
    assert actual.context.metadata == {
        "beta": 0.5,
        "nested": {"particle_transform": "user value"},
    }
    assert actual.evaluation_counts == expected.evaluation_counts
    assert isinstance(actual.outputs[0][0], dict)
    assert (
        actual.outputs[0][0]["child_output"]["resampling_scheme"]
        == "systematic"
    )
    assert isinstance(actual.outputs[0][1], dict)
    assert isinstance(actual.outputs[0][2], dict)
    assert actual.outputs[0][2]["algorithm"] == "hmc"
    assert actual.outputs[0][2]["metrics"] == {
        "n_steps": 3,
        "n_attempted": 6,
        "n_accepted": 4,
        "values": {"custom_metric": 0.5},
    }
    np.testing.assert_array_equal(
        actual.outputs[0][3]["nested"][0], np.array([1, 2])
    )
    assert actual.log_evidence() == pytest.approx(expected.log_evidence())


def test_smc_result_save_to_file_preserves_existing_content(tmp_path) -> None:
    file_path = tmp_path / "result.h5"
    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset("unrelated", data=7)

    expected = SMCResult([make_state()], StepContext(t=0))
    expected.save_to_file(file_path)

    with h5py.File(file_path, "r") as h5_file:
        assert h5_file["unrelated"][()] == 7
        actual = SMCResult.load(h5_file)
    assert_state_equal(actual.final_state, expected.final_state)

    with pytest.raises(FileExistsError):
        expected.save_to_file(file_path)


def test_meta_and_nested_smc_results_round_trip(tmp_path) -> None:
    child = SMCResult([make_state()], StepContext(t=0))
    parent = SMCResult(
        [make_state()], StepContext(t=1), outputs=(({"child": child},),)
    )
    expected = MetaSMCResult([child, parent])
    with h5py.File(tmp_path / "meta.h5", "w") as h5_file:
        expected.save(h5_file)
    with h5py.File(tmp_path / "meta.h5", "r") as h5_file:
        actual = MetaSMCResult.load(h5_file)
    assert len(actual.results) == 2
    assert isinstance(actual.results[1].outputs[0][0]["child"], SMCResult)


def test_mcmc_result_round_trip(tmp_path) -> None:
    chain = np.arange(8.0).reshape(2, 2, 2)
    expected = MCMCResult(
        states=[make_state()],
        context=StepContext(t=2),
        outputs=(({"chain": chain, "acceptance": 0.75},),),
    )
    with h5py.File(tmp_path / "mcmc.h5", "w") as h5_file:
        expected.save(h5_file)
    with h5py.File(tmp_path / "mcmc.h5", "r") as h5_file:
        actual = MCMCResult.load(h5_file)
    np.testing.assert_array_equal(actual.chain, expected.chain)
    assert actual.diagnostics == {"acceptance": 0.75}


def test_wrong_kind_schema_and_incompatible_formats_are_rejected(
    tmp_path,
) -> None:
    path = tmp_path / "formats.h5"
    checkpoint = SMCCheckpoint(
        states=(make_state(),),
        next_iteration=1,
        next_observation_index=0,
        metadata={},
        outputs=(),
        step_states=(),
    )
    with h5py.File(path, "w") as h5_file:
        make_state().save(h5_file, "wrong_kind")
        payload = make_state().encode_for_hdf5()
        payload["_sequince_schema_version"] = 999
        from sequince.utils.hdf5 import recursively_save_to_h5_file

        recursively_save_to_h5_file(h5_file, "wrong_schema", payload)
        checkpoint.save_hdf5(h5_file)
    with h5py.File(path, "r") as h5_file:
        with pytest.raises(ValueError, match="kind"):
            SMCResult.load(h5_file, "wrong_kind")
        with pytest.raises(ValueError, match="schema version"):
            ParticleState.load(h5_file, "wrong_schema")
        with pytest.raises((KeyError, ValueError, TypeError)):
            SMCResult.load(h5_file, "checkpoint")
        with pytest.raises(KeyError):
            SMCCheckpoint.load_hdf5(h5_file, "wrong_kind")


def test_checkpoint_preserves_persistent_state(tmp_path) -> None:
    persistent = make_state(persistent=True)
    initial = make_state(persistent=True)
    checkpoint = SMCCheckpoint(
        states=(persistent,),
        next_iteration=2,
        next_observation_index=1,
        metadata={},
        outputs=(),
        step_states=(),
        initial_state=initial,
    )

    payload = checkpoint.to_dict()
    assert payload["version"] == 1
    assert payload["states"][0]["_sequince_state_kind"] == (
        "persistent_particle_state"
    )
    restored = SMCCheckpoint.from_dict(payload)
    assert_state_equal(restored.state, persistent)
    assert_state_equal(restored.initial_state, initial)

    with h5py.File(tmp_path / "checkpoint.h5", "w") as h5_file:
        checkpoint.save_hdf5(h5_file)
    with h5py.File(tmp_path / "checkpoint.h5", "r") as h5_file:
        restored_hdf5 = SMCCheckpoint.load_hdf5(h5_file)
    assert_state_equal(restored_hdf5.state, persistent)
    assert_state_equal(restored_hdf5.initial_state, initial)


def test_transformation_output_respects_strict_mode(tmp_path) -> None:
    result = SMCResult(
        [make_state()],
        StepContext(t=0),
        outputs=(({"transformation": SMCTransformation()},),),
    )
    with h5py.File(tmp_path / "transformation.h5", "w") as h5_file:
        with pytest.raises(TypeError, match="outputs"):
            result.save(h5_file, strict=True)
        assert "result" not in h5_file
        with pytest.warns(UserWarning, match="outputs/0/0/transformation"):
            result.save(h5_file)
    with h5py.File(tmp_path / "transformation.h5", "r") as h5_file:
        loaded = SMCResult.load(h5_file)
    assert loaded.outputs[0][0]["transformation"] == {}


def test_custom_output_encode_for_hdf5_protocol(tmp_path) -> None:
    class CustomOutput:
        def encode_for_hdf5(self):
            return {"value": np.array([1.0, 2.0])}

    result = SMCResult(
        [make_state()], StepContext(t=0), outputs=((CustomOutput(),),)
    )
    with h5py.File(tmp_path / "custom-output.h5", "w") as h5_file:
        result.save(h5_file)
    with h5py.File(tmp_path / "custom-output.h5", "r") as h5_file:
        loaded = SMCResult.load(h5_file)
    assert isinstance(loaded.outputs[0][0], dict)
    np.testing.assert_array_equal(
        loaded.outputs[0][0]["value"], np.array([1.0, 2.0])
    )


def test_output_dataclass_is_converted_with_warning_by_default(
    tmp_path,
) -> None:
    @dataclass
    class CustomOutput:
        value: np.ndarray

    result = SMCResult(
        [make_state()],
        StepContext(t=0),
        outputs=((CustomOutput(np.array([3.0])),),),
    )
    with h5py.File(tmp_path / "dataclass-output.h5", "w") as h5_file:
        with pytest.raises(TypeError, match="CustomOutput"):
            result.save(h5_file, strict=True)
        assert "result" not in h5_file
        with pytest.warns(UserWarning, match="outputs/0/0"):
            result.save(h5_file)
    with h5py.File(tmp_path / "dataclass-output.h5", "r") as h5_file:
        loaded = SMCResult.load(h5_file)
    assert isinstance(loaded.outputs[0][0], dict)
    np.testing.assert_array_equal(
        loaded.outputs[0][0]["value"], np.array([3.0])
    )


def test_unknown_state_subtype_and_unsupported_context_metadata_fail(
    tmp_path,
) -> None:
    class CustomState(ParticleState):
        pass

    custom = CustomState(np.ones((2, 1)))
    with h5py.File(tmp_path / "unsupported.h5", "w") as h5_file:
        with pytest.raises(TypeError, match="CustomState"):
            custom.save(h5_file)
        result = SMCResult(
            [make_state()],
            StepContext(t=0, metadata={"user_callable": lambda x: x}),
        )
        with pytest.raises(TypeError, match="metadata/user_callable"):
            result.save(h5_file)


def test_jax_source_arrays_load_as_numpy(tmp_path) -> None:
    jnp = pytest.importorskip("jax.numpy")
    state = ParticleState(jnp.asarray([[1.0], [2.0]]))
    with h5py.File(tmp_path / "jax.h5", "w") as h5_file:
        state.save(h5_file)
    with h5py.File(tmp_path / "jax.h5", "r") as h5_file:
        loaded = ParticleState.load(h5_file)
    assert isinstance(loaded.particles, np.ndarray)
    np.testing.assert_array_equal(loaded.particles, np.array([[1.0], [2.0]]))


def test_torch_source_arrays_load_as_numpy(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    state = ParticleState(torch.tensor([[1.0], [2.0]]))
    with h5py.File(tmp_path / "torch.h5", "w") as h5_file:
        state.save(h5_file)
    with h5py.File(tmp_path / "torch.h5", "r") as h5_file:
        loaded = ParticleState.load(h5_file)
    assert isinstance(loaded.particles, np.ndarray)
    np.testing.assert_array_equal(loaded.particles, np.array([[1.0], [2.0]]))


def test_cupy_source_arrays_load_as_numpy(tmp_path) -> None:
    cupy = pytest.importorskip("cupy")
    state = ParticleState(cupy.asarray([[1.0], [2.0]]))
    with h5py.File(tmp_path / "cupy.h5", "w") as h5_file:
        state.save(h5_file)
    with h5py.File(tmp_path / "cupy.h5", "r") as h5_file:
        loaded = ParticleState.load(h5_file)
    assert isinstance(loaded.particles, np.ndarray)
    np.testing.assert_array_equal(loaded.particles, np.array([[1.0], [2.0]]))
