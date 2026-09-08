from __future__ import annotations

import numpy as np
import pytest

import sequince.transforms as transforms_module
from sequince.steps.base import SMCStep
from sequince.steps.control import RepeatStep, SequentialStep
from sequince.steps.mutation import (
    CallableMutationKernel,
    MetropolisHastingsMutation,
    MutationStep,
    RepeatedMutationStep,
)
from sequince.transforms import FlowTransform, ParticleTransform


class CountingTransform(ParticleTransform):
    def __init__(
        self, fits: list[np.ndarray], fit_rngs: list[object] | None = None
    ):
        self.fits = fits
        self.fit_rngs = fit_rngs

    def clone(self):
        return CountingTransform(self.fits, self.fit_rngs)

    def fit(self, particles, context=None, *, rng=None):
        self.fits.append(np.asarray(particles).copy())
        if self.fit_rngs is not None:
            self.fit_rngs.append(rng)
        return self


class ReplacingTransform(ParticleTransform):
    def __init__(self, fitted: bool = False):
        self.fitted = fitted

    def fit(self, particles, context=None, *, rng=None):
        del rng
        return ReplacingTransform(fitted=True)


class SignFoldingTransform(ParticleTransform):
    def __init__(self):
        self.inverse_batch_sizes: list[int] = []

    def forward(self, particles, context=None, *, rng=None):
        del context, rng
        values = np.asarray(particles)
        signs = np.where(values < 0, -1.0, 1.0)
        return np.abs(values), np.zeros(values.shape[0]), signs

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        del context, rng
        values = np.asarray(values)
        signs = np.asarray(auxiliary)
        self.inverse_batch_sizes.append(values.shape[0])
        if values.shape[0] != signs.shape[0]:
            raise ValueError(
                "auxiliary is not aligned with the particle batch"
            )
        return values * signs, np.zeros(values.shape[0])


class PreparingMutation(MutationStep):
    def apply(self, state, context, rng):
        _, context = self._prepare_transform(state.particles, context, rng=rng)
        return state, context, None


class CountingGibbsStep(SMCStep):
    def apply(self, state, context, rng):
        context = context.with_metadata(
            gibbs_updates=context.metadata.get("gibbs_updates", 0) + 1
        )
        return state, context, None


class RecordingFixedMutation(MetropolisHastingsMutation):
    def __init__(self):
        super().__init__(
            lambda p, c: np.zeros(p.shape[0]),
            CallableMutationKernel(lambda p, r: p, symmetric=True),
        )
        self.lengths: list[int] = []

    def apply_with_mutation_state(
        self, state, context, rng, *, n_steps, mutation_state=None
    ):
        self.lengths.append(n_steps)
        return super().apply_with_mutation_state(
            state, context, rng, n_steps=n_steps, mutation_state=mutation_state
        )


def _mutation(refit, fits):
    return PreparingMutation(
        lambda particles, context: np.zeros(particles.shape[0]),
        transform=CountingTransform(fits),
        transform_refit=refit,
    )


def test_always_refits_transform_for_each_application(
    particle_state, context_factory, rng
) -> None:
    fits: list[np.ndarray] = []
    mutation = _mutation("always", fits)
    context = context_factory()

    mutation(particle_state, context, rng)
    mutation(particle_state, context, rng)

    assert len(fits) == 2


def test_mutation_passes_rng_when_fitting_transform(
    particle_state, context_factory, rng
) -> None:
    fits: list[np.ndarray] = []
    fit_rngs: list[object] = []
    mutation = PreparingMutation(
        lambda particles, context: np.zeros(particles.shape[0]),
        transform=CountingTransform(fits, fit_rngs),
    )

    mutation(particle_state, context_factory(), rng)

    assert fit_rngs == [rng]


def test_mutation_retains_transform_returned_by_fit(
    particle_state, context_factory, rng
) -> None:
    mutation = PreparingMutation(
        lambda particles, context: np.zeros(particles.shape[0]),
        transform=ReplacingTransform(),
    )

    _, context, _ = mutation(particle_state, context_factory(), rng)

    fitted = context.metadata["particle_transform"]
    assert isinstance(fitted, ReplacingTransform)
    assert fitted.fitted


def test_mutation_clones_flow_once_when_fitting_transform(
    particle_state,
    context_factory,
    rng,
    shift_flow_factory,
    monkeypatch,
) -> None:
    clone_calls = 0
    clone_flow = transforms_module.clone_flow

    def counting_clone(flow):
        nonlocal clone_calls
        clone_calls += 1
        return clone_flow(flow)

    monkeypatch.setattr(transforms_module, "clone_flow", counting_clone)
    mutation = PreparingMutation(
        lambda particles, context: np.zeros(particles.shape[0]),
        transform=FlowTransform(shift_flow_factory(n_dim=2)),
    )

    _, context, _ = mutation(particle_state, context_factory(), rng)

    assert clone_calls == 1
    fitted = context.metadata["particle_transform"]
    assert isinstance(fitted, FlowTransform)
    assert getattr(fitted.flow, "fit_rng") is rng


def test_per_context_refit_reuses_transform_across_alternating_rounds(
    particle_state, context_factory, rng
) -> None:
    fits: list[np.ndarray] = []
    mutation = _mutation("per-context", fits)
    first_context = context_factory(t=0)
    alternating_round = SequentialStep([mutation, CountingGibbsStep()])

    _, first_context, outputs = RepeatStep(alternating_round, n=3)(
        particle_state, first_context, rng
    )
    first_transform = first_context.metadata["particle_transform"]

    assert len(fits) == 1
    assert first_context.metadata["gibbs_updates"] == 3
    assert first_context.metadata["particle_transform"] is first_transform
    assert len(outputs) == 3

    mutation(particle_state, context_factory(t=1), rng)

    assert len(fits) == 2


def test_repeat_sequence_distributes_total_budget_and_keeps_rounds_fixed(
    particle_state, context_factory, rng
) -> None:
    mutation = RecordingFixedMutation()
    rounds = RepeatedMutationStep(
        mutation,
        after_round=SequentialStep([CountingGibbsStep()]),
        n_rounds=4,
    )

    _, context, outputs = rounds.apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=10
    )

    assert mutation.lengths == [3, 3, 2, 2]
    assert context.metadata["gibbs_updates"] == 4
    assert len(outputs.round_outputs) == 4


def test_repeat_sequence_zero_budget_still_runs_nonmutation_round_steps(
    particle_state, context_factory, rng
) -> None:
    mutation = RecordingFixedMutation()
    rounds = RepeatedMutationStep(
        mutation,
        after_round=SequentialStep([CountingGibbsStep()]),
        n_rounds=3,
    )

    _, context, _ = rounds.apply_with_n_steps(
        particle_state, context_factory(), rng, n_steps=0
    )

    assert mutation.lengths == [0, 0, 0]
    assert context.metadata["gibbs_updates"] == 3


def test_sequential_step_runs_multiple_mutations_normally(
    particle_state, context_factory, rng
) -> None:
    first, second = RecordingFixedMutation(), RecordingFixedMutation()
    sequence = SequentialStep([first, second])
    _, _, outputs = sequence(particle_state, context_factory(), rng)
    assert first.lengths == second.lengths == [1]
    assert len(outputs) == 2
    assert all(output["metrics"].n_steps == 1 for output in outputs)


def test_once_refit_reuses_transform_across_contexts(
    particle_state, context_factory, rng
) -> None:
    fits: list[np.ndarray] = []
    mutation = _mutation("once", fits)

    mutation(particle_state, context_factory(t=0), rng)
    mutation(particle_state, context_factory(t=1), rng)

    assert len(fits) == 1


def test_transform_refit_trigger_can_consume_context_flag(
    particle_state, context_factory, rng
) -> None:
    fits: list[np.ndarray] = []
    trigger_calls: list[np.ndarray] = []

    refit_requested = True

    def consume_trigger(particles, context):
        nonlocal refit_requested
        trigger_calls.append(np.asarray(particles).copy())
        result = refit_requested
        refit_requested = False
        return result

    mutation = _mutation(consume_trigger, fits)
    context = context_factory()

    mutation(particle_state, context, rng)

    assert len(fits) == 1
    assert len(trigger_calls) == 1

    mutation(particle_state, context, rng)

    assert len(fits) == 1
    assert len(trigger_calls) == 2

    refit_requested = True
    mutation(particle_state, context, rng)

    assert len(fits) == 2
    assert len(trigger_calls) == 3


def test_false_transform_refit_trigger_still_fits_on_first_use(
    particle_state, context_factory, rng
) -> None:
    fits: list[np.ndarray] = []
    trigger_calls = 0

    def never_refit(particles, context):
        nonlocal trigger_calls
        trigger_calls += 1
        return False

    mutation = _mutation(never_refit, fits)
    context = context_factory()

    mutation(particle_state, context, rng)
    mutation(particle_state, context, rng)

    assert trigger_calls == 2
    assert len(fits) == 1
    np.testing.assert_array_equal(fits[0], particle_state.particles)


def test_transform_refit_is_validated() -> None:
    with pytest.raises(ValueError, match="transform_refit"):
        _mutation("invalid", [])


def test_transform_refit_type_is_validated() -> None:
    with pytest.raises(TypeError, match="transform_refit"):
        _mutation(True, [])


def test_inverse_chain_reuses_particle_auxiliary_for_each_step(
    context_factory,
) -> None:
    transform = SignFoldingTransform()
    mutation = PreparingMutation(
        lambda particles, context: np.zeros(particles.shape[0]),
        transform=transform,
    )
    folded, _, auxiliary = transform.forward(np.array([[-2.0], [3.0]]))
    latent_chain = np.stack([folded + 1.0, folded + 2.0, folded + 3.0])

    chain = mutation._inverse_chain(
        latent_chain,
        transform,
        context_factory(),
        auxiliary=auxiliary,
        like=folded,
    )

    np.testing.assert_array_equal(
        chain,
        [
            [[-3.0], [4.0]],
            [[-4.0], [5.0]],
            [[-5.0], [6.0]],
        ],
    )
    assert transform.inverse_batch_sizes == [6]
