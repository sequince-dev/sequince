from __future__ import annotations

import numpy as np
import pytest

import sequince.transforms as transforms_module
from sequince.steps.base import SMCStep
from sequince.steps.control import RepeatStep, SequentialStep
from sequince.steps.mutation import MutationStep
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
