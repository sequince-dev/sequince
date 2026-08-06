from __future__ import annotations

import numpy as np
import pytest

from sequince.steps.base import SMCStep
from sequince.steps.control import RepeatStep, SequentialStep
from sequince.steps.mutation import MutationStep
from sequince.transforms import ParticleTransform


class CountingTransform(ParticleTransform):
    def __init__(self, fits: list[np.ndarray]):
        self.fits = fits

    def clone(self):
        return CountingTransform(self.fits)

    def fit(self, particles, context=None):
        self.fits.append(np.asarray(particles).copy())
        return self


class PreparingMutation(MutationStep):
    def apply(self, state, context, rng):
        _, context = self._prepare_transform(state.particles, context)
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
