from __future__ import annotations

import numpy as np
import pytest

from sequince import ParticleState
from sequince.array_api import namespace
from sequince.steps.base import SMCStep
from sequince.steps.control import SequentialStep
from sequince.steps.mutation import (
    CallableMutationKernel,
    MetropolisHastingsMutation,
    MutateIfResampled,
    RepeatedMutationStep,
)


def mutation(n_steps=2):
    return MetropolisHastingsMutation(
        lambda p, c: namespace(p).zeros(p.shape[0]),
        CallableMutationKernel(lambda p, r: p + 1, symmetric=True),
        n_steps=n_steps,
    )


class ShiftAfterRound(SMCStep):
    def __init__(self):
        self.calls = 0

    def apply(self, state, context, rng):
        self.calls += 1
        return (
            state.replace_particles(state.particles + 100),
            context.with_metadata(resampled=False),
            {"call": self.calls},
        )

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


def test_default_budget_and_override_are_independent(context_factory, rng):
    child = mutation(n_steps=2)
    block = RepeatedMutationStep(child, n_rounds=3)
    state = ParticleState(np.array([[-1.0], [1.0]]))

    default_state, _, default = block(state, context_factory(), rng)
    override_state, _, override = block.apply_with_n_steps(
        state, context_factory(), rng, n_steps=4
    )
    replay_state, _, replay = block(state, context_factory(), rng)

    assert block.n_steps == 6
    assert child.n_steps == 2
    assert [output["steps"] for output in default.round_outputs] == [2, 2, 2]
    assert [output["steps"] for output in override.round_outputs] == [2, 1, 1]
    assert default.after_round_outputs == (None, None, None)
    assert default.metrics.n_steps == replay.metrics.n_steps == 6
    np.testing.assert_array_equal(
        default_state.particles, replay_state.particles
    )
    np.testing.assert_array_equal(
        override_state.particles, state.particles + 4
    )


@pytest.mark.parametrize("n_steps", [0, 2, 5])
def test_after_round_runs_after_every_round_with_one_conditional_decision(
    n_steps, context_factory, rng
):
    after = ShiftAfterRound()
    block = RepeatedMutationStep(
        mutation(),
        n_rounds=3,
        n_steps=n_steps,
        after_round=SequentialStep([after]),
    )
    wrapper = MutateIfResampled(block)
    state = ParticleState(np.array([[-1.0], [1.0]]))
    context = context_factory().with_metadata(resampled=True)

    updated, context, wrapped_output = wrapper(state, context, rng)
    output = wrapped_output.child_output

    assert wrapped_output.executed
    assert after.calls == 3
    assert not context.metadata["resampled"]
    assert [result[0]["call"] for result in output.after_round_outputs] == [
        1,
        2,
        3,
    ]
    assert output.metrics.n_steps == n_steps
    assert output.metrics.n_attempted == 2 * n_steps
    np.testing.assert_array_equal(
        updated.particles, state.particles + n_steps + 300
    )


@pytest.mark.parametrize("n_rounds", [0, -1, True, 1.5])
def test_invalid_round_count(n_rounds):
    with pytest.raises((TypeError, ValueError)):
        RepeatedMutationStep(mutation(), n_rounds=n_rounds)


@pytest.mark.parametrize("n_steps", [-1, True, 1.5])
def test_invalid_default_budget(n_steps):
    with pytest.raises((TypeError, ValueError)):
        RepeatedMutationStep(mutation(), n_steps=n_steps)


def test_invalid_after_round():
    with pytest.raises(TypeError, match="after_round"):
        RepeatedMutationStep(
            mutation(), after_round=lambda s, c, r: (s, c, None)
        )


def test_checkpoint_restores_child_and_after_round(context_factory, rng):
    class RecordingMutation(MetropolisHastingsMutation):
        def __init__(self):
            super().__init__(
                lambda p, c: np.zeros(p.shape[0]),
                CallableMutationKernel(lambda p, r: p + 1, symmetric=True),
            )
            self.calls = 0

        def apply_with_mutation_state(
            self, state, context, rng, *, n_steps, mutation_state=None
        ):
            self.calls += 1
            return super().apply_with_mutation_state(
                state,
                context,
                rng,
                n_steps=n_steps,
                mutation_state=mutation_state,
            )

        def state_dict(self):
            return {"calls": self.calls}

        def load_state_dict(self, state):
            self.calls = state["calls"]

    child, after = RecordingMutation(), ShiftAfterRound()
    block = RepeatedMutationStep(child, n_rounds=2, after_round=after)
    state = ParticleState(np.array([[-1.0], [1.0]]))
    block(state, context_factory(), rng)
    checkpoint = block.state_dict()
    restored_child, restored_after = RecordingMutation(), ShiftAfterRound()
    restored = RepeatedMutationStep(
        restored_child, n_rounds=2, after_round=restored_after
    )
    restored.load_state_dict(checkpoint)
    assert restored_child.calls == restored_after.calls == 2
    _, _, output = restored(state, context_factory(), rng)
    assert restored_child.calls == restored_after.calls == 4
    assert output.metrics.n_steps == 2


def test_blackjax_block_functional_transform_and_after_round():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from dataclasses import dataclass

    from sequince.steps.context import StepContext
    from sequince.steps.mutation import BlackJAXMutation
    from sequince.transforms import ParticleTransform
    from sequince.utils.pytrees import pytree_dataclass

    @pytree_dataclass(data_fields=())
    @dataclass(frozen=True)
    class ShiftTransform(ParticleTransform):
        def forward(self, particles, context=None, *, rng=None):
            return particles - 10, jnp.zeros(particles.shape[0]), None

        def inverse(self, values, context=None, *, auxiliary=None, rng=None):
            return values + 10, jnp.zeros_like(values[..., 0])

    class Reflect(SMCStep):
        def apply(self, state, context, rng):
            return state.replace_particles(20 - state.particles), context, None

    block = RepeatedMutationStep(
        BlackJAXMutation(
            lambda p, c: -0.5 * jnp.sum((p - 10) ** 2),
            algorithm="random-walk",
            algorithm_kwargs={"sigma": 0.1},
            transform=ShiftTransform(),
            store_chains=True,
        ),
        n_rounds=2,
        n_steps=4,
        after_round=Reflect(),
    )
    state = ParticleState(jnp.array([[9.0], [11.0]]))
    state, context = block.initialize_functional_state_and_context(
        state, StepContext(t=0)
    )
    _, _, _, output = jax.jit(block.apply_functional)(
        state, context, jax.random.key(42)
    )
    assert output.metrics.n_steps == 4
    assert output.metrics.n_attempted == 8
    assert all(child.chain.shape[0] == 2 for child in output.round_outputs)


@pytest.mark.parametrize("n_steps", [0, 5])
def test_proposal_scale_adapts_after_complete_block(
    n_steps, context_factory, rng
):
    from sequince.steps import AdaptiveGaussianProposal

    class RecordScale(SMCStep):
        def __init__(self):
            self.scales = []

        def apply(self, state, context, rng):
            self.scales.append(context.metadata["proposal_scale"])
            return state, context, None

    after = RecordScale()
    block = RepeatedMutationStep(mutation(), n_rounds=3, after_round=after)
    wrapper = AdaptiveGaussianProposal(
        block, initial_scale=0.5, adaptation_rate=0.2
    )
    state = ParticleState(np.array([[-1.0], [1.0]]))
    original_context = context_factory()
    updated, context, output = wrapper.apply_with_n_steps(
        state, original_context, rng, n_steps=n_steps
    )
    assert after.scales == [0.5] * 3
    assert output.metrics.n_steps == n_steps
    assert "proposal_scale" not in original_context.metadata
    expected = 0.5 * np.exp(0.2 * 0.7) if n_steps else 0.5
    assert context.metadata["proposal_scale"] == pytest.approx(expected)
    _, _, next_output = wrapper(updated, context, rng)
    assert after.scales[3:] == pytest.approx([expected] * 3)
    assert next_output.metrics.n_steps == block.n_steps


def test_proposal_adaptation_preserves_nested_checkpoint(context_factory, rng):
    from sequince.steps import AdaptiveGaussianProposal

    after = ShiftAfterRound()
    wrapper = AdaptiveGaussianProposal(
        RepeatedMutationStep(mutation(), n_rounds=2, after_round=after)
    )
    state = ParticleState(np.array([[-1.0], [1.0]]))
    wrapper(state, context_factory(), rng)
    saved = wrapper.state_dict()
    after.calls = 0
    wrapper.load_state_dict(saved)
    assert after.calls == 2


def test_minipcn_rounds_match_uninterrupted_adaptation(context_factory):
    from minipcn import Sampler
    from orng import RandomGenerator

    from sequince.steps.mutation import MiniPCNMutation

    particles = np.random.default_rng(12).normal(size=(64, 2))

    def target(particles):
        return -0.5 * np.sum(particles**2, axis=-1)

    child = MiniPCNMutation(target, step_fn="pcn", adaptive=True)
    block = RepeatedMutationStep(child, n_rounds=3, n_steps=5)
    updated, _, output = block(
        ParticleState(particles), context_factory(), RandomGenerator(seed=13)
    )
    expected, history = Sampler(target, "pcn", dims=2).sample(
        particles, 5, seed=13, verbose=False, return_last_only=True
    )
    np.testing.assert_array_equal(updated.particles, expected)
    np.testing.assert_array_equal(
        np.concatenate([o.acceptance_by_step for o in output.round_outputs]),
        history.acceptance_rate,
    )
    assert output.metrics.n_steps == 5


@pytest.mark.parametrize("refit,expected_fits", [("once", 1), ("always", 3)])
def test_minipcn_refits_only_when_transform_changes(
    refit, expected_fits, context_factory, monkeypatch
):
    from minipcn.step import PCNStep

    from sequince.steps.mutation import MiniPCNMutation
    from sequince.transforms import AffineTransform

    fits = []
    original = PCNStep.init_state

    def fit(step, particles):
        fits.append(np.array(particles, copy=True))
        return original(step, particles)

    monkeypatch.setattr(PCNStep, "init_state", fit)
    child = MiniPCNMutation(
        lambda p: -0.5 * np.sum(p**2, axis=-1),
        step_fn="pcn",
        transform=AffineTransform(),
        transform_refit=refit,
    )
    block = RepeatedMutationStep(
        child, n_rounds=3, after_round=ShiftAfterRound()
    )
    state = ParticleState(np.random.default_rng(14).normal(size=(64, 2)))
    updated, _, output = block(state, context_factory(), 15)
    assert len(fits) == expected_fits
    assert output.metrics.n_steps == 3
    # Each new block fits again, even if the transform is reused.
    block(updated, context_factory(), 16)
    assert len(fits) == 2 * expected_fits
