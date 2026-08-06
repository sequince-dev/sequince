User guide
==========

The Sequential Monte Carlo (SMC) loop in ``sequince`` is built from a set of
composable *steps*. Each step inspects the current :class:`~sequince.state.ParticleState`,
optionally consumes observations and shared configuration, and returns an updated
state. At the end of each loop iteration, a set of *callbacks* are executed
that can perform operations like logging, diagnostics, and plotting.
The :class:`~sequince.smc.SequentialMonteCarlo` class drives the loop by
sequentially executing the configured steps and callbacks at every iteration.

Configuring the runner
----------------------

The runner accepts a ``steps`` sequence containing callables that conform to the
``SMCStep`` protocol defined in :mod:`sequince.steps.base`. Similarly, the runner
accepts a sequence of optional ``callbacks`` that adhere to the
``SMCCallback`` protocol in :mod:`sequince.callbacks`.
The quickest way to build a useful pipeline is to reuse the pre-built helper:

.. code-block:: python

    from sequince import make_standard_bayesian_smc
    from sequince.callbackss import TracePlot
    smc, posterior, shared_context = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        mutation="emcee",
        callbacks = [TracePlot()],  # Optional
    )

``make_standard_bayesian_smc`` wires together
:class:`~sequince.steps.weighting.AdaptiveTemperedWeighting`,
:class:`~sequince.steps.resampling.SystematicResampling`, and a mutation step
chosen via the ``mutation`` argument. Pass ``mutation="minipcn"`` to opt into
the :class:`~sequince.steps.mutation.MiniPCNMutation` transition kernel or
``mutation="none"`` to skip rejuvenation.

BlackJAX mutation
-----------------

Install the optional BlackJAX integration with ``uv add "sequince[blackjax]"``.
BlackJAX mutation requires JAX-backed particles and JAX-compatible target
functions. NUTS is the default algorithm:

.. code-block:: python

    import jax.numpy as jnp

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        mutation="blackjax",
        mutation_n_steps=20,
        mutation_kwargs={
            "algorithm": "nuts",
            "algorithm_kwargs": {
                "step_size": 0.1,
                "inverse_mass_matrix": jnp.ones(n_dimensions),
            },
        },
    )

``algorithm`` may also be ``"hmc"``, ``"random-walk"``, or a callable that
constructs a BlackJAX sampling algorithm from the current scalar log-density.
Acceptance and divergence diagnostics are stored in particle-state metadata as
``blackjax_acceptance`` and ``blackjax_divergence``.

If you need full control, construct ``SequentialMonteCarlo`` manually by
instantiating step and callback objects and supplying them in the desired order:

.. code-block:: python

    from sequince import SequentialMonteCarlo
    from sequince.steps import (
        AdaptiveTemperedWeighting,
        SystematicResampling,
        EmceeMutation,
    )
    from sequince.callbacks import TracePlot

    smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(posterior, target_ess_ratio=0.8),
            SystematicResampling(always_resample=True),
            EmceeMutation(posterior, n_steps=10),
        ],
        callbacks=[TracePlot()],  # Optional
    )

Working with particle state
---------------------------

:class:`~sequince.state.ParticleState` stores the particle locations,
log-weights, accumulated log-evidence, and auxiliary metadata. A few convenience
methods are helpful when implementing new steps:

``ParticleState.normalize()``
    Returns a fresh state with renormalised weights while keeping the original
    untouched.

``ParticleState.resample(indices)``
    Constructs a new state from ancestor indices, resetting the weights to a
    uniform distribution.

``ParticleState.with_updates(**kwargs)``
    Produces a copy with selected fields replaced.

When possible, treat the state as immutable and return a new instance instead of
mutating in place. This keeps the iteration trace recorded in
:class:`~sequince.result.SMCResults` consistent and easier to debug.

Incorporating observations
--------------------------

The runner accepts an optional iterable of observations. Each iteration pulls
the next observation (if available) and exposes it through the
:class:`~sequince.steps.base.StepContext` supplied to steps. This allows for
particle filtering style updates where likelihood evaluations depend on time
indexed data.

Custom steps and callbacks
--------------------------

To implement bespoke step logic, derive from
:class:`~sequince.steps.base.SMCStep`. Override the ``apply`` method to
perform work and return ``(state, context, output)``. ``StepContext.shared``
contains read-only user data, while current cross-step coordination values live
in ``StepContext.metadata``. Update metadata with
``context.with_metadata(...)`` and return the replacement context. Historical
diagnostics belong in the explicit step output; the runner stores ordered
per-iteration output tuples on :class:`~sequince.result.SMCResult`.

Similarly, custom callbacks can be implemented by sub-classing
:class:`~sequince.callbacks.SMCCallback` and overriding the ``apply`` method
Callbacks are provided with the current state, context, and result history. An
observational callback returns ``None``; a callback that updates metadata or
stop controls returns a replacement context.

See :mod:`sequince.steps` and :mod:`sequince.callbacks` for a catalogue of the built-in components.
