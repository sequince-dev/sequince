User guide
==========

The Sequential Monte Carlo (SMC) loop in ``sequince`` is built from a set of
composable *steps*. Each step inspects the current :class:`~sequince.state.ParticleState`,
optionally consumes observations and shared configuration, and returns an updated
state. The :class:`~sequince.smc.SequentialMonteCarlo` class drives the loop by
executing the configured steps at every iteration.

Configuring the runner
----------------------

The runner accepts a ``steps`` sequence containing callables that conform to the
``SMCStep`` protocol defined in :mod:`sequince.steps.base`. The quickest way to
build a useful pipeline is to reuse the pre-built helper:

.. code-block:: python

    from sequince import make_standard_bayesian_smc

    smc, posterior, shared_context = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        mutation="emcee",
    )

``make_standard_bayesian_smc`` wires together
:class:`~sequince.steps.weighting.AdaptiveTemperedWeighting`,
:class:`~sequince.steps.resampling.SystematicResampling`, and a mutation step
chosen via the ``mutation`` argument. Pass ``mutation="minipcn"`` to opt into
the :class:`~sequince.steps.mutation.MiniPCNMutation` transition kernel or
``mutation="none"`` to skip rejuvenation.

If you need full control, construct ``SequentialMonteCarlo`` manually by
instantiating step objects and supplying them in the desired order:

.. code-block:: python

    from sequince import SequentialMonteCarlo
    from sequince.steps import (
        AdaptiveTemperedWeighting,
        SystematicResampling,
        EmceeMutation,
    )

    smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(posterior, target_ess_ratio=0.8),
            SystematicResampling(always_resample=True),
            EmceeMutation(posterior, n_steps=10),
        ]
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

Custom steps
------------

To implement bespoke logic, derive from
:class:`~sequince.steps.base.SMCStep`. Override the ``__call__`` method to
perform work and return the new state. The base class offers utilities for
logging, stopping the overall run, and accessing metadata stored on the context,
including the mutable ``StepContext.shared`` dictionary for cross-step state.

See :mod:`sequince.steps` for a catalogue of the built-in components.
