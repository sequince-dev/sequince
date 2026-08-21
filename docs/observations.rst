Using observations
==================

Pass an iterable to ``observations`` when an SMC iteration should process one
item from a data stream. The runner consumes at most one item at the beginning
of each iteration and exposes it as
:attr:`~sequince.steps.context.StepContext.observation`. Every step and callback
in that iteration sees the same observation.

An observation can be any object: a scalar measurement, an array, a batch, or a
structured value containing several fields. Sequince does not interpret it; the
configured steps or target functions decide how it affects the particle state.

A sequential likelihood update
------------------------------

For particle filtering or online Bayesian updates, an incremental weighting
step can evaluate the likelihood contribution from the current observation:

.. code-block:: python

    import numpy as np

    from sequince import ParticleState, SequentialMonteCarlo
    from sequince.steps import LogLikelihoodWeighting, SystematicResampling


    def log_likelihood_increment(particles, context):
        observation = context.observation
        variance = context.shared["observation_variance"]
        residual = observation - particles[:, 0]
        return -0.5 * residual**2 / variance


    smc = SequentialMonteCarlo(
        steps=[
            LogLikelihoodWeighting(
                log_likelihood_increment,
                normalize_after=False,
            ),
            SystematicResampling(ess_threshold=0.5),
        ]
    )

    initial_state = ParticleState(
        particles=np.random.default_rng(42).normal(size=(500, 1))
    )
    result = smc.run(
        initial_state,
        observations=[0.2, -0.1, 0.4],
        rng=42,
        shared_context={"observation_variance": 1.0},
    )

This run produces three states: ``result.states[i]`` is the population after
processing ``observations[i]``. The state before any observations were processed
is available as ``result.initial_state``.

Each item is one iteration
--------------------------

``observations`` defines iteration input, not an automatically vectorized data
axis. A list containing three arrays causes three iterations; each array is the
complete ``context.observation`` for its iteration. All configured steps run
once for each consumed item.

The interaction between ``observations`` and ``num_steps`` is important:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - ``observations``
     - ``num_steps``
     - Behavior
   * - Finite iterable
     - ``None``
     - Run until the iterable is exhausted.
   * - Finite iterable
     - Positive integer ``N``
     - Run at most ``N`` iterations. If the iterable ends first, later
       iterations receive ``observation=None``.
   * - Infinite iterable
     - Positive integer ``N``
     - Consume the first ``N`` items and stop.
   * - ``None``
     - Positive integer ``N``
     - Run ``N`` iterations, each with ``observation=None``.
   * - ``None`` or infinite iterable
     - ``None``
     - Continue until a step or callback requests a stop.

A stop request can always end a run before either limit. Conversely, passing
neither a finite observation stream nor ``num_steps`` requires the pipeline to
request its own stop; otherwise the run has no natural bound.

A run must complete at least one iteration. Consequently, an empty observation
iterable with no ``num_steps`` produces no state and raises ``ValueError``.

Iterables and generators
------------------------

The runner consumes a non-sequence iterable lazily, so a generator can represent
a stream without first loading it into memory:

.. code-block:: python

    def measurements(source):
        for record in source:
            yield record.value


    result = smc.run(
        initial_state,
        observations=measurements(source),
        rng=rng,
    )

If ``num_steps`` is omitted, normal generator exhaustion ends the run. If it is
specified, exhaustion instead supplies ``None`` through the remaining requested
iterations. A step that permits ``None`` observations should handle that case
explicitly.

Observation position
--------------------

``context.t`` is the zero-based SMC iteration. It is not necessarily an
observation index because iterations may continue after the input is exhausted.
``context.next_observation_index`` counts observations already consumed and is
the position from which a resumable sequence will continue. During the
iteration processing ``observations[i]``, it is already ``i + 1``.

Use ``context.observation`` to read the current item. The next-index field is
primarily runtime and checkpoint state; it should not normally be used to index
the current observation.

Checkpointing and resuming
--------------------------

Checkpoints store ``next_observation_index`` but do not store the full input
stream. When resuming a run that has already consumed observations, pass the
original indexable sequence to ``run``:

.. code-block:: python

    runtime = smc.resume_from(
        checkpoint,
        shared_context=shared_context,
    )
    result = smc.run(
        runtime=runtime,
        observations=all_observations,
    )

The runner resumes with ``all_observations[checkpoint.next_observation_index]``.
A one-shot iterator cannot be repositioned and is rejected after any
observations have already been consumed. See :doc:`checkpointing` for the full
checkpoint workflow.

JAX observations
----------------

``run_functional`` has the same host-side observation and termination semantics.
When using a compiled iteration, all observations must have a pytree structure,
array shapes, and dtypes compatible with the compiled signature. Supply a
representative first observation to the compilation helper:

.. code-block:: python

    iteration = compile_functional_iteration(
        smc,
        initial_state,
        key,
        observation=observations[0],
        next_observation_index=1,
        shared_context=shared_context,
        metadata=metadata,
    )
    result, key = smc.run_functional(
        initial_state,
        observations=observations,
        random_state=key,
        shared_context=shared_context,
        metadata=metadata,
        iteration=iteration,
    )

See :doc:`jax_support` for compilation, random-state threading, and fixed-carry
requirements.
