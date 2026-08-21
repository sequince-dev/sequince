Particle state and results
==========================

:class:`~sequince.state.ParticleState` stores the particle locations,
log-weights, accumulated log-evidence, and auxiliary metadata. The state passed
to a step represents the current particle population; the step returns the
state that should be passed onward.

When possible, treat a particle state as immutable and return a new instance.
This keeps the iteration trace recorded in
:class:`~sequince.result.SMCResult` consistent and easier to debug.

Updating state
--------------

The main copy-returning operations are:

``ParticleState.normalize()``
    Returns a state with normalized log-weights and the log-normalizer, while
    leaving the original state untouched.

``ParticleState.resample(indices)``
    Constructs a state from the selected ancestor indices and resets the
    log-weights to a uniform distribution.

``ParticleState.with_updates(**updates)``
    Returns a copy with the selected fields replaced.

For example, a step can replace the particles without mutating its input:

.. code-block:: python

    updated_state = state.with_updates(particles=proposed_particles)

State and context serve different purposes. Particle-specific sampling data
belongs in ``ParticleState``. Run-level configuration and values shared between
steps belong in :doc:`step_context`.

Result history
--------------

The runner records the state produced by each completed iteration in
:class:`~sequince.result.SMCResult`. ``result.final_state`` returns the final
population, while indexing the result accesses earlier populations. Step-level
diagnostics that do not describe particle state should be returned as the
step's explicit output instead; the runner stores these in ``result.outputs``.

Summarising a result
--------------------

The final state remains weighted unless the last configured step resampled it.
Use the result helpers to preserve those weights when computing summaries:

.. code-block:: python

    final_state = result.final_state
    mean = result.weighted_mean()
    covariance = result.weighted_covariance()
    samples = result.posterior_samples(1_000, rng=42)

``posterior_samples`` draws an equally weighted sample from the final weighted
population. For diagnostics across the complete run, use ``result.summary()``,
``result.effective_sample_sizes()``, ``result.log_evidence_history()``, and
``result.metadata_history(key)``. See :doc:`evidence_uncertainty` for
interpreting the final evidence estimate and its uncertainty.
