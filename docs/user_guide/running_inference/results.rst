Working with results
====================

The SMC runner returns an :class:`~sequince.result.SMCResult`. It contains
the state after each completed iteration in ``result.states``, the starting
population in ``result.initial_state``, and step outputs in ``result.outputs``.
``result.final_state`` is the final population; ``result[i]`` accesses an
earlier population. See :doc:`../core_concepts/particle_state` for the particle-state interface.

Posterior samples and summaries
-------------------------------

For standard and persistent Bayesian SMC runs that have reached the posterior
target, use the final population to compute weighted summaries:

.. code-block:: python

    mean = result.weighted_mean()
    covariance = result.weighted_covariance()
    samples, weights = result.posterior_samples()

The default ``method="weights"`` returns the final particles and their
normalized weights. To draw an equally weighted sample with replacement:

.. code-block:: python

    samples, weights = result.posterior_samples(
        n_samples=1_000,
        method="multinomial",
        rng=42,
    )

Respect the weights when analysing the particles directly. Intermediate
tempered populations do not yet represent the full posterior, and
resampling does not make particles independent.

Run diagnostics
---------------

Inspect the run summary and histories to understand how the population evolved:

.. code-block:: python

    summary = result.summary()
    ess = result.effective_sample_sizes()
    beta = result.metadata_history("beta")
    log_evidence = result.log_evidence()
    log_evidence_history = result.log_evidence_history()

The ``beta`` history is relevant to tempered runs. ESS describes weight
concentration; it does not measure the number of independent posterior draws.
Step-specific diagnostics are available in ``result.outputs``.
See :doc:`evidence_uncertainty` for interpreting evidence uncertainty.

Nested-sampling results
-----------------------

For nested-sampling SMC, use the shell-weighted result interface:

.. code-block:: python

    nested = result.as_nested_sampling_result()
    complete = nested.complete
    log_evidence = nested.log_evidence
    # Once complete, with store_shell_particles=True:
    samples, weights = nested.posterior_samples()

The final state represents the constrained prior. Bayesian evidence and
posterior weights come from the accumulated shells, so use this interface
for posterior analysis. A run suspended by ``num_steps`` must be resumed to
completion before extracting posterior samples.
