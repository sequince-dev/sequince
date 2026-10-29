Pre-built samplers
==================

``sequince`` provides three helpers for assembling Bayesian SMC pipelines.
Each returns a runner, its target, and initial metadata. Supply your own
initial :class:`~sequince.state.ParticleState` and pass the returned metadata
to ``runner.run(..., metadata=metadata)``.

Standard Bayesian SMC
---------------------

:func:`~sequince.prebuilt_samplers.make_standard_bayesian_smc` moves a
particle population through a sequence of tempered distributions, combining
weighting, resampling, and mutation. It uses adaptive temperatures by default;
``adaptive=False`` with ``num_steps`` selects a fixed temperature schedule.
The usual posterior endpoint is ``max_beta=1.0``. For the SMC sampler
framework, see `Del Moral, Doucet, and Jasra (2006)
<https://doi.org/10.1111/j.1467-9868.2006.00553.x>`_.

This is the helper used in the :ref:`quick start <quick-start>`. See
:doc:`smc_configuration` for configuration and manual pipeline assembly.

Persistent Bayesian SMC
-----------------------

Persistent sampling was introduced by `Karamanis and Seljak (2025)
<https://doi.org/10.1007/s11222-025-10682-y>`_.

:func:`~sequince.prebuilt_samplers.make_persistent_bayesian_smc` retains an
archive of previous particle batches and reweights that archive as the
temperature changes. ``n_particles`` sets the active batch size;
``target_ess`` and ``target_total_ess`` are absolute effective sample sizes
of the archive.

The initial population must contain a positive multiple of ``n_particles``
equally weighted draws from the prior, or from ``log_proposal`` when supplied.
By default, the sampler finishes with an equally weighted active population.
Set ``resample_at_end=False`` to return the complete weighted archive instead.

Nested-sampling SMC
-------------------

Nested sampling via SMC (NS-SMC) follows the framework of
`Salomone et al. (2025) <https://doi.org/10.1093/jrsssb/qkaf015>`_.
The paper distinguishes fixed-threshold NS-SMC from adaptive NS-SMC (ANS-SMC).
The helper uses adaptive thresholds by default; these do not inherit the
fixed-schedule unbiasedness guarantee.

:func:`~sequince.prebuilt_samplers.make_nested_sampling_smc` raises a
likelihood threshold and explores the remaining constrained prior. It
accumulates evidence over the discarded likelihood shells and the final
live population. Initial particles must represent a normalized prior;
arbitrary importance proposals are not supported.

Use ``threshold_quantile`` or ``threshold_schedule`` to control the thresholds.
``evidence_tolerance`` controls an estimated remaining-evidence stopping
criterion; it is not a guaranteed error bound.

Read evidence and reconstruct posterior samples through
``result.as_nested_sampling_result()``. Posterior reconstruction requires a
completed run and ``store_shell_particles=True`` (the default). See
:doc:`results` for the corresponding result interface.

All three helpers expose mutation, resampling, and particle-transform options;
see :doc:`../sampling_options/index` and their API references for supported settings.
For standalone MCMC, see :doc:`../advanced_workflows/mcmc_sampler`.

Waste-free SMC
--------------

Waste-free SMC, introduced by `Dau and Chopin (2022)
<https://doi.org/10.1111/rssb.12475>`_, retains intermediate MCMC states within
each stage. Assemble it in a custom pipeline with
:class:`~sequince.steps.waste_free.WasteFreeResampleMove`; see the
`worked tutorial <https://sequince.readthedocs.io/projects/sequince-examples/en/latest/_generated/examples/waste_free_smc/README.html>`_.
Persistent sampling instead reuses populations across stages.
Full method references are collected in :doc:`../../citation`.
