Resampling options
==================

Resampling replaces a weighted particle population with equally weighted
particles drawn from it. This prevents a small number of particles from
carrying almost all of the weight, at the cost of duplicating some particles.
Mutation is commonly applied afterwards to restore diversity.

Choosing a strategy
-------------------

The :func:`~sequince.prebuilt_samplers.make_standard_bayesian_smc` helper
selects the ancestor-sampling algorithm with ``resampling``:

``"systematic"``
    Use :class:`~sequince.steps.resampling.SystematicResampling`. This is the
    default and uses one random offset for the complete population.

``"stratified"``
    Use :class:`~sequince.steps.resampling.StratifiedResampling`, with one
    random position in each equal-width stratum.

``"multinomial"``
    Use :class:`~sequince.steps.resampling.MultinomialResampling`, drawing each
    ancestor independently according to the particle weights.

For example:

.. code-block:: python

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        resampling="stratified",
    )

Choosing when to resample
-------------------------

The resampling algorithm and its schedule are configured separately:

.. code-block:: python

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        weighting_criterion="cess",
        resampling="stratified",
        resampling_ess_threshold=0.4,
        always_resample=False,
    )

With ``always_resample=False``, resampling occurs when the effective sample size
divided by the number of particles falls below ``resampling_ess_threshold``.
Set the threshold to ``None`` to disable threshold-triggered resampling. With
``always_resample=True``, the runner resamples every iteration and ignores the
threshold.

When ``always_resample`` is omitted, its default depends on
``weighting_criterion``: ESS-based tempering resamples every iteration, while
CESS-based tempering uses the ESS threshold. Setting ``always_resample=False``
with ``weighting_criterion="ess"`` is invalid because ESS-based tempering relies
on resampling to meet its target.

Functional execution currently requires ``always_resample=True``. See
:doc:`jax_support` for the other functional-execution constraints.

Configuring a resampling step directly
---------------------------------------

In a manually assembled pipeline, instantiate the selected step with the same
schedule arguments:

.. code-block:: python

    from sequince.steps import StratifiedResampling

    resampling = StratifiedResampling(
        ess_threshold=0.4,
        always_resample=False,
    )

The step records the pre-resampling ESS and whether resampling occurred in both
the context metadata and its output. The particle state records selected
ancestor indices when resampling occurs, which allows particle genealogies to
be reconstructed.

Resampling and evidence estimates
---------------------------------

The resampling strategy affects Monte Carlo variability even though all three
built-in algorithms target the same weighted population. See
:doc:`evidence_uncertainty` for the uncertainty estimators and their resampling
requirements.
