Resampling configuration
========================

Resampling draws particles from a weighted population and assigns equal
weights to the result. Some particles may be duplicated; mutation can then
move them to new positions.

.. list-table::
   :header-rows: 1
   :widths: 18 20 30 32

   * - Method
     - ``resampling`` name
     - Implementation
     - Ancestor selection
   * - Systematic
     - ``"systematic"``
     - :class:`~sequince.steps.resampling.SystematicResampling`
     - Equally spaced positions with one random offset.
   * - Stratified
     - ``"stratified"``
     - :class:`~sequince.steps.resampling.StratifiedResampling`
     - One random position in each equal-width stratum.
   * - Multinomial
     - ``"multinomial"``
     - :class:`~sequince.steps.resampling.MultinomialResampling`
     - Independent draws according to particle weights.

Common configuration
--------------------

The method determines how ancestors are drawn. Constructor options control
when to resample and how many particles to draw:

.. code-block:: python

    from sequince.steps.resampling import StratifiedResampling

    resampling_step = StratifiedResampling(
        ess_threshold=0.4,
        always_resample=False,
        resample_at_end=True,
    )

All three methods accept:

* ``ess_threshold``: resample when ESS divided by the input particle count
  falls below this value (default: ``0.5``). ``None`` disables the ESS trigger.
* ``always_resample``: resample on every call, regardless of ESS
  (default: ``False``).
* ``resample_at_end``: resample when ``context.stop_requested`` is set,
  regardless of ESS (default: ``True``).
* ``n_samples``: number of output particles when resampling occurs. ``None``
  (default) keeps the input count. Skipped resampling leaves the count unchanged.

To disable resampling, set ``ess_threshold=None``, ``always_resample=False``,
and ``resample_at_end=False``.

Functional execution currently requires ``always_resample=True``. Compiled
pipelines must also preserve the particle count; see :doc:`../advanced_workflows/jax_support`.

Outputs and ancestry
--------------------

The step returns :class:`~sequince.steps.resampling.ResamplingOutput` with
``ess`` (before resampling), ``resampled``, ``ancestors``, and
``resampling_scheme``. ``ancestors`` indexes the input population; when
resampling is skipped, it contains the identity indices. The context metadata
also records ``ess`` and ``resampled``.

The state tracks each particle's original ancestor in
``initial_ancestor_indices``. Resampling leaves the accumulated normalizing
constant unchanged. See :doc:`../running_inference/evidence_uncertainty` for the assumptions needed
by genealogy-based evidence estimators, including multinomial resampling.

Prebuilt samplers
-----------------

:func:`~sequince.prebuilt_samplers.make_standard_bayesian_smc` selects a method
with ``resampling`` (default: ``"systematic"``). Set the ESS threshold through
``resampling_ess_threshold`` (default: ``0.5``); ``always_resample`` and
``resample_at_end`` are also helper arguments. Pass extra constructor options,
such as ``n_samples``, through ``resampling_kwargs``.

With ``weighting_criterion="ess"`` (default), the helper resamples every
iteration and rejects ``always_resample=False``. With
``weighting_criterion="cess"``, it defaults to ESS-triggered resampling.
Terminal resampling remains enabled unless ``resample_at_end=False``.
