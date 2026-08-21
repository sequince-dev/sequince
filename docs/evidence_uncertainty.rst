Evidence and uncertainty
========================

:class:`~sequince.result.SMCResult` provides three estimates of log-evidence
uncertainty from a single SMC run:

.. list-table::
   :header-rows: 1

   * - Method
     - Use
     - Main limitation
   * - ``"incremental"``
     - Default; uses each incremental weight update
     - Ignores dependence between SMC iterations
   * - ``"lee-whiteley"``
     - Uses the complete particle genealogy
     - Requires multinomial resampling at every iteration
   * - ``"fixed-lag-lee-whiteley"``
     - Uses only recent particle ancestry
     - Trades genealogical stability for truncation bias

Usage
-----

``log_evidence_variance`` estimates
:math:`\operatorname{Var}(\log \widehat Z)`. ``log_evidence_error`` returns the
corresponding standard error:

.. code-block:: python

   log_evidence = result.log_evidence()
   variance = result.log_evidence_variance()
   error = result.log_evidence_error()

   lw_variance = result.log_evidence_variance(method="lee-whiteley")

   fixed_lag_variance = result.log_evidence_variance(
       method="fixed-lag-lee-whiteley",
       lag=10,
   )

These values describe Monte Carlo error in the SMC evidence estimate, not
uncertainty in the statistical model or posterior.

Assumptions and limitations
---------------------------

``incremental``
   This estimator is inexpensive, requires no ancestry, and supports adaptive
   temperatures and adaptive resampling. It ignores dependence between
   successive SMC iterations and may therefore underestimate uncertainty.

   It requires ``log_evidence_increment_variance`` in every particle state's
   metadata. This is recorded by
   :class:`~sequince.steps.weighting.FixedTemperedWeighting` and
   :class:`~sequince.steps.weighting.AdaptiveTemperedWeighting`; missing values
   produce ``nan``. The correct particle-count scaling is already present in
   the squared normalised weights, so no extra :math:`N` or :math:`N-1` factor
   is needed.

``lee-whiteley``
   This is the full-genealogy estimator described by `Lee and Whiteley (2018)
   <https://doi.org/10.1093/biomet/asy028>`_. It requires at least two
   particles, a constant particle count, exactly one resampling step per
   iteration, resampling at every iteration, and multinomial resampling.

   ``sequince`` records the scheme used by its built-in resampling steps. The
   estimator warns when it finds systematic, stratified, missing, or unknown
   scheme information. A custom resampling step should set its
   ``resampling_scheme`` class attribute. The calculation still returns a value
   after a warning, but it does not then have the stated theoretical
   justification. The estimator also becomes unreliable as the time-zero
   ancestor families collapse.

``fixed-lag-lee-whiteley``
   This estimator uses only the most recent resampling events. Short lags retain
   more ancestor families but omit older variance contributions; long lags
   approach the full-genealogy estimator and its degeneracy.

   ``lag`` counts resampling events, not temperature updates or SMC iterations.
   Skipped adaptive-resampling events do not consume lag. The constant particle
   count and multinomial-resampling assumptions still apply, and each iteration
   must contain exactly one resampling-step output. Mutation should not mix
   particle paths.

   Recent-ancestor variance estimation is discussed by `Olsson and Douc (2019)
   <https://doi.org/10.3150/18-BEJ1028>`_. Their estimator targets normalised
   particle expectations rather than evidence. The method provided here is an
   evidence-specific fixed-lag Lee--Whiteley estimator.

Diagnostics
-----------

Inspect the selected genealogy before relying on a fixed-lag estimate:

.. code-block:: python

   diagnostics = result.fixed_lag_lee_whiteley_diagnostics(lag=10)
   print(diagnostics.traced_resampling_events)
   print(diagnostics.distinct_ancestors)
   print(diagnostics.family_ess)

The diagnostics report the requested and traced lag, total resampling events,
distinct selected-lag ancestors, and family ESS. A family ESS near one indicates
severe genealogical depletion. Comparing several plausible lags is generally
more informative than reporting one arbitrary choice.

Finite-sample Lee--Whiteley estimates can be negative. The variance method
preserves such values because they signal an unreliable estimate;
``log_evidence_error`` raises ``ValueError`` because no real-valued standard
error can then be computed.

Repeated independent SMC runs remain the most direct uncertainty check when
they are affordable.
