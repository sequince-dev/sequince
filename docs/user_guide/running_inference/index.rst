Running inference
=================

This section explains how to choose a pre-built sampler, configure a run,
and interpret its results and evidence estimate (normalizing constant).
Custom pipelines can also be assembled from composable steps.

Configuring inference
---------------------

Start with the available pre-built samplers, then configure the SMC runner.
For sequential data or particle filtering, see :doc:`observations` for
passing an observation stream to the runner.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   prebuilt_samplers
   smc_configuration
   callbacks
   observations

Understanding outputs
---------------------

After a run, inspect the particle population, extract posterior samples, and
check diagnostics. The evidence guide explains the normalizing constant and
the assumptions behind its uncertainty estimates.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   results
   evidence_uncertainty
