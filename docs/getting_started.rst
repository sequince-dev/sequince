Getting started
===============

Installation
------------

Install from PyPI::

    pip install sequince

``sequince`` targets Python 3.11+. Its core dependencies are installed
automatically; integrations such as emcee, BlackJAX, and HDF5 checkpointing are
available as optional extras.

Quick start
-----------

Below, we show a short example of how to use ``sequince``.
For more details, see the :doc:`user_guide`, and see
`sequince-examples <https://github.com/sequince-dev/sequince-examples>`_
for more advanced usage.
Here, we fit a straight line, ``y = slope * x + intercept``, to synthetic data with
independent Gaussian measurement errors and uniform priors on both parameters.
The log-prior and log-likelihood callbacks accept particles shaped
``(n_particles, n_parameters)`` and return one log density per particle.

.. literalinclude:: ../README.md
   :language: python
   :start-after: ```python
   :end-before: ```

Next steps
----------

See :doc:`evidence_uncertainty` for interpreting the evidence estimate and its
uncertainty. To customise the run, continue with :doc:`smc_configuration` and
the sampling choices in :doc:`resampling_options` and :doc:`mutation_options`.
The :doc:`step_context` guide explains the distinction between shared inputs
and evolving metadata. For sampling without SMC weighting and resampling, see
:doc:`mcmc_sampler`.
