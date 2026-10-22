Getting started
===============

Installation
------------

Install from PyPI::

    pip install sequince

``sequince`` targets Python 3.11+. Its core dependencies are installed
automatically. The example below uses the included MiniPCN mutation backend.
See :doc:`installation` for optional integrations and
:doc:`user_guide/core_concepts/array_backends` for backend selection and the
``SCIPY_ARRAY_API`` setting for SciPy dispatch.

Quick start
-----------

Below, we show a short example of how to use ``sequince``.
For more details, see the :doc:`user_guide/index`, and see
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

Continue with :doc:`user_guide/running_inference/results` to understand the
returned result, extract posterior samples, and inspect diagnostics. See
:doc:`user_guide/running_inference/evidence_uncertainty` for interpreting the
evidence estimate and its uncertainty.

To customise the run, see
:doc:`user_guide/running_inference/smc_configuration` and the sampling choices
in :doc:`user_guide/sampling_options/resampling_configuration` and
:doc:`user_guide/sampling_options/mutation_configuration`.
The :doc:`user_guide/core_concepts/step_context` guide explains the distinction
between shared inputs and evolving metadata.
