sequince documentation
======================

``sequince`` provides modular building blocks for assembling Sequential Monte
Carlo (SMC) algorithms for Bayesian inference, particle filtering, and related
sequential methods.

If you use ``sequince`` in your work, please follow our suggested :doc:`citation guidelines <citation>`. Thank you!

Installation
------------

Install from PyPI::

    pip install sequince

``sequince`` targets Python 3.11+. Its core dependencies are installed
automatically. The example below uses the included MiniPCN mutation backend,
see :doc:`installation` for all optional integrations supported.

.. _quick-start:

Quick start
-----------

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

* :doc:`user_guide/running_inference/results` explains the returned result,
  how to extract posterior samples, and how to inspect diagnostics;
  :doc:`user_guide/running_inference/evidence_uncertainty` covers the evidence
  estimate and its uncertainty.
* To customise the run, see :doc:`user_guide/running_inference/smc_configuration`,
  :doc:`user_guide/sampling_options/resampling_configuration`, and
  :doc:`user_guide/sampling_options/mutation_configuration`.
* The :doc:`user_guide/index` covers the full picture, starting with
  :doc:`user_guide/core_concepts/index` (including the
  :doc:`user_guide/core_concepts/step_context` distinction between shared
  inputs and evolving metadata).
* :doc:`examples` links to worked examples in the
  `sequince-examples <https://github.com/sequince-dev/sequince-examples>`_
  repository.
* The :doc:`API reference <autoapi/sequince/index>` lists classes, functions,
  and their parameters.
* The :doc:`developer_guide/index` links to the contribution guidelines and
  AI policy.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   installation
   user_guide/index
   examples
   API reference <autoapi/sequince/index>
   developer_guide/index
   citation

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
