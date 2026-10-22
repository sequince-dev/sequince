Installation
============

``sequince`` requires Python 3.11 or later. Install it from PyPI:

.. code-block:: bash

    python -m pip install sequince

The core installation includes NumPy, SciPy, MiniPCN, and the Array API
compatibility packages. It supports NumPy-based sampling with the default
MiniPCN mutation backend.

Optional integrations
---------------------

Install extras for the integrations you need, for example:

.. code-block:: bash

    python -m pip install "sequince[jax,blackjax,hdf5]"

.. list-table::
   :header-rows: 1

   * - Extra
     - Provides
   * - ``jax``
     - JAX arrays and functional execution
   * - ``blackjax``
     - BlackJAX mutation algorithms
   * - ``emcee``
     - emcee ensemble mutation
   * - ``eryn``
     - Eryn ensemble mutation
   * - ``hdf5``
     - HDF5 result storage and checkpointing
   * - ``corner``
     - Corner plots

For example, install ``sequince[emcee]`` to select emcee mutation instead of
the default MiniPCN backend. See
:doc:`user_guide/sampling_options/mutation_configuration` for mutation-specific
requirements.

See :doc:`user_guide/core_concepts/array_backends` for backend selection,
accelerator setup, and SciPy Array API configuration.
