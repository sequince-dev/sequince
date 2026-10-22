Array backends
==============

Choosing an array backend
-------------------------

Sequince uses the Python Array API through ``array-api-compat`` to work with
NumPy, JAX, PyTorch, and CuPy arrays. The initial particle array selects the
backend; there is no separate global backend setting. Install the array
library you intend to use:

* NumPy is included in the core installation.
* For JAX, install ``sequince[jax]``. See the
  `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
  for accelerator-specific packages.
* For PyTorch, follow the
  `PyTorch installation selector <https://pytorch.org/get-started/locally/>`_
  for your platform and compute device.
* For CuPy, follow the
  `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_
  to select a package compatible with your CUDA environment.

PyTorch and CuPy are installed separately; Sequince does not provide extras
for them. Backend support also depends on the steps and mutation algorithms
in your pipeline. In particular, selecting JAX arrays alone does not compile
a run; see :doc:`../advanced_workflows/jax_support` for functional
execution and compilation requirements.

Enabling SciPy Array API support
--------------------------------

SciPy's experimental Array API dispatch must be enabled before SciPy is
imported. Set the environment variable before starting Python:

.. code-block:: bash

    export SCIPY_ARRAY_API=1
    python your_script.py

Alternatively, set it at the beginning of your script, before importing
SciPy or packages that import SciPy:

.. code-block:: python

    import os

    os.environ["SCIPY_ARRAY_API"] = "1"

    import sequince

Sequince sets ``SCIPY_ARRAY_API=1`` during import if it is unset. It preserves
an existing value and warns if that value differs from ``"1"``. Setting it
explicitly avoids import-order problems. If SciPy has already been imported
in a notebook, restart the kernel and set the variable before importing it.
See `SciPy's Array API guide
<https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html>`_ for details.

Writing backend-compatible targets
----------------------------------

Create particles with your chosen library, and keep target evaluations in
the same array namespace. The following target works with each of the
backends above:

.. code-block:: python

    from sequince.array_api import namespace


    def log_likelihood(particles, context):
        xp = namespace(particles)
        return -0.5 * xp.sum(particles**2, axis=-1)

Keep related arrays on compatible devices and use floating-point particles.
Avoid converting particles to NumPy inside a target when using another
backend. For JAX runs requiring 64-bit precision, call
``jax.config.update("jax_enable_x64", True)`` before creating arrays.
