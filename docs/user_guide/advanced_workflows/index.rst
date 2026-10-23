Advanced workflows
==================

These guides cover running mutation algorithms independently, compiling SMC
iterations with JAX, and continuing a run from a saved checkpoint. Each builds
on the configuration and result interfaces introduced in
:doc:`../running_inference/index`.

* :doc:`mcmc_sampler` shows how to use mutation steps for posterior sampling
  with a standalone MCMC runner, then extract chains and diagnostics.
* :doc:`jax_support` explains functional execution with explicit random keys,
  how to compile an iteration, and the requirements on steps, targets, and
  runtime state.
* :doc:`checkpointing` covers checkpoint writers, saved state, and resuming
  execution, including restoring the random-number-generator state.

For JAX workflows, first configure your
:doc:`array backend <../core_concepts/array_backends>` and review
:doc:`../core_concepts/random_number_generation`. To select mutation and
resampling algorithms, see :doc:`../sampling_options/index`.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   mcmc_sampler
   jax_support
   checkpointing
