User guide
==========

The Sequential Monte Carlo (SMC) loop in ``sequince`` is assembled from
composable steps. Each step receives the current particle state and runtime
context, then returns an updated state, context, and optional output. The same
mutation implementations can also be used through the standalone MCMC runner.
At the end of an iteration, callbacks can inspect the accumulated result and
update the context.

Start with the workflow guides when configuring a run and interpreting its
output. The remaining sections explain sampling choices, the objects passed
through the SMC loop, and more specialised execution and extension points.

.. toctree::
   :maxdepth: 1
   :caption: Running inference

   smc_configuration
   mcmc_sampler
   observations
   evidence_uncertainty

.. toctree::
   :maxdepth: 1
   :caption: Sampling options

   resampling_options
   mutation_options

.. toctree::
   :maxdepth: 1
   :caption: Core concepts

   particle_state
   step_context
   random_number_generation

.. toctree::
   :maxdepth: 1
   :caption: Advanced workflows

   jax_support
   checkpointing

.. toctree::
   :maxdepth: 1
   :caption: Extending Sequince

   custom_steps_callbacks
