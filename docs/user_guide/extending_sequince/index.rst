Extending Sequince
==================

Extend a pipeline with custom steps to update the particle population or
perform work within an iteration. Use callbacks to inspect completed
iterations, report progress, update run metadata, or request a stop.

* :doc:`custom_steps` explains the step interface, returning updated state
  and context, and recording diagnostics in the result. It includes an
  example that records the particle mean.
* :doc:`custom_callbacks` explains the callback interface and returning
  context updates. It includes examples that report progress or stop a run.

Before implementing a component, review
:doc:`particle state <../core_concepts/particle_state>` and
:doc:`step context <../core_concepts/step_context>` to decide where its data
belongs. Stochastic steps should use the runner's generator as described in
:doc:`../core_concepts/random_number_generation`.

For components that will run inside a compiled JAX iteration, also follow
the functional step contract and initialization requirements in
:doc:`../advanced_workflows/jax_support`. See
:doc:`../running_inference/smc_configuration` for assembling the components
into a pipeline.

.. toctree::
   :maxdepth: 1

   custom_steps
   custom_callbacks
