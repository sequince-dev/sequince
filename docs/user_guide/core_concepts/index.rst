Core concepts
=============

An SMC run passes a particle population, a context, and a source of randomness
through a sequence of steps. These guides explain how those objects fit
together and how the choice of array backend affects your target functions.

* :doc:`array_backends` explains backend selection, SciPy Array API setup,
  and writing target functions that use the same namespace as the particles.
* :doc:`particle_state` introduces particle locations, weights, and metadata,
  how to return updated states, and how the result records iteration history.
* :doc:`step_context` describes shared inputs, evolving run metadata,
  observations, and stop requests used to coordinate steps and callbacks.
* :doc:`random_number_generation` covers seeds, generator ownership, and
  explicit random keys for reproducible eager and functional execution.

Read these guides before building a custom pipeline or implementing new
steps. To configure a sampler and interpret its output, continue with
:doc:`../running_inference/index`; to implement your own components, see
:doc:`../extending_sequince/index`.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   array_backends
   particle_state
   step_context
   random_number_generation
