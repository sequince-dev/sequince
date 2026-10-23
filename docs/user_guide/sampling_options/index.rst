Sampling options
================

Resampling and mutation control how a particle population is carried between
SMC targets. Resampling draws from the weighted population and assigns equal
weights to the selected particles. Mutation applies MCMC transitions that
explore the current target while preserving the particle weights.

The pre-built samplers expose these choices through their ``resampling`` and
``mutation`` arguments. For a custom pipeline, configure the corresponding
steps directly. See :doc:`../running_inference/prebuilt_samplers` for the
available sampler designs.

* :doc:`resampling_configuration` explains systematic, stratified, and
  multinomial resampling, when to trigger resampling, and how to inspect
  ancestry and ESS diagnostics.
* :doc:`mutation_configuration` covers mutation algorithms, transition counts,
  proposal kernels, particle transforms, and mutation diagnostics.

Choose options compatible with your array backend and execution mode; see
:doc:`../core_concepts/array_backends` and
:doc:`../advanced_workflows/jax_support`. If you use genealogy-based evidence
uncertainty estimates, check their resampling assumptions in
:doc:`../running_inference/evidence_uncertainty` before choosing a scheme.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   resampling_configuration
   mutation_configuration
