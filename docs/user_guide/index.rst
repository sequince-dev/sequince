User guide
==========

The Sequential Monte Carlo (SMC) loop in ``sequince`` is assembled from
composable steps. Each step receives the current particle state and runtime
context, then returns an updated state, context, and optional output.
At the end of an iteration, callbacks can inspect the accumulated result and
update the context. This proceeds until a user-specified termination
criterion is satisfied, which returns a result object for further processing
and results visualisation.

Start with the core concepts to understand the objects passed through the
SMC loop. The running-inference guides cover pre-built samplers, configuration,
and interpreting outputs. The remaining sections explain sampling choices and
more specialised execution and extension points.

.. toctree::
   :maxdepth: 2

   core_concepts/index
   running_inference/index
   sampling_options/index
   advanced_workflows/index
   extending_sequince/index
