User guide
==========

The Sequential Monte Carlo (SMC) loop in ``sequince`` is assembled from
composable steps. Each step receives the current particle state and runtime
context, then returns an updated state, context, and optional output.
At the end of an iteration, callbacks can inspect the accumulated result and
update the context. This proceeds until a user-specified termination
criterion is satisfied, which returns a result object for further processing
and results visualisation.

Start with the workflow guides when configuring a run and interpreting its
output. The remaining sections explain sampling choices, the objects passed
through the SMC loop, and more specialised execution and extension points.

.. toctree::
   :maxdepth: 2

   user_guide/running_inference
   user_guide/sampling_options
   user_guide/core_concepts
   user_guide/advanced_workflows
   user_guide/extending_sequince
