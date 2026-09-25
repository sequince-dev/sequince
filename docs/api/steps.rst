SMC steps
=========

Common interface
----------------

.. automodule:: sequince.steps.base
   :members:
   :undoc-members:
   :show-inheritance:

Weighting strategies
--------------------

.. automodule:: sequince.steps.weighting
   :members:
   :undoc-members:
   :show-inheritance:

Nested-sampling SMC
-------------------

Use ``make_nested_sampling_smc`` to construct the eager pipeline and wrap its
result in ``NestedSamplingSMCResult`` for shell evidence and posterior samples.
``NestedSamplingLevel`` provides the per-shell diagnostics.

Custom threshold schedules must implement
``schedule(log_likelihoods, state, context) -> float``. All three arguments are
passed positionally. Raise ``StopIteration`` to integrate the remaining live
tail and finish; other exceptions propagate to the caller. Fixed, weighted
quantile, and worst-particle schedules implement this same signature.

During execution, ``context.metadata["nested_sampling"]`` holds the current
threshold and bookkeeping as a plain dictionary. Steps replace this dictionary
rather than mutate an earlier context. Shell expectations and evidence
increments are stored in ``NestedSamplingLevel`` records.
State metadata records snapshots after survivor
weighting or terminal-shell integration, preserving result histories. Resume
through the standard checkpoint API to restore both particles and context.

.. automodule:: sequince.steps.constraints
   :members:
   :undoc-members:
   :show-inheritance:

Resampling
----------

.. automodule:: sequince.steps.resampling
   :members:
   :undoc-members:
   :show-inheritance:

Mutation kernels
----------------

.. automodule:: sequince.steps.mutation
   :members:
   :undoc-members:
   :show-inheritance:

Proposals and dimensionality control
------------------------------------

.. automodule:: sequince.steps.proposal
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sequince.steps.dimensionality
   :members:
   :undoc-members:
   :show-inheritance:

Control flow and utilities
--------------------------

.. automodule:: sequince.steps.control
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sequince.steps.log
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sequince.steps.waste_free
   :members:
   :undoc-members:
   :show-inheritance:
