SMC steps
=========

Common interface
----------------

.. automodule:: sequince.steps.base
   :members:
   :undoc-members:
   :show-inheritance:

Before functional initialization, the runner calls each step's
``validate_functional_configuration(config, state)`` hook, passing its
``smc.config`` snapshot and the initial particle state.
``SMCConfig`` exposes ``steps``, ``callbacks``, ``checkpoint_writers``, and
``auto_normalize`` as read-only fields with tuple containers. Referenced step
and callback objects retain their identity and mutable state. The config is
host-side configuration, separate from runtime metadata and checkpoints.
Steps can reject incompatible pipeline settings here, before any target
evaluation or cache preparation. The default implementation does nothing.
Metadata and cache preparation remain in ``initialize_functional_metadata``
and ``initialize_functional_state_and_context``. Algorithm-specific checks
belong to the relevant steps, rather than the runner.

Functional conditional execution
--------------------------------

``sequince.steps.apply_functional_if(predicate, apply, state, context, key)``
is a reusable control-flow helper for functional steps, using ``acai.cond``.
It executes
``apply(state, context, key)`` only when the scalar predicate is true. Otherwise
it returns the unchanged state, context, and RNG. On NumPy, CuPy, and PyTorch,
only the selected branch executes and skipped output is None. PyTorch control
flow is eager, without structured ``torch.compile`` guarantees. On JAX, skipped
outputs contain zero array placeholders in the normal output structure, with
static fields retained. Both JAX branches are traced; the operation must be
pure and traceable even when skipped, and initialization must establish matching
carry structures, shapes, and dtypes. Placeholder outputs do not describe
executed transitions.

After each functional step, the runner guards subsequent steps using the
runtime ``context.stop_immediately`` flag. ``StopIf(immediate=True)``, NS-SMC
shell steps, and custom steps returning ``context.request_stop_immediately()``
all stop execution without an additional capability declaration. An ordinary
stop request still permits the rest of the iteration to execute. Use
``initialize_functional`` before directly invoking a compiled iteration so
all guarded steps have compatible carries; ``run_functional`` and
``compile_functional_iteration`` perform this initialization automatically.
As in eager execution, runner-level automatic normalization still runs after
the step sequence when enabled.

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
