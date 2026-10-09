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

Use ``make_nested_sampling_smc`` to construct the composable pipeline and wrap its
result in ``NestedSamplingSMCResult`` for shell evidence and posterior samples.
``NestedSamplingLevel`` provides the per-shell diagnostics. The ordinary
``SMCResult.log_evidence()`` and ``SMCResult.posterior_samples()`` methods raise
an error for NS-SMC results and direct callers to this wrapper.
``SMCResult.log_normalizing_constant()`` remains available for the constrained
prior normalization.

Custom threshold schedules must implement
``schedule(log_likelihoods, state, context) -> float | scalar array``. All three arguments are
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

Functional and compiled execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``run_functional`` supports JAX arrays with unconditional resampling
(``always_resample=True``) and an unchanged population size. BlackJAX and
MiniPCN provide functional mutation kernels; host-only kernels such as emcee,
Eryn, and the current Metropolis-Hastings implementations are rejected. Set
``logging_frequency=0`` and use runner callbacks for host logging. Existing
functional restrictions on transforms and mutation options still apply.

For example, using JAX-compatible density functions and particles::

    import jax
    from sequince import ParticleState, make_nested_sampling_smc
    from sequince.jax import compile_functional_iteration

    smc, target, metadata = make_nested_sampling_smc(
        log_prior, log_likelihood,
        threshold_schedule=[-10.0, -5.0, -1.0],
        evidence_tolerance=None,
        mutation="blackjax",
        mutation_kwargs={"algorithm": "random-walk",
                         "algorithm_kwargs": {"sigma": 0.1}},
        particle_transform=None,
        logging_frequency=0,
    )
    state = ParticleState(particles)
    key = jax.random.key(42)
    iteration = compile_functional_iteration(smc, state, key, metadata=metadata)
    result, key = smc.run_functional(
        state, random_state=key, metadata=metadata, iteration=iteration,
    )
    nested = result.as_nested_sampling_result()
    samples, weights = nested.posterior_samples()

Omit ``iteration`` for uncompiled functional execution, or supply
``jax.jit(smc.functional_iteration)``. The runner initializes numeric metadata
and a complete target cache before the first iteration. Call
``initialize_functional`` yourself when invoking ``functional_iteration``
directly. Context metadata remains authoritative; state metadata contains
independent historical snapshots. Stop reasons are numeric during functional
execution; ``nested.stop_reason`` and ``nested.summary()`` expose readable
strings. Shapes, dtypes, dictionary keys, and cache components must remain fixed.

Compiled schedules use the same three-argument signature. They must use JAX
operations without host conversions, Python branching on traced values, mutable
iterator state, or data-dependent exceptions. Return positive infinity to
integrate the terminal tail. The built-in fixed schedule uses this sentinel on
exhaustion while retaining ``StopIteration`` in eager execution. An explicit
infinite threshold (including one returned by a custom schedule) is reported
as ``no_survivors``; exhausting a built-in fixed schedule is reported as
``schedule_exhausted``. Ordinary user exceptions propagate during execution or
tracing. Target functions and transformations must also be pure and traceable;
log densities must be finite or negative infinity. Shape and cache ownership
checks run during tracing; density values and threshold errors are checked at
initialization and after each host-controlled iteration. Direct compiled calls
must invoke the selection step's ``validate_functional_state_and_context`` at
the host boundary. Rejected internal mutation proposals are not inspected by
these host checks.

After terminal-shell integration, later steps are conditionally skipped,
including weighting, resampling, and mutation. Their carry (particles, caches,
and RNG) is unchanged. Skipped outputs retain the usual structure with zero
array placeholders; they do not represent executed transitions. Functional
shell outputs use population-sized arrays and a particle mask. Optional
prior-mass tail outputs also carry a ``valid`` flag, allowing two shells in one
iteration. The result wrapper filters these flags and masks when reconstructing
levels and posterior samples. Eager shell storage remains compact, and
``store_shell_particles=False`` avoids retaining shell particle arrays in both
modes.

Evidence remains separate from constrained-prior normalizing constants. A
``num_steps`` limit suspends an incomplete run without integrating its live
tail. Use ``resume_functional_from(checkpoint)`` and then
``run_functional(runtime=runtime, ...)`` to continue. With a newly constructed
target, initialization refreshes incompatible cached evaluations before
compilation/execution; RNG and historical shells are restored. Portable result
serialization retains shell masks, diagnostics, and posterior interpretation.

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
