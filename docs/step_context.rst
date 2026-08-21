Step context
============

Every SMC step receives a :class:`~sequince.steps.context.StepContext` alongside
the particle state and random-number generator. The context describes the
current iteration and carries information that does not belong to an individual
particle. The same interface may also be passed to target functions and is
passed to callbacks.

You normally do not construct a context yourself. The
:class:`~sequince.smc.SequentialMonteCarlo` runner creates one at the start of
each iteration from the arguments passed to
:meth:`~sequince.smc.SequentialMonteCarlo.run` or
:meth:`~sequince.smc.SequentialMonteCarlo.run_functional`. Steps may return
updated copies, and callbacks may return an updated copy or ``None``.

Fields
------

``t``
    The zero-based index of the current SMC iteration. ``iteration`` is a
    read-only alias for the same value.

``observation``
    The observation selected for the current iteration, or ``None`` when no
    observation is available. If ``num_steps`` is specified, a run can continue
    after its observations are exhausted; those additional iterations receive
    ``None``. See :doc:`observations` for complete iteration and exhaustion
    semantics.

``next_observation_index``
    The index at which the runner will look for the next observation. When the
    current iteration consumed an observation at index ``i``, this value is
    already ``i + 1``. It is primarily useful for checkpointing and resuming a
    run; use ``observation`` to access the current value.

``shared``
    User-supplied, run-wide configuration from ``shared_context``. Treat this
    mapping as read-only. It is suitable for static data such as bounds,
    constants, or model configuration.

``metadata``
    Values that steps and callbacks communicate to later steps, callbacks, and
    iterations. Use :meth:`~sequince.steps.context.StepContext.with_metadata`
    to return a context containing additions or replacements.

``stop_requested`` and ``stop_immediately``
    Control flags set by the stop-request methods described below. Most code
    should use those methods rather than setting the fields directly.

Shared data and metadata
------------------------

The distinction between ``shared`` and ``metadata`` is whether a value changes
during a run. Supply static dependencies once through ``shared_context``:

.. code-block:: python

    result = smc.run(
        initial_state,
        rng=rng,
        shared_context={"bounds": (-5.0, 5.0)},
        metadata={"accepted": 0},
    )

A custom step can read both mappings and publish updated metadata:

.. code-block:: python

    from sequince.steps.base import SMCStep


    class CountCalls(SMCStep):
        def apply(self, state, context, rng):
            bounds = context.shared["bounds"]
            calls = context.metadata.get("calls", 0) + 1
            context = context.with_metadata(calls=calls)
            return state, context, {"bounds_used": bounds}

``StepContext`` is a frozen dataclass. Its helper methods return replacements
instead of changing the context object in place. ``with_metadata`` performs a
shallow copy and merge: keys supplied to the method replace existing keys, and
all other keys are retained. Values inside either mapping may still be mutable,
so avoid modifying them in place if iteration history or functional execution
must remain reproducible.

The runner carries the final ``metadata`` mapping into the next iteration.
Within an iteration, each step receives the context returned by the preceding
step. After the steps finish, callbacks run in order; a callback that returns a
context passes it to the next callback, while returning ``None`` leaves the
context unchanged. The final context is available as ``result.context``.

Stopping a run
--------------

:meth:`~sequince.steps.context.StepContext.request_stop` requests a deferred
stop. The remaining steps, automatic weight normalization, and callbacks for
the current iteration still run. The runner then finishes the run.

:meth:`~sequince.steps.context.StepContext.request_stop_immediately` skips the
remaining steps in the current iteration. Automatic normalization and callbacks
still run before the runner finishes. Because callbacks execute after the step
pipeline, either stop method called from a callback stops after that iteration;
an immediate request does not skip later callbacks.

Functional execution is intentionally different: a call to
:meth:`~sequince.smc.SequentialMonteCarlo.functional_iteration` always completes
the fixed step pipeline so that it remains suitable for compilation. In a
functional run, both kinds of stop request are handled by the host loop only
after the complete iteration.

:meth:`~sequince.steps.context.StepContext.clear_stop_request` clears both stop
flags. This is useful in control-flow steps that conditionally override a stop
requested by an inner step.

.. code-block:: python

    class StopAtIteration(SMCStep):
        def __init__(self, final_iteration):
            self.final_iteration = final_iteration

        def apply(self, state, context, rng):
            if context.t >= self.final_iteration:
                context = context.request_stop()
            return state, context, None

Context lifecycle
-----------------

For each iteration, the runner performs the following operations:

#. Selects the current observation and creates a fresh context with the current
   ``t``, observation position, shared data, and metadata.
#. Passes the state and context through each configured step in order. Eager
   execution can end this pipeline early for an immediate stop request.
#. Normalizes particle weights when ``auto_normalize=True``.
#. Records the state and step outputs, then invokes callbacks in order.
#. Writes any due checkpoints and either stops or advances to the next
   iteration with the returned metadata.

When resuming from a checkpoint, the iteration number, observation position,
and metadata are restored. ``shared_context`` is supplied again when creating
the resumed runtime because it is deliberately not stored in the checkpoint.

API reference
-------------

.. autoclass:: sequince.steps.context.StepContext
   :members:
