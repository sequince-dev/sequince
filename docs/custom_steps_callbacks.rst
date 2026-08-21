Custom steps and callbacks
==========================

Steps transform particle state within an iteration. Callbacks run after the
step pipeline and are intended for observation, reporting, or run-level control.
Both receive a :class:`~sequince.steps.context.StepContext`; see
:doc:`step_context` for its lifecycle and update methods.

Custom steps
------------

Derive a custom step from :class:`~sequince.steps.base.SMCStep` and implement
``apply``. It receives the current state, context, and random-number generator,
and must return ``(state, context, output)``:

.. code-block:: python

    from sequince.steps.base import SMCStep


    class RecordParticleMean(SMCStep):
        def apply(self, state, context, rng):
            mean = state.particles.mean(axis=0)
            context = context.with_metadata(last_mean=mean)
            return state, context, {"particle_mean": mean}

Use ``context.shared`` for read-only configuration and
``context.with_metadata(...)`` for current coordination values. Return
historical diagnostics as the explicit output so that the runner records them
in the corresponding entry of :attr:`~sequince.result.SMCResult.outputs`.

Custom callbacks
----------------

Derive a callback from :class:`~sequince.callbacks.SMCCallback` and implement
``apply``. A callback receives the state produced by the current iteration, the
current context, and the result accumulated so far:

.. code-block:: python

    from sequince.callbacks import SMCCallback


    class ReportProgress(SMCCallback):
        def apply(self, state, context, result):
            print(f"iteration={context.t}, states={len(result)}")

An observational callback may return ``None``. To update metadata or request a
stop, return a replacement context:

.. code-block:: python

    class StopAfter(SMCCallback):
        def __init__(self, final_iteration):
            self.final_iteration = final_iteration

        def apply(self, state, context, result):
            if context.t >= self.final_iteration:
                return context.request_stop()
            return None

Callbacks run in their configured order. A replacement context returned by one
callback is passed to the next callback and becomes ``result.context``.

For compiled JAX execution, custom steps must also obey the functional
interface's fixed-structure requirements. See :doc:`jax_support` for the
functional step contract and initialization hooks.
