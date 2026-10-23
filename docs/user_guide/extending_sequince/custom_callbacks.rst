Custom callbacks
================

Callbacks run after the step pipeline and can inspect completed iterations,
report progress, or control the run. They receive a
:class:`~sequince.steps.context.StepContext`; see
:doc:`../core_concepts/step_context` for its lifecycle and update methods.
To perform work within the step pipeline, see :doc:`custom_steps`.

For ready-made plotting callbacks and their configuration, see
:doc:`../running_inference/callbacks`.

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
