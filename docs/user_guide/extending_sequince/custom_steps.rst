Custom steps
============

Steps perform work within an SMC iteration, receiving the current particle
state and a :class:`~sequince.steps.context.StepContext`. For reporting and
run-level control after an iteration, see :doc:`custom_callbacks`.

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

For compiled JAX execution, custom steps must also obey the functional
interface's fixed-structure requirements. See :doc:`../advanced_workflows/jax_support` for the
functional step contract and initialization hooks.
