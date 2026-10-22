Random number generation
========================

Sequince makes the source and ownership of randomness explicit so runs can be
reproduced, resumed, and executed with different array backends. Every
stochastic step receives its random-number generator from the runner; custom
steps should use that object rather than global functions such as
``numpy.random.random``.

Eager execution
---------------

The ``rng`` argument accepted by
:meth:`~sequince.smc.SequentialMonteCarlo.run` may be:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Value
     - Behaviour
   * - An integer seed
     - Creates a NumPy-backed ORNG generator. Reusing the seed for an otherwise
       identical run reproduces the random stream.
   * - A ``numpy.random.Generator``
     - Wraps and advances that mutable generator.
   * - An ``orng.RandomGenerator``
     - Uses the supplied generator directly, including its configured backend.
   * - ``None``
     - Creates a fresh NumPy-backed generator without a user-specified seed.

For a simple reproducible run, pass an integer:

.. code-block:: python

    result = smc.run(initial_state, rng=42, metadata=metadata)

When initialisation is also random, create one generator and pass the same
object onward. The runner continues from its current position:

.. code-block:: python

    import numpy as np

    rng = np.random.default_rng(42)
    initial_particles = rng.normal(size=(500, 2))
    initial_state = ParticleState(initial_particles)

    result = smc.run(initial_state, rng=rng, metadata=metadata)

The generator is mutable in eager execution. Supplying the same generator
object to another operation continues its stream; it does not restart from the
original seed. To reproduce a run, recreate both the initial state and runner
inputs from the same seed and configuration.

RNG in custom steps
-------------------

An eager step receives an :class:`orng.RandomGenerator` as its third argument:

.. code-block:: python

    from sequince.steps.base import SMCStep


    class GaussianJitter(SMCStep):
        def apply(self, state, context, rng):
            noise = rng.normal(size=state.particles.shape)
            particles = state.particles + 0.01 * noise
            return state.with_updates(particles=particles), context, None

Do not create a new fixed-seed generator inside ``apply``. Doing so restarts the
same random sequence on every iteration and bypasses checkpointed RNG state.

Functional and JAX execution
----------------------------

Functional execution does not mutate a generator. Pass backend-native state as
``random_state`` and retain the advanced state returned by the runner:

.. code-block:: python

    key = jax.random.key(42)
    result, key = smc.run_functional(
        initial_state,
        random_state=key,
        metadata=metadata,
    )

Each functional step accepts the current random state and must return its
replacement. Reusing the input key instead of the returned key repeats the same
random stream. See :doc:`../advanced_workflows/jax_support` for compiling the functional iteration
and splitting a key for random particle initialisation.

Checkpointing and resuming
--------------------------

Checkpoint writers save RNG state by default. When that state is present,
``resume_from`` restores it and ignores any replacement RNG supplied to the
resume call. This makes a split run follow the same random stream as an
uninterrupted run.

With ``save_rng=False``, pass ``rng`` to ``resume_from`` to deliberately start
the continuation from a new random state. Once a runtime has been prepared,
pass only ``runtime`` to ``run``; RNG and context arguments cannot be supplied
again. See :doc:`../advanced_workflows/checkpointing` for the complete resume workflow.

Random posterior draws
----------------------

:meth:`~sequince.result.SMCResult.posterior_samples` also accepts a seed,
NumPy generator, or ORNG generator. Pass one explicitly when the resampled
posterior array must be reproducible:

.. code-block:: python

    samples = result.posterior_samples(1_000, rng=42)
