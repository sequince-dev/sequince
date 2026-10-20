JAX and functional execution
============================

Sequince can run an SMC pipeline with JAX arrays and an explicitly threaded JAX
random key. The functional API separates pure iteration work from the host-side
loop, making one SMC iteration suitable for ``jax.jit`` while retaining
normal result collection, callbacks, and checkpoint handling between
iterations.

Install JAX support with ``pip install "sequince[jax]"``. To use BlackJAX mutation,
install ``pip install "sequince[blackjax]"`` instead; BlackJAX brings its JAX
dependency with it.

The functional API is not limited to BlackJAX. Any pipeline whose steps obey
the functional contract can use it; the pre-built MiniPCN pipeline is one other
supported option.

Running a functional pipeline
-----------------------------

Use JAX arrays for the particles and ensure that every target function called
inside the iteration uses JAX-compatible operations. Pass a JAX key as
``random_state`` rather than using the mutable ``rng`` argument accepted by
:meth:`~sequince.smc.SequentialMonteCarlo.run`:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    from sequince import ParticleState, make_standard_bayesian_smc
    from sequince.jax import compile_functional_iteration


    def log_prior(particles, context):
        inside = jnp.all(jnp.abs(particles) <= 5.0, axis=-1)
        return jnp.where(inside, 0.0, -jnp.inf)


    def log_likelihood(particles, context):
        return -0.5 * jnp.sum(particles**2, axis=-1)


    key = jax.random.key(42)
    particle_key, key = jax.random.split(key)
    initial_state = ParticleState(
        particles=jax.random.uniform(
            particle_key,
            shape=(500, 2),
            minval=-5.0,
            maxval=5.0,
        )
    )

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        mutation="blackjax",
        always_resample=True,
        logging_frequency=0,
    )

    iteration = compile_functional_iteration(
        smc,
        initial_state,
        key,
        metadata=metadata,
    )
    result, key = smc.run_functional(
        initial_state,
        random_state=key,
        metadata=metadata,
        iteration=iteration,
    )

:meth:`~sequince.smc.SequentialMonteCarlo.run_functional` returns both the
:class:`~sequince.result.SMCResult` and the advanced random state. Reuse the
returned key, rather than the input key, when continuing random work.
See :doc:`random_number_generation` for the corresponding eager-execution
contract and reproducibility guidance.

The example selects BlackJAX's default NUTS mutation. Configure the number of
transitions with ``mutation_n_steps`` and pass algorithm options through
``mutation_kwargs``. Supported named algorithms are ``"nuts"``, ``"hmc"``, and
``"random-walk"``; a callable algorithm factory is also accepted. Mutation
diagnostics such as acceptance and divergence are recorded in
``result.outputs``.

Compiling an iteration
----------------------

:func:`sequince.jax.compile_functional_iteration` initializes the state and
context fields required by the configured steps, lowers one iteration, verifies
that its input and output carry have matching tree structures, shapes, and
dtypes, and returns the compiled callable. Compilation does not execute or
discard a warm-up SMC transition.

The ``observation``, ``shared_context``, and ``metadata`` values supplied while
compiling must have structures compatible with the subsequent run. Array values
may change between iterations, but their tree structure, shapes, and dtypes must
remain compatible with the compiled signature.

For a simpler setup, JIT the bound iteration directly:

.. code-block:: python

    iteration = jax.jit(smc.functional_iteration)
    result, key = smc.run_functional(
        initial_state,
        random_state=key,
        metadata=metadata,
        iteration=iteration,
    )

This compiles on the first call. Passing no ``iteration`` also works, but runs
``smc.functional_iteration`` without an external transformation.

The functional step contract
----------------------------

An eager step returns ``(state, context, output)``. A functional step receives
an explicit random state and returns four values:

.. code-block:: python

    def apply_functional(self, state, context, random_state):
        # Perform array operations and advance random_state.
        return state, context, next_random_state, output

The default :meth:`~sequince.steps.base.SMCStep.apply_functional` adapter wraps
the backend random state in an ORNG generator and calls ``apply``. This is useful
when an eager implementation already consists of traceable array operations.
Override ``apply_functional`` when the functional implementation needs explicit
random-state control or JAX-specific control flow.

Compiled carries must keep a stable pytree structure. A custom step that adds
metadata or state leaves during its first call should create shape-complete
placeholder leaves before compilation by overriding
``initialize_functional_metadata`` or
``initialize_functional_state_and_context``. The runner invokes these hooks
automatically at the beginning of :meth:`~sequince.smc.SequentialMonteCarlo.run_functional`,
and the compilation helper invokes them before lowering.

Current functional limitations
------------------------------

* Functional resampling currently requires ``always_resample=True`` because a
  traced ESS-triggered RNG transition is not yet implemented.
* A functional iteration always completes its configured step pipeline. An
  immediate stop request is handled by the host loop after the iteration rather
  than skipping later compiled steps.
* Python side effects and NumPy-only operations inside a compiled step or target
  function are not JAX traceable. Keep logging in host-side callbacks or disable
  the built-in logging step with ``logging_frequency=0``.
* Changing carry structure, array shape, or dtype triggers a new JAX signature;
  the compilation helper rejects a pipeline whose input and output carry
  signatures differ.

The host loop in ``run_functional`` remains Python-driven. For a fixed-length
loop entirely inside a JAX transformation, call ``smc.functional_iteration``
from ``jax.lax.scan`` and explicitly carry the state, context, and random
key.

API reference
-------------

See :func:`~sequince.jax.compile_functional_iteration` in the generated API
reference.
