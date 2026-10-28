Standalone MCMC
===============

Sequince's mutation implementations can be run as a standalone Markov chain
Monte Carlo sampler, without SMC weighting or resampling. This is useful for
ordinary posterior sampling and for testing a mutation configuration before
using it to rejuvenate an SMC population.

Creating a sampler
------------------

Use :func:`~sequince.mcmc.make_mcmc` with a vectorised log-density. Each row of
the particle array is a chain. The default differential-evolution method
updates interacting chains and requires at least four particles:

.. code-block:: python

    import numpy as np

    from sequince import ParticleState, make_mcmc


    def log_prob(particles, context):
        return -0.5 * np.sum(particles**2, axis=-1)


    rng = np.random.default_rng(42)
    initial_state = ParticleState(
        particles=rng.normal(size=(32, 2)),
    )

    mcmc = make_mcmc(
        log_prob,
        n_steps=1_000,
    )
    result = mcmc.run(initial_state, rng=rng)

``n_steps`` is the number of MCMC transitions in the returned chain, not the
number of chains. The number of rows in ``initial_state.particles`` determines
the number of chains.

Reading chains and diagnostics
------------------------------

:class:`~sequince.mcmc.MCMCResult.chain` has shape
``(n_steps, n_chains, n_dimensions)``. Use ``samples`` to discard burn-in,
thin, and optionally flatten the chain axis:

.. code-block:: python

    chain = result.chain
    samples = result.samples(burn_in=200, thin=5)
    unflattened = result.samples(burn_in=200, thin=5, flat=False)

    final_positions = result.final_state.particles
    diagnostics = result.diagnostics

The flattened result has shape ``(retained_steps * n_chains, n_dimensions)``.
``diagnostics`` contains the final mutation block's backend-specific values,
such as acceptance information and the number of transitions.

Choosing a mutation backend
---------------------------

``make_mcmc`` accepts the same named mutation backends used by the standard SMC
helper: ``"differential-evolution"`` (default), ``"minipcn"``, ``"emcee"``,
``"blackjax"``, and ``"metropolis-hastings"``. Backend-specific settings are passed through
``mutation_kwargs``. Install ``sequince[minipcn]`` before selecting MiniPCN:

.. code-block:: python

    mcmc = make_mcmc(
        log_prob,
        mutation="metropolis-hastings",
        n_steps=1_000,
        mutation_kwargs={
            "kernel": "gaussian-random-walk",
            "kernel_kwargs": {"scale": 0.2},
        },
    )

See :doc:`../sampling_options/mutation_configuration` for available backends, optional dependencies, and
kernel configuration. A configured
:class:`~sequince.steps.mutation.base.MutationStep` can also be passed directly to
:class:`~sequince.mcmc.MarkovChainMonteCarlo`; the runner enables chain storage
on that step automatically.

Context and particle transforms
-------------------------------

A target may omit ``context`` when it does not need run-level inputs. When it
does, supply values using ``shared_context`` just as for SMC:

.. code-block:: python

    def log_prob(particles, context):
        scale = context.shared["scale"]
        return -0.5 * np.sum((particles / scale) ** 2, axis=-1)


    result = mcmc.run(
        initial_state,
        rng=42,
        shared_context={"scale": 2.0},
    )

Pass a configured :class:`~sequince.transforms.ParticleTransform` as
``particle_transform`` when the chains should evolve in transformed
coordinates. Stored chains and final particles are returned in the original
particle coordinates.

Checkpointing
-------------

The standalone runner accepts callbacks and checkpoint writers. It executes
one mutation block per call to ``run``; a checkpoint therefore represents the
end of that block. Reconstruct the same mutation configuration, call
``resume_from(checkpoint)``, and pass the returned runtime to ``run`` to append
another block to the stored chain. See :doc:`checkpointing` for formats and RNG
restoration.
