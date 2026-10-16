Configuring an SMC runner
=========================

The :class:`~sequince.smc.SequentialMonteCarlo` runner applies a sequence of
steps and optional callbacks at every iteration. The quickest way to build a
Bayesian inference pipeline is the pre-built helper:

.. code-block:: python

    from sequince import make_standard_bayesian_smc
    from sequince.callbacks import TracePlot

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        mutation="emcee",
        resampling="systematic",
        callbacks=[TracePlot()],
    )

``make_standard_bayesian_smc`` constructs an SMC runner that adaptively
steps through a series of simulated annealing temperatures
(:class:`~sequince.steps.weighting.AdaptiveTemperedWeighting`), applying
See :doc:`resampling_options` and :doc:`mutation_options` for the available
sampling strategies.
See :doc:`jax_support` for JAX-backed execution, which needs some additional
function calls to enable full JAX functionality (``jit``, ``vmap``, ``grad``, ...).

When starting the run, pass the ``metadata`` returned by the convenience constructor:

.. code-block:: python

    result = smc.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )

See :doc:`step_context` for the distinction between static ``shared_context``
data and evolving metadata.

Building a pipeline manually
----------------------------

For full control, instantiate the runner with steps in execution order. For example,
to reproduce the pre-built SMC defined above:

.. code-block:: python

    from sequince import SequentialMonteCarlo
    from sequince.callbacks import TracePlot
    from sequince.steps import (
        AdaptiveTemperedWeighting,
        EmceeMutation,
        SystematicResampling,
    )

    smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(posterior),
            SystematicResampling(always_resample=True),
            EmceeMutation(posterior, n_steps=10),
        ],
        callbacks=[TracePlot()],
    )

Each entry in ``steps`` must follow the
:class:`~sequince.steps.base.SMCStep` interface. Each callback must follow the
:class:`~sequince.callbacks.SMCCallback` interface. See
:doc:`custom_steps_callbacks` when implementing either interface yourself.
