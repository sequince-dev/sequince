Getting started
===============

Installation
------------

Install from PyPI::

    pip install sequince

``sequince`` targets Python 3.11+. Its core dependencies are installed
automatically; integrations such as emcee, BlackJAX, and HDF5 checkpointing are
available as optional extras.

Quick start
-----------

The library exposes a high-level helper,
``sequince.make_standard_bayesian_smc``, that wires together a sensible set of
steps for many Bayesian posterior inference workflows. The basic usage is as
follows:

.. code-block:: python

    import numpy as np
    from sequince import ParticleState, make_standard_bayesian_smc

    # Set RNG
    rng = np.random.default_rng(42)

    # User-supplied log-density functions, these must be vectorized
    # Theta is a 2D array of shape (n_particles, n_parameters)
    # Context is dataclass that contains additional information
    def log_prior(theta, context):
        """Uniform prior over the specified bounds"""
        bounds = context.shared["bounds"]
        return np.where(
            np.all((theta >= bounds[0]) & (theta <= bounds[1]), axis=1),
            0.0,
            -np.inf,
        )

    def log_likelihood(theta, context):
        """Unnormalized Gaussian likelihood with mean 0 and std 1"""
        return -0.5 * np.sum(theta**2, axis=1)


    # Create a standard SMC runner for Bayesian inference
    # smc is the main runner
    # posterior is a convenience wrapper around the log-likelihood and log-prior
    # metadata is dictionary that can be used to pass information between steps
    # in the SMC loop
    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
    )

    # Shared context is a dictionary that can be used to pass user-defined
    # information to steps and the log-density functions.
    shared_context = dict(
        bounds=[-5.0, 5.0]
    )

    # Build the initial particle cloud with samples from the prior
    n = 500
    initial_particles = rng.uniform(*shared_context["bounds"], size=(n, 2))
    # Create a particle state object to hold the initial particles
    # Unless specified, the particles are assumed to be equally weighted
    initial_state = ParticleState(particles=initial_particles)

    # Run the SMC sampler
    result = smc.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )

    # Extract the posterior samples and log-evidence from the result object
    samples = result.posterior_samples()
    log_evidence = result.log_evidence()
    log_evidence_error = result.log_evidence_error()

Next steps
----------

See :doc:`evidence_uncertainty` for interpreting the evidence estimate and its
uncertainty. To customise the run, continue with :doc:`smc_configuration` and
the sampling choices in :doc:`resampling_options` and :doc:`mutation_options`.
The :doc:`step_context` guide explains the distinction between shared inputs
and evolving metadata. For sampling without SMC weighting and resampling, see
:doc:`mcmc_sampler`.
