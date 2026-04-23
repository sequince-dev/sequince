Getting started
===============

Installation
------------

The project is published from the main branch and can be installed directly
from GitHub::

    pip install git+https://github.com/sequince-dev/sequince.git

``sequince`` targets Python 3.10+ and depends on ``numpy``, ``scipy``,
``matplotlib``, and ``emcee``. These will be installed automatically when the
package is installed via ``pip``.

Quick start
-----------

The library exposes a high-level helper,
``sequince.make_standard_bayesian_smc``, that wires together a sensible set of
steps for many Bayesian posterior inference workflows. The example below mirrors
the usage demonstrated in the README

.. code-block:: python

    import numpy as np
    from sequince import ParticleState, make_standard_bayesian_smc

    # User-supplied log-density functions
    def log_prior(theta):
        ...

    def log_likelihood(theta):
        ...

    smc, posterior, meta_data = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
    )

    shared_context = dict()

    # Build the initial particle cloud
    n = 2_048
    initial_particles = np.random.normal(size=(n, 2))
    initial_state = ParticleState(
        particles=initial_particles,
        log_weights=np.full(n, -np.log(n)),
    )

    results = smc.run(
        initial_state,
        rng=np.random.default_rng(),
        shared_context=shared_context,
        meta_data=meta_data,
    )
    samples = results.posterior_samples(1_000)

Check the :doc:`user_guide` for a deeper look at how to customise the SMC loop
and swap in different building blocks.


Meta data & shared context
--------------------------

The ``meta_data`` dictionary is intended to hold information that needs to be
passed between steps in the SMC loop. It can be updated at each step.

The ``shared_context`` dictionary is intended to hold information that needs to be
passed between steps in the SMC loop, but is not intended to be updated. For example,
it can be used to store static data that is needed by the likelihood function.
