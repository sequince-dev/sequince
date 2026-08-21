Mutation options
================

Mutation rejuvenates the particle population after weighting and resampling.
It applies MCMC transitions that preserve the current target distribution,
helping duplicated particles explore again without changing their weights.

The :func:`~sequince.prebuilt_samplers.make_standard_bayesian_smc` helper
selects a mutation implementation with ``mutation``:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Value
     - Implementation
     - Typical use
   * - ``"minipcn"``
     - :class:`~sequince.steps.mutation.MiniPCNMutation`
     - General default with eager and functional execution support.
   * - ``"emcee"``
     - :class:`~sequince.steps.mutation.EmceeMutation`
     - Ensemble MCMC with emcee's collection of proposal moves.
   * - ``"blackjax"``
     - :class:`~sequince.steps.mutation.BlackJAXMutation`
     - JAX-backed NUTS, HMC, or random-walk mutation.
   * - ``"metropolis-hastings"``
     - :class:`~sequince.steps.mutation.MetropolisHastingsMutation`
     - Explicit control over a proposal kernel and transition ratio.
   * - ``"none"`` or ``None``
     - No mutation step
     - Reweighting and resampling only.
   * - An ``SMCStep`` instance
     - The supplied step
     - A custom, already configured mutation implementation.

Common configuration
--------------------

``mutation_n_steps`` controls how many MCMC transitions are applied to every
particle during one SMC iteration. The pre-built sampler defaults to five.
Strategy-specific options are passed through ``mutation_kwargs``:

.. code-block:: python

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        mutation="minipcn",
        mutation_n_steps=10,
        mutation_kwargs={
            "store_chains": True,
            "transform_refit": "per-context",
        },
        particle_transform="affine",
    )

All built-in mutation steps support these common options:

``store_chains``
    Include the intermediate MCMC positions in the step output. It defaults to
    ``False`` because full chains can be large. Acceptance diagnostics and the
    number of transitions are returned even when chains are not stored.

``transform_refit``
    Control how an adaptive particle transform is fitted. ``"always"`` fits on
    every mutation call, ``"per-context"`` reuses it within the same iteration
    and observation position, and ``"once"`` reuses the first fit. Eager
    execution also accepts a callable predicate; compiled functional execution
    requires one of the named policies.

Particle transforms
-------------------

The helper mutates particles in transformed coordinates by default using
``particle_transform="affine"``. This standardizes the particle cloud before
the MCMC transition and maps accepted positions back afterward. Configure it
through ``particle_transform_kwargs``:

.. code-block:: python

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior,
        log_likelihood,
        particle_transform="affine",
        particle_transform_kwargs={"eps": 1e-5},
    )

Other registered names are ``"identity"``,
``"bounded_to_unit_hypercube"``, ``"bounded_to_unbounded"``, and
``"periodic_boundary"``. The bounded and periodic transforms require bounds in
``particle_transform_kwargs``. Pass ``None`` to mutate directly in the original
coordinates, or pass a configured
:class:`~sequince.transforms.ParticleTransform` instance.

Auxiliary information for custom transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A custom transform may return invocation-specific information that its inverse
needs without storing mutable state on the transform. For example, a
many-to-one transform can preserve the branch assigned to each particle::

    from sequince.array_api import namespace
    from sequince.transforms import ParticleTransform


    class SignFoldingTransform(ParticleTransform):
        def forward(self, values, context=None, *, rng=None):
            xp = namespace(values)
            signs = xp.where(values < 0, -xp.ones_like(values), xp.ones_like(values))
            transformed = xp.abs(values)
            log_det = xp.zeros_like(values[..., 0])
            return transformed, log_det, signs

        def inverse(
            self,
            values,
            context=None,
            *,
            auxiliary=None,
            rng=None,
        ):
            if auxiliary is None:
                raise ValueError("Sign information is required for inversion.")
            xp = namespace(values)
            return values * auxiliary, xp.zeros_like(values[..., 0])

Auxiliary information is ephemeral: mutation passes it from one ``forward``
call through transformed-space operations to the corresponding ``inverse``.
It may be a single array or any Optree-compatible structure, including a tuple,
list, or dictionary such as ``(mode_ids, mask)`` or
``{"mode_ids": mode_ids, "mask": mask}``. Simple transforms do not need a
Sequince-specific wrapper. Libraries needing named fields may register their
own custom pytree types; immutable dataclasses can use
:func:`sequince.utils.pytree_dataclass` for Optree and JAX registration.

For a batch of ``n_particles``, Sequince treats an auxiliary array leaf as
particle-aligned exactly when ``leaf.ndim > 0`` and
``leaf.shape[0] == n_particles``. A mutation operating on a particle subset
applies the same selection to those leaves before inversion, while preserving
other leaves unchanged. Consequently, an unrelated array whose leading size
happens to equal ``n_particles`` is also selected; this is an intentional
limitation of the current shape-based convention.

Particle-aligned auxiliary works with batch-preserving mutation and BlackJAX.
It is currently rejected by emcee: emcee selects walker subsets internally,
but its standard log-probability callback does not expose their original walker
indices, so Sequince cannot guarantee the correct auxiliary entry.

MiniPCN
-------

``"minipcn"`` is the default and requires no extra installation. It uses the
``"tpcn"`` step function by default and targets an acceptance rate of ``0.234``.
Both can be changed through ``mutation_kwargs``:

.. code-block:: python

    mutation="minipcn",
    mutation_kwargs={
        "step_fn": "tpcn",
        "target_acceptance_rate": 0.3,
    },

MiniPCN implements explicit random-state threading and can be used in a compiled
functional pipeline. See :doc:`jax_support` for the functional runner.

emcee
-----

Install emcee support with ``uv add "sequince[emcee]"``. With no move specified,
emcee uses its default stretch move. Named choices are ``"de"``,
``"de-snooker"``, ``"gaussian"``, ``"kde"``, ``"stretch"``, and ``"walk"``.
The ``"bilby-de"`` preset supplies a Bilby-style weighted
differential-evolution schedule:

.. code-block:: python

    mutation="emcee",
    mutation_n_steps=20,
    mutation_kwargs={
        "move": "bilby-de",
    },

The preset is equivalent to using emcee's ``DEMove`` with its automatically
chosen differential-evolution scale 90% of the time and ``gamma0=1.0`` 10% of
the time. Both moves use ``sigma=1e-5``:

.. code-block:: python

    mutation_kwargs={
        "move": [
            {
                "move": "de",
                "move_kwargs": {"gamma0": None, "sigma": 1e-5},
                "weight": 0.9,
            },
            {
                "move": "de",
                "move_kwargs": {"gamma0": 1.0, "sigma": 1e-5},
                "weight": 0.1,
            },
        ],
    },

This preset does not require Bilby itself; it constructs the schedule from
emcee move objects. The expanded form can be adapted to create a custom
Bilby-like schedule with different moves, parameters, or weights.

``move`` may also be an emcee move object, a mapping containing ``move`` and
``move_kwargs``, or a weighted sequence of move specifications. Other keyword
arguments are forwarded to ``emcee.EnsembleSampler``; ``run_kwargs`` is
forwarded to its ``run_mcmc`` method.

Metropolis--Hastings
--------------------

The Metropolis--Hastings option requires a proposal ``kernel``. Sequince
provides ``"gaussian-random-walk"`` and ``"mixture"`` as named kernels:

.. code-block:: python

    mutation="metropolis-hastings",
    mutation_n_steps=10,
    mutation_kwargs={
        "kernel": "gaussian-random-walk",
        "kernel_kwargs": {"scale": 0.25},
    },

The mixture kernel accepts a ``kernels`` sequence and optional ``weights`` in
``kernel_kwargs``. A custom kernel object or factory can be passed directly;
it must provide proposals and one log transition ratio per particle. When a
kernel object is supplied, configure it directly rather than also passing
``kernel_kwargs``.

BlackJAX
--------

``"blackjax"`` accepts ``"nuts"`` (the default), ``"hmc"``,
``"random-walk"``, or a callable algorithm factory through
``mutation_kwargs["algorithm"]``. Algorithm parameters belong in
``mutation_kwargs["algorithm_kwargs"]``. NUTS and HMC require differentiable,
JAX-compatible targets and JAX-backed particles.

See :doc:`jax_support` for installation, a complete compiled example, random-key
threading, and current functional limitations.

Disabling or replacing mutation
-------------------------------

Use ``mutation="none"`` when mutation is deliberately unnecessary. Resampling
without mutation can rapidly reduce particle diversity, so monitor effective
sample size and duplicated particles when choosing this mode.

You can also pass an initialized :class:`~sequince.steps.base.SMCStep` instance
as ``mutation``. The helper inserts that object unchanged; it does not inject
the generated posterior, particle transform, ``mutation_n_steps``, or
``mutation_kwargs``. Configure a custom step completely before passing it, or
construct the full pipeline manually as described in :doc:`smc_configuration`.

Reading mutation outputs
------------------------

Each mutation step returns diagnostics in its position within the iteration's
output tuple. The runner collects those tuples in ``result.outputs``. Built-in
outputs include acceptance information, the configured number of transitions,
and ``chain``; the chain value is ``None`` unless ``store_chains=True``.
