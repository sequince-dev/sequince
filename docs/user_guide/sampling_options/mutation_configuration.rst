Mutation configuration
======================

Mutation steps apply MCMC transitions to a particle population without changing
its weights. Use them in an SMC pipeline or with the standalone
:class:`~sequince.mcmc.MarkovChainMonteCarlo` runner; see :doc:`../advanced_workflows/mcmc_sampler`.
Each step takes a target callable or :class:`~sequince.target.Target` object.

.. list-table::
   :header-rows: 1
   :widths: 18 24 28 30

   * - Method
     - ``mutation`` name
     - Implementation
     - Typical use
   * - MiniPCN
     - ``"minipcn"``
     - :class:`~sequince.steps.mutation.minipcn.MiniPCNMutation`
     - Supports eager and functional execution.
   * - emcee
     - ``"emcee"``
     - :class:`~sequince.steps.mutation.emcee.EmceeMutation`
     - Ensemble MCMC using emcee moves.
   * - BlackJAX
     - ``"blackjax"``
     - :class:`~sequince.steps.mutation.blackjax.BlackJAXMutation`
     - JAX-backed NUTS, HMC, or random-walk mutation.
   * - Eryn
     - ``"eryn"``
     - :class:`~sequince.steps.mutation.eryn.ErynMutation`
     - Fixed-dimensional ensemble MCMC using Eryn moves.
   * - Differential evolution
     - ``"differential-evolution"``
     - :class:`~sequince.steps.mutation.metropolis.DifferentialEvolutionMutation`
     - Red-blue ensemble updates using differential-evolution proposals.
   * - Red-blue Metropolis--Hastings
     - ``"red-blue-metropolis-hastings"``, ``"red-blue-mh"``, ``"red-blue"``
     - :class:`~sequince.steps.mutation.metropolis.RedBlueMetropolisHastingsMutation`
     - Custom kernels using a frozen complementary ensemble.
   * - Metropolis--Hastings
     - ``"metropolis-hastings"``
     - :class:`~sequince.steps.mutation.metropolis.MetropolisHastingsMutation`
     - User-supplied proposal kernel and transition correction.

MH steps take a proposal through the ``kernel`` argument. See
:mod:`sequince.steps.mutation.kernels` for Gaussian random walks, masked
updates, mixtures, and differential-evolution kernels. Red-blue MH requires a
:class:`~sequince.steps.mutation.kernels.ComplementaryEnsembleKernel`.

Common configuration
--------------------

Set ``n_steps`` on the mutation step to choose the number of transitions per
particle per call. For red-blue methods, one transition is a sweep over both
groups. For a target callable or ``Target`` object named ``target``:

.. code-block:: python

    from sequince.steps.mutation import MiniPCNMutation
    from sequince.transforms import AffineTransform

    mutation_step = MiniPCNMutation(
        target,
        n_steps=10,
        store_chains=True,
        transform=AffineTransform(),
    )

The implementations in the table accept these constructor options:

* ``store_chains``: store intermediate positions (default: ``False``).
* ``transform_refit``: fit the transform on every call (``"always"``, default),
  once per iteration and observation position (``"per-context"``), or once
  (``"once"``). Eager execution also accepts a callable predicate.
* ``metrics``: diagnostic objects such as ``ESJDMetric()`` or
  ``RelativeJumpDistanceMetric()``. Results appear in the output's
  ``metrics.values``; chains need not be stored.

In a custom SMC pipeline, place mutation where it should run, usually after
resampling. Wrap it in
:class:`~sequince.steps.mutation.conditional.MutateIfResampled` to run it only
when resampling occurs.

Install optional backends with ``pip install "sequince[emcee]"``,
``pip install "sequince[eryn]"``, or ``pip install "sequince[blackjax]"``.
See :doc:`../advanced_workflows/jax_support` for compiled execution.

Particle transforms
-------------------

Pass a :class:`~sequince.transforms.ParticleTransform` instance through the
step's ``transform`` argument. ``AffineTransform`` standardizes coordinates;
``WhiteningTransform`` also removes correlations. Bounds and periodic
transforms are available in :mod:`sequince.transforms`.

Configure the transform's parameters when constructing it. Mutation maps
particles into transformed coordinates and back after the transitions. With
``transform=None`` (the constructor default), it uses the original coordinates.

Auxiliary data
~~~~~~~~~~~~~~

A custom transform's ``forward`` method can return
``(values, log_det, auxiliary)``. Mutation passes ``auxiliary`` to the matching
``inverse`` call, which returns ``(particles, log_det)``. For example, a
transform taking absolute values can save each particle's signs for inversion.

Auxiliary data can be an array or an Optree-compatible pytree, such as a tuple
or dictionary. Use :func:`~sequince.utils.pytrees.pytree_dataclass` to register
immutable dataclasses with Optree and JAX.

When mutation selects a particle subset, it also selects auxiliary array leaves
with ``leaf.ndim > 0`` and ``leaf.shape[0] == n_particles``. Other leaves are
unchanged. This rule uses shape alone, so unrelated arrays of that leading size
are also selected. Particle-aligned auxiliary data works with batch-preserving
mutation and BlackJAX; emcee and Eryn reject it because their callbacks do not
expose the original walker indices.

Blocked updates and repetition
------------------------------

Use these steps to compose mutation updates:

* :class:`~sequince.steps.mutation.blocked.BlockedMetropolisHastingsMutation`:
  sequential parameter-block updates, each with its own kernel and step count.
  Block indices refer to transformed coordinates.
* :class:`~sequince.steps.mutation.repeated.RepeatedMutationStep`: split a total
  transition count across fixed rounds, with an optional step after each round.

Prebuilt samplers
-----------------

:func:`~sequince.prebuilt_samplers.make_standard_bayesian_smc` accepts a method
name from the table in ``mutation`` (default: ``"differential-evolution"``), a
transition count in ``mutation_n_steps`` (default: five), and constructor
options in ``mutation_kwargs``. Use ``mutation="none"`` or ``None`` to disable mutation.

Differential evolution requires at least four particles. Its default
``noise_scale=1e-5`` adds independent Gaussian noise in mutation coordinates.
Install ``sequince[minipcn]`` and use ``mutation="minipcn"`` to select MiniPCN
explicitly. The persistent SMC,
nested-sampling SMC, standalone MCMC, and additional-posterior-sample helpers
also default to differential evolution. Functional nested-sampling execution
requires an explicit ``mutation="minipcn"`` or ``mutation="blackjax"``.

The helper defaults to ``particle_transform="affine"``; configure it through
``particle_transform_kwargs`` or pass a transform object. With default ESS
tempering, mutation runs every iteration. With ``weighting_criterion="cess"``
and default execution flags, it runs only after resampling unless
``always_mutate=True``.

You can also pass a configured step as ``mutation``. Set its target, transform,
and options directly: the helper does not apply ``mutation_n_steps`` or
``mutation_kwargs`` to it. See :doc:`../running_inference/smc_configuration` for custom pipelines
and the API links above for method-specific options and outputs.


Adaptive mutation kernels
-------------------------

Adaptation can adjust the mutation budget or proposal parameters between
iterations using completed diagnostics. The adaptation wrappers support eager
execution only. Place conditional steps outside them, for example
``MutateIfResampled(mutation_step)``.

Acceptance-based mutation budgets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For standard and persistent SMC, set ``adaptive_mutation`` to adjust the number
of transitions from the previous iteration's acceptance rate. Named
``"minipcn"`` and ``"differential-evolution"`` mutations support this option:

.. code-block:: python

    mutation_n_steps=100,
    adaptive_mutation={
        "min_steps": 50,
        "max_steps": 1000,
        "target_accepted_moves": 5.0,
    }

``mutation_n_steps`` sets the initial budget and must lie within the bounds.
The mapping defaults to ``min_steps=0``, ``max_steps=1000``, and
``target_accepted_moves=5.0``. Omit it or set ``enabled=False`` to keep a fixed
budget.

Each mutation block uses a fixed budget; its acceptance rate determines the
next block's budget, clipped to the bounds. Accepted moves need not be
independent samples. Resume checkpoints with the same mutation configuration
and bounds. See :func:`~sequince.steps.mutation.factory.get_mutation_step` for
option details, or use
:class:`~sequince.steps.mutation.adaptive.steps.AdaptMutationSteps` in a custom
pipeline.

Differential-evolution scale weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the named ``"differential-evolution"`` mutation, set
``mutation_kwargs={"adapt_scale_weights": True}`` to adapt mixture weights
with :class:`~sequince.steps.mutation.adaptive.de.AdaptDEScaleWeights` and its
default :class:`~sequince.steps.mutation.adaptive.policies.de.ESJDScaleWeightPolicy`.
The factory consumes this flag; ``scale_multipliers`` and initial
``scale_weights`` remain constructor options. Existing defaults are unchanged,
and scale-weight adaptation defaults to ``False``.

Pass a policy instance as ``scale_weight_policy`` in ``mutation_kwargs`` to
customize adaptation, for example:

.. code-block:: python

    from sequince.steps.mutation import ESJDScaleWeightPolicy

    mutation_kwargs = {
        "adapt_scale_weights": True,
        "scale_weight_policy": ESJDScaleWeightPolicy(
            learning_rate=0.2,
            exploration_probability=0.05,
            min_attempts=10,
        ),
    }

Alternatively, specify the policy by name and pass its constructor options
separately:

.. code-block:: python

    mutation_kwargs = {
        "adapt_scale_weights": True,
        "scale_weight_policy": "esjd",
        "scale_weight_policy_kwargs": {"learning_rate": 0.2, "min_attempts": 10},
    }

``"esjd"`` is currently the supported policy name.
``scale_weight_policy_kwargs`` requires a named policy; configure policy
instances directly. Custom policies implement the
:class:`~sequince.steps.mutation.adaptive.policies.de.DEScaleWeightPolicy`
interface. Omit ``scale_weight_policy`` or set it to ``None`` to use the default
ESJD policy. Supplying a policy requires ``adapt_scale_weights=True``.

Scale-weight adaptation requires eager execution. Enable it independently or
combine it with adaptive mutation, for example through
:func:`~sequince.prebuilt_samplers.make_persistent_bayesian_smc`:

.. code-block:: python

    smc, posterior, metadata = make_persistent_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        mutation="differential-evolution",
        mutation_kwargs={
            "scale_multipliers": (0.25, 1.0, 4.0),
            "adapt_scale_weights": True,
        },
        adaptive_mutation={"target_accepted_moves": 5.0},
    )

These multipliers are illustrative. When both options are enabled, the factory
constructs ``AdaptMutationSteps(AdaptDEScaleWeights(de), ...)``: the outer
wrapper sets the sweep count and the inner wrapper adapts mixture weights.
Both decisions use completed blocks and remain fixed within each block.

MiniPCN proposal adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a custom pipeline, use
:class:`~sequince.steps.mutation.adaptive.minipcn.AdaptiveMiniPCNMutation`
to tune MiniPCN rho between iterations. It requires equal, finite weights and
a child mutation with ``adaptive=False``.

Gaussian proposal scale adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~sequince.steps.mutation.adaptive.proposal.AdaptiveGaussianProposal`
to update a scale in context metadata for a kernel factory to read.
