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

Blocked updates, repetition, and adaptation
-------------------------------------------

Use these steps to compose mutation updates:

* :class:`~sequince.steps.mutation.blocked.BlockedMetropolisHastingsMutation`:
  sequential parameter-block updates, each with its own kernel and step count.
  Block indices refer to transformed coordinates.
* :class:`~sequince.steps.mutation.repeated.RepeatedMutationStep`: split a total
  transition count across fixed rounds, with an optional step after each round.
* :class:`~sequince.steps.mutation.adaptive.steps.AdaptMutationSteps`: choose
  the next iteration's transition count from completed diagnostics.
* :class:`~sequince.steps.mutation.adaptive.minipcn.AdaptiveMiniPCNMutation`:
  tune MiniPCN rho between iterations. Requires equal, finite weights and a
  child with ``adaptive=False``.
* :class:`~sequince.steps.mutation.adaptive.de.AdaptDEScaleWeights`: tune DE
  mixture weights between iterations.
* :class:`~sequince.steps.mutation.adaptive.proposal.AdaptiveGaussianProposal`:
  update a scale in context metadata for a kernel factory to read.

The adaptation wrappers support eager execution only. Place conditional steps
outside them, for example ``MutateIfResampled(mutation_step)``.

Mixing DE with global proposals
-------------------------------

Start with a physical-space prior proposal implementing
:class:`~sequince.proposals.ProposalDistribution`: ``sample_and_log_prob`` draws
one sample per requested particle using the supplied ``rng`` and ``xp``, and
``log_prob`` evaluates one density per particle. Both methods must describe the
same distribution. Given such an object named ``prior_proposal`` and a target
named ``target``, a fixed-weight mixture is:

.. code-block:: python

    from sequince.steps import (
        DifferentialEvolutionKernel,
        IndependentProposalKernel,
        MixtureComplementaryEnsembleKernel,
        MixtureComponentMetrics,
        RedBlueMetropolisHastingsMutation,
    )

    kernel = MixtureComplementaryEnsembleKernel(
        kernels=(
            DifferentialEvolutionKernel(),
            IndependentProposalKernel(prior_proposal),
        ),
        weights=(0.8, 0.2),
    )
    mutation = RedBlueMetropolisHastingsMutation(
        target,
        kernel,
        n_steps=10,
        transform=None,
        metrics=[MixtureComponentMetrics(2)],
    )

Each active particle independently selects DE or the prior on each half-sweep.
Only selected particles are proposed by each child; DE sees the frozen
complement. Acceptance uses that child's transition correction. For the
independence proposal this is ``log q(current) - log q(proposed)``; no marginal
mixture density is evaluated. Weights are immutable, normalized, non-negative,
and finite, with positive total. Zero weights support pure-component limits.
This dispatch supports eager execution only, including eager JAX arrays; it does
not support JIT compilation.

To adapt the top-level probabilities in SMC, wrap the complete repeated block:

.. code-block:: python

    from sequince.steps import (
        AdaptMixtureWeights,
        ESJDMixtureWeightPolicy,
        RepeatedMutationStep,
    )

    adaptive_mutation = AdaptMixtureWeights(
        RepeatedMutationStep(mutation, n_rounds=2, n_steps=20),
        ESJDMixtureWeightPolicy(
            learning_rate=0.5,
            exploration_probability=0.1,
            min_attempts=20,
        ),
    )

The adapter installs component metrics automatically when absent. For each
component, ``mixture_components`` reports ``n_attempted``, ``n_accepted``,
``acceptance``, ``squared_jump_distance``, and ``esjd``. Rejections contribute zero
movement, so ESJD divides accepted squared-jump sums by *all* attempts. Scores
measure movement in mutation coordinates and are not adjusted for runtime or
coordinate scale.

Weights stay frozen throughout an outer iteration, including all repeated
rounds. The policy damps ESJD-based probabilities and assigns a positive uniform
exploration fraction; each initialized weight is at least
``exploration_probability / n_components``. Insufficient attempts in any
component, or zero movement everywhere, retain the current weights. Pending
weights take effect on entering the next outer iteration. DE's internal scale
weights remain fixed. Calls at the same context iteration consult the policy
only once; wrap all rounds whose statistics should contribute in one block.
Use ``AdaptMutationSteps(adaptive_mutation, ...)`` for budget adaptation, and
place conditional execution outside the adapters.

``mixture_weight_adaptation`` records current and pending weights. Checkpoints
retain these decisions, iteration bookkeeping, and nested child and policy
state. Restore into the same component order, kernel and proposal configuration,
and policy configuration. This is outer-iteration adaptation for SMC;
standalone MCMC requires a separate adaptation policy.

The proposal must describe the coordinates actually supplied to the kernel.
Use ``transform=None`` for physical-space prior mixing; when configuring a
prebuilt sampler by name, also set ``particle_transform=None``. Transformed
mixtures require a proposal whose sampling and density are defined in that
transformed space, including the appropriate change of variables. A
physical-space prior cannot be passed unchanged to a latent-space kernel.

The complementary-kernel factory accepts ``kernel="mixture"`` with ``kernels``
and ``weights`` in ``kernel_kwargs``. Children may be objects, names such as
``"differential-evolution"``, or mappings containing ``kernel`` and
``kernel_kwargs``. Use ``"independent-proposal"`` with a ``proposal`` object for
the global child. Ordinary ``MixtureMutationKernel`` retains its existing
whole-batch selection behavior, and MiniPCN behavior is unchanged.

For custom diagnostics, mixture auxiliary data has particle-aligned
``mixture_component`` IDs, selected-child ``correction``, and a fixed tuple of
``children``. Each child contains a ``selected`` mask and padded ``diagnostics``;
read diagnostics only under that mask. DE's scale components, multipliers and
donor indices are preserved in its diagnostic slot. DE and independence schemas
are automatic. Other children returning auxiliary data must supply
``auxiliary_templates`` with one pytree per child; leaves specify dtype and
trailing shape without a particle axis; floating leaves use the particle dtype.
Children returning no diagnostics use
``None``. This keeps both half-updates compatible even for unselected children.

Prebuilt samplers
-----------------

:func:`~sequince.prebuilt_samplers.make_standard_bayesian_smc` accepts a method
name from the table in ``mutation`` (default: ``"minipcn"``), a transition count in
``mutation_n_steps`` (default: five), and constructor options in
``mutation_kwargs``. Use ``mutation="none"`` or ``None`` to disable mutation.

The helper defaults to ``particle_transform="affine"``; configure it through
``particle_transform_kwargs`` or pass a transform object. With default ESS
tempering, mutation runs every iteration. With ``weighting_criterion="cess"``
and default execution flags, it runs only after resampling unless
``always_mutate=True``.

You can also pass a configured step as ``mutation``. Set its target, transform,
and options directly: the helper does not apply ``mutation_n_steps`` or
``mutation_kwargs`` to it. See :doc:`../running_inference/smc_configuration` for custom pipelines
and the API links above for method-specific options and outputs.
