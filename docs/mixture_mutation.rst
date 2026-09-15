Randomly interleaved mutation
=============================

``MixtureMutation`` selects one named child for the entire population at every
transition. The selected child executes exactly one transition, or one complete
red-blue sweep. ``n_steps`` is the total transition budget across all children.
Weights are relative, finite, nonnegative numbers in mapping order; at least one
must be positive. Child defaults such as ``n_steps=100`` do not affect this budget.

Fixed weights and adaptive length
---------------------------------

.. code-block:: python

   from sequince.steps import MixtureMutation, AdaptMutationSteps
   from sequince.steps.mutation import (
       GaussianRandomWalk, MetropolisHastingsMutation,
   )
   from sequince.steps.mutation.adaptive.policies.acceptance import (
       AcceptanceRatePolicy,
   )

   # log_target returns one log density per particle.
   mixture = MixtureMutation(
       {
           "small": MetropolisHastingsMutation(
               log_target, GaussianRandomWalk(scale=0.2)),
           "large": MetropolisHastingsMutation(
               log_target, GaussianRandomWalk(scale=1.5)),
       },
       weights=[0.7, 0.3],
       n_steps=10,
   )
   adaptive_length = AdaptMutationSteps(
       mixture,
       AcceptanceRatePolicy(3, min_steps=2, max_steps=50),
       initial_steps=10, min_steps=2, max_steps=50,
   )

Use ``mixture(state, context, rng)`` for the configured length, or
``mixture.apply_with_n_steps(state, context, rng, n_steps=20, weights=[1, 1])``
for an explicit budget and optional weight override. Overrides never change the
configured defaults. Zero steps return unchanged particles and context, zero
counts and displacements, and entries for every child. Unselected children have
empty custom metric results because no metric state was initialized.

Joint adaptation
----------------

.. code-block:: python

   from sequince.steps import AdaptMoveWeights, JumpDistancePerEvaluation

   adaptive_weights = AdaptMoveWeights(
       mixture,
       JumpDistancePerEvaluation(
           min_probability=0.05,
           score_smoothing=0.3,
           probability_smoothing=0.3,
           evaluation_costs={"log_target_evaluations": 1.0},
       ),
   )
   joint = AdaptMutationSteps(
       adaptive_weights,
       AcceptanceRatePolicy(3, min_steps=2, max_steps=50),
       initial_steps=10, min_steps=2, max_steps=50,
   )

Both wrappers freeze decisions per ``StepContext.iteration``. The first completed
block in an iteration proposes the next iteration's decision; repeated calls
reuse current decisions and do not update either policy again. Child outputs
pass through both wrappers unchanged. There is no trajectory-dependent stopping.
Between-iteration adaptation is intentional. Target invariance of individual
frozen moves does not by itself establish finite-sample exactness of the adaptive
algorithm.

A custom ``MoveWeightPolicy`` implements ``initialise(initial_weights=...,
context=...)`` and ``update(current_weights=..., output=..., context=...)`` and
returns relative weights in child order. Decisions are validated before use.
Stateful policies must also implement ``state_dict`` and ``load_state_dict``.
Do not reuse a stateful policy instance across different mixtures.

Supported children and preparation
----------------------------------

The initial implementation supports these concrete classes:

* ``MetropolisHastingsMutation`` with a persistent kernel instance.
* ``RedBlueMetropolisHastingsMutation``.
* ``DifferentialEvolutionMutation``.

Children must share the identical ``Target`` object or log-density callable.
Proposals must preserve the target, population size, and particle correspondence.
Asymmetric kernels must supply the existing MH transition correction. Custom
kernels must not adapt to the trajectory internally or hold mutable sampling
state needing a checkpoint. Each child must be a distinct mutation instance.

Transforms and kernel factories are rejected because their per-call preparation
can refit from the evolving population. Mutation wrappers and other child classes
are also rejected. Emcee, Eryn, and MiniPCN reconstruct samplers on application;
BlackJAX prepares its algorithm on application. Supporting these children needs
a separate preparation lifecycle audit and implementation. No new preparation
hook is introduced for the currently supported children, which do not warm up
or reconstruct their persistent kernels.

Execution is eager only, including eager NumPy, JAX and Torch arrays. Functional
execution raises ``NotImplementedError``. Selection and diagnostics synchronize
host scalars; this implementation does not promise compiled-loop performance.
The mixture does not retain full chains; any child ``store_chains`` output is
discarded, so leave it disabled to avoid unnecessary work.

Diagnostics and costs
---------------------

``MixtureMutationOutput`` provides:

* ``metrics``: a ``MutationMetricsOutput`` with aggregate transitions, attempts,
  and accepted counts. Acceptance is total accepted divided by total attempted.
* ``moves``: names mapped to ``MoveDiagnostics``, containing selection counts,
  independent child metrics, evaluation counts, and scaled squared displacement.
* ``weights``: normalized probabilities used for this block.
* ``evaluation_counts``: summed work attributed to children.
* ``coordinate_scale``: the common coordinate scaling frozen at block entry.

Custom metric state stays separate per child and is finalized once for each
selected child. Custom metric values are never summed. A metric sees that child's
selected transitions; other moves can change particles between its updates.
Use the mixture displacement diagnostic for comparable local jumps.

Displacement is measured immediately before and after each selected move in
original particle coordinates. It sums squared coordinate differences divided
by the initial unweighted population variance over all particles and dimensions.
Constant or nonfinite variance uses unit scale in that dimension. This fallback
is finite but depends on coordinate units. Nonfinite jumps remain visible in raw
diagnostics and are ignored by the heuristic. Empty populations are rejected.
Only scalar summaries are accumulated, with no extra target evaluations.

Costs are differences of existing context counters around child execution.
Callable targets currently evaluate both the current and proposed population
on each selected one-step call; that overhead is counted. ``Target`` caches may
avoid repeated current evaluations. The mixture adds no counts to the context
itself, so work is not counted twice.

The heuristic smooths each move's summed scaled squared displacement divided by
its weighted evaluation cost, normalizes scores, smooths probabilities, and
applies ``floor + (1 - n_moves * floor) * probability``. The floor must be feasible.
Unselected moves, insufficient selections, zero cost, and nonfinite measurements
retain old scores. Probabilities stay unchanged until every move has an estimate,
and when all scores are zero. A measured zero jump updates its score toward zero.
This is an opt-in heuristic, not a guarantee of improved mixing or efficiency.

Cost coefficients accept the existing target, likelihood, proposal, and increment
counter names. Unspecified counters cost zero. Target and likelihood counters can
overlap: choose coefficients to reflect actual computational work. There is no
gradient counter in the current evaluation interface, and gradient-based children
are unsupported. Raw counters are always retained for independent analysis.

Checkpoints and reproducible comparison
---------------------------------------

Mixture checkpoints forward each child's state and validate names and order.
Adaptive checkpoints additionally save current/next probabilities, iteration
markers and policy scores. Restore with the same child and policy configuration;
name/order validation cannot detect changed target functions or kernel settings.
Use the existing runner checkpoint mechanism to also save particle state, context,
and RNG. Mutation state dictionaries alone do not save random state.

Run the reproducible standard-normal comparison:

.. code-block:: bash

   pixi run --frozen python examples/mixture_mutation.py

It compares fixed weights, adaptive length and joint adaptation using 8,000
particles, eight iterations, and seed 731. It includes an asymmetric independent
proposal with its MH correction, reports mean and variance against the known
values 0 and 1, and reports actual target evaluation counts. A single target and
seed illustrate behavior; they do not establish a general efficiency advantage.
