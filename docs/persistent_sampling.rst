Persistent sampling and dynamic ESS
===================================

``make_persistent_bayesian_smc(log_prior, log_likelihood, ...)`` returns
``(sampler, posterior, metadata)``. Run the sampler with a ``ParticleState``
and an explicit RNG. Persistent sampling remains eager-only: its archive grows
and each completed active batch has exactly ``n_particles`` entries. Systematic
resampling remains the default. The deterministic-mixture importance weights
and direct evidence estimator are unchanged.

Initialization
--------------

Supply an equally weighted ``ParticleState`` containing any positive multiple
of ``n_particles`` draws. Its size determines the initial population in both
static and dynamic mode. There is no ``n_prior`` or sampling callable in the
sampler API: choosing the population size and drawing it belong to the caller.
Nonmultiples are rejected, not rounded or truncated. A one-batch initial state
preserves the previous initialization behavior.

For a pocomc-style population, the caller can choose
``2 * max(int(target_ess) // n_particles, 1) * n_particles`` draws. This is an
example choice, not a sampler default. ``log_prior`` only evaluates a density.
The caller may use its own RNG (including ``orng.RandomGenerator``) for draws,
and passes a separate RNG to the runner for resampling and mutation.
For example::

    import numpy as np
    from sequince import ParticleState, make_persistent_bayesian_smc

    def log_prior(x, context=None):
        return -0.5 * np.sum(x**2, axis=1) - np.log(2 * np.pi)

    def log_likelihood(x, context=None):
        return -0.5 * np.sum((x - 1)**2, axis=1)

    sampler, posterior, metadata = make_persistent_bayesian_smc(
        log_prior, log_likelihood, n_particles=128,
        dynamic=True, target_ess=256, target_total_ess=1024,
    )
    initial = np.random.default_rng(1).normal(size=(512, 2))
    result = sampler.run(ParticleState(initial), rng=2,
                         metadata=metadata, num_steps=500)
    # Check result.context.stop_requested: a finite budget may expire first.

Draws must come from the initial target: the prior by default, or the proposal
when ``log_proposal`` is supplied. In the latter case the caller must draw from
that proposal, and the tempering increment retains the existing
``log_prior + log_likelihood - log_proposal`` correction. The sampler cannot
verify the distribution of user-supplied draws. Do not pass prior draws when
using a different initial proposal.

Initial batches are archived at beta zero without mutation or resampling.
All supplied batches are archived on the first call; adaptive tempering and
resampling to ``n_particles`` then proceed in that same iteration. Likelihood evaluations are counted when
performed; precomputed caller evaluations are not counted again. Unlike the
reference's replacement of infinite-likelihood draws, zero-likelihood prior
entries are retained, preserving their prior mass and the evidence formula.

Feedback and diagnostics
------------------------

ESS targets must be finite and positive. In dynamic mode the initial
``target_ess`` must be at least one and is truncated to an integer; static mode
retains fractional targets. ``target_total_ess`` remains fixed and may be
fractional. Bisection tolerance must be finite and positive and its iteration
limit a positive integer.

For normalized archive weights ``w`` and active batch size ``n``, the expected
number of unique entries under multinomial resampling is::

    U = sum(1 - (1 - w_i)**n)

The implementation uses ``-expm1(n * log1p(-w_i))`` with explicit handling of
unit weights. Zeros contribute zero. For initial threshold ``E``, the fixed
reference fraction is ``r = E * (1 - (1 - 1/E)**n) / n``. After weighting, let
``f = U / n``. If ``f < 0.95*r``, the next threshold is ``int(E_current/f)``;
if ``f > min(1.05*r, 1)``, it is ``int(E_current*f)`` (at least one). Otherwise
it is unchanged. The reference never changes. Updates affect the next
iteration, including when a temperature repeats. At beta one only the fixed
terminal ESS condition controls stopping.

Weighting outputs and metadata expose ``target_ess`` (the threshold actually
used). Dynamic iterations also expose ``next_target_ess``,
``expected_unique_fraction`` and ``reference_unique_fraction``.
Logging reports the actual threshold used, or the terminal ESS requirement at
beta one. Static runs do not pay for the unique-count reduction.

This statistic counts distinct archive **entries**, not independent samples,
unique coordinates, or discovered modes. Correlated mutation output and repeated
coordinates can look diverse by this criterion. Under systematic resampling it
is a diversity heuristic, not the actual expected number of distinct ancestors.
Weight ESS and a successful terminal condition do not establish mode coverage.

Costs and checkpoints
---------------------

The extra diversity reduction costs O(archive size) per adaptive iteration.
Larger initialization costs one likelihood evaluation per supplied draw unless supplied
cached, and retains more particles. The existing mixture denominator costs
O(archive size * number of archived batches) in work and temporary storage;
repeated temperatures and higher thresholds can make runs much more expensive.
Use a finite ``num_steps`` budget and inspect the stop flag.

Checkpoints include the current threshold, initial reference threshold, fixed
terminal target and archive. Initialization is atomic: an interrupted first call
must be restarted with the original draws. There is no pending population or
initialization progress to checkpoint. After the first iteration, resume uses
the archive without reevaluating the initial population. Recreate the same
configuration before resume; archive batch-size mismatches are rejected.
Old checkpoints with empty weighting state are accepted only in static mode.
Checkpoints from the earlier staged-initialization development implementation
are explicitly rejected.

Reference and intentional differences
-------------------------------------

The feedback formulas and reference fraction follow the current
`pocomc sampler source <https://github.com/minaskar/pocomc/blob/master/pocomc/sampler.py>`_
(inspected 2026-09-24, ``Sampler.__init__`` and ``_reweight``). Sequince opts in to
dynamic mode, keeps systematic resampling, its existing beta solver and evidence
formula, and does not trim archive weights. It uses backend-compatible stable
arithmetic and explicit caller-controlled initial sampling.

For very small thresholds, Sequince floors downward integer updates at one:
pocomc's unconstrained truncation can otherwise produce a zero ESS threshold.

Unlike pocomc, Sequince infers the initial population size from caller-supplied
draws and requires an exact batch multiple. It neither selects a default
population size nor rounds a requested size. The examples implement their own
population sizing; integer feedback updates match the reference.
The companion ``sequince-examples/persistent_sampling/benchmark.py`` separates
initialization effects from adaptation effects and reports incomplete runs.
