Nested sampling via SMC
=======================

``make_nested_sampling_smc`` constructs an eager likelihood-constrained SMC
sampler. It repeatedly integrates a discarded likelihood shell, reweights the
survivors, resamples, and mutates within the new constrained prior. The final
shell includes all remaining live particles.

This is an experimental alternative to tempered SMC. Adaptive thresholds and
stopping do not inherit fixed-schedule unbiasedness guarantees. See
`Salomone et al. (2025) <https://doi.org/10.1093/jrsssb/qkaf015>`_ for the
method and the distinction between fixed and adaptive algorithms.

Usage
-----

Start with particles sampled from a normalized prior. Density callables must
remain fixed throughout the run; cached prior and likelihood evaluations are
reused when only the likelihood threshold changes.

.. code-block:: python

   import numpy as np
   from sequince import (
       NestedSamplingSMCResult,
       ParticleState,
       make_nested_sampling_smc,
   )

   def log_prior(x):
       return -0.5 * np.sum(x**2, axis=-1) - 0.5 * np.log(2 * np.pi)

   def log_likelihood(x):
       return -0.5 * np.sum(x**2, axis=-1)

   sampler, target, metadata = make_nested_sampling_smc(
       log_prior,
       log_likelihood,
       threshold_quantile=0.5,
       evidence_tolerance=1e-3,
       mutation="minipcn",
       mutation_n_steps=10,
       logging_frequency=0,
   )
   initial = ParticleState(np.random.default_rng(42).normal(size=(1024, 1)))
   raw = sampler.run(initial, metadata=metadata, rng=123, num_steps=100)
   result = NestedSamplingSMCResult(raw)
   assert result.complete
   print(result.log_evidence)  # analytic answer: -0.5 * log(2)
   samples, weights = result.posterior_samples()
   draws, _ = result.posterior_samples(
       method="multinomial", n_samples=1000, rng=456
   )

The raw final state represents a constrained prior, **not the posterior**.
Its ``log_normalizing_constant`` (including the ``log_evidence`` alias) tracks
remaining prior mass. Use the wrapper's ``log_evidence`` and
``posterior_samples`` for evidence and posterior inference. Posterior weights
combine particles from every discarded shell and the final live shell.
They do not imply independent samples: resampling and mutation introduce
correlations. The ordinary SMC normalizing-constant uncertainty methods do
not estimate uncertainty in the shell evidence; use independent runs.

Stopping and schedules
----------------------

The default schedule uses the weighted median of the current likelihoods.
``threshold_quantile`` selects a different discarded fraction; ``"worst"``
retains the prototype's minimum-likelihood schedule. Mutating the entire
population after discarding one particle can be expensive.

A sequence supplied as ``threshold_schedule`` contains **log likelihoods**.
The sampler integrates the remaining population when the sequence ends.
For fixed-schedule experiments, set ``evidence_tolerance=None`` to disable
adaptive evidence stopping. A callable accepts likelihood values and optionally
``context`` or a second argument named ``state``, or all three as
``(values, state, context)``. Raising ``StopIteration`` finalizes the run.
Exceptions raised by a schedule otherwise propagate unchanged.

``evidence_tolerance`` compares the estimated remaining contribution
``X * mean(L)`` with the accumulated evidence. It is a heuristic, not a
bound on missed modes. If a global likelihood maximum is known, supply
``log_likelihood_upper_bound`` to use ``X * Lmax`` instead. This protects the
stopping decision against an undiscovered spike, although X is itself estimated.
``min_log_prior_volume`` supplies an alternative
stopping limit. Both paths integrate the remaining contribution before stopping.

Likelihood ties are discarded together under a strict constraint. A sampled
plateau with no survivors is integrated as a terminal shell; this handles
constant likelihoods but cannot distinguish a true plateau from population
collapse or an undiscovered higher-likelihood region. This implementation does
not implement the auxiliary tie-breaking variables of the paper's general
adaptive algorithm. Inclusive thresholds are supported for fixed schedules;
adaptive schedules require strict constraints to make progress.

Reaching ``run(num_steps=...)`` pauses execution without integrating the live
population. The wrapper reports ``complete=False`` and refuses posterior
reconstruction, preventing a truncated result from being mistaken for a complete
posterior. Normal sequince checkpoint writers and ``resume_from`` preserve the
shell archive and runtime metadata. Resume a partial run with the same model
and sampler configuration. Completed runs must not be resumed.

Storage and execution
---------------------

``store_shell_particles=False`` disables posterior reconstruction and retains
only evidence diagnostics in the shell outputs. The standard runner still
retains its usual state history. Results can be saved with ``raw.save`` and
loaded with ``SMCResult.load``, then wrapped again. This portable result format
is for analysis; use checkpoint writers for resuming sampling.

NumPy, Torch, and JAX arrays are supported in eager execution. Adaptive quantile
selection transfers likelihoods and weights to NumPy on the host, so GPU
execution includes synchronization. Compiled/functional NS-SMC execution is
explicitly unsupported.

Comparison benchmark
--------------------

``sequince-examples/ns-smc/benchmark.py`` compares the method with current tempered SMC on
three two-dimensional problems with analytic evidence and posterior moments:
a Gaussian, a separated Gaussian mixture, and a spike-and-slab Gaussian mixture.
Both methods start from the same normal-prior draws. The script reports each
seed and aggregate errors, runtime, and actual likelihood evaluations.

.. code-block:: bash

   pixi run python ../sequince-examples/ns-smc/benchmark.py --seeds 20
   pixi run python ../sequince-examples/ns-smc/benchmark.py --seeds 20 --tempered-mutation-steps 40

The second command gives tempered SMC more mutation steps to bring its likelihood
cost closer to NS-SMC. Equal population size and mutation length alone do not
imply equal computational budgets. Larger populations, more seeds, and realistic
likelihoods are needed before choosing a default for an application.

Initial measurements (2026-09-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Twenty seeds (0 through 19), 512 particles, a normal prior with standard deviation
5 in each coordinate, MiniPCN mutation with affine preconditioning, and default
systematic resampling gave the following log-evidence RMSE. NS-SMC used a median
threshold and remaining-evidence tolerance of 0.001. Columns show both error and
mean likelihood evaluations, including rejected proposals.

.. list-table:: Log-evidence RMSE (mean likelihood evaluations)
   :header-rows: 1

   * - Target
     - Tempered, 10 mutation steps
     - NS-SMC, 10 mutation steps
     - Tempered, 40 mutation steps
   * - Gaussian
     - 0.0592 (30,618)
     - 0.0717 (92,672)
     - 0.0574 (108,032)
   * - Separated mixture
     - 0.0554 (37,376)
     - 0.0944 (100,659)
     - 0.0696 (129,536)
   * - Spike-and-slab
     - 0.1709 (30,925)
     - 0.0816 (129,536)
     - 0.0660 (108,032)

At equal mutation length NS-SMC improved spike-and-slab evidence accuracy but
spent about four times as many likelihood evaluations. Giving tempered SMC more
mutation steps removed that advantage in this experiment. NS-SMC improved some
posterior moment estimates: spike-and-slab mean RMSE was 0.0235 versus 0.0603 for
the 40-step tempered run, and relative second-moment RMSE was 0.0873 versus 0.1160.
The separated mixture's posterior mean error was worse with NS-SMC.

These small experiments support retaining NS-SMC as an experimental option, not
replacing tempered SMC as the default. The budgets are approximate, not identical;
20 seeds give limited precision, and the examples are only two-dimensional.
The scalar analytic likelihoods are cheap, so their wall times are dominated by
framework overhead and are not representative of expensive scientific models.
No general efficiency or unbiasedness claim follows from these results.

Phase transitions and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-dimensional benchmarks above do not test the nested path's advantage
at a phase transition. Likelihood-contour compression avoids the need to cross
the same temperature-annealing bottleneck; practical performance still depends
on sampling each constrained prior effectively. A finite-budget comparison with
MiniPCN does not refute that path advantage. See
`Yallup and Handley, Appendix J <https://arxiv.org/html/2609.29413v1#A10>`_.
The examples save path diagnostics for further investigation, but a dedicated
study should vary dimension and transition severity and examine mode occupancy,
mutation mixing, and repeated-run evidence errors.

A dedicated runnable demonstration is now provided in
``sequince-examples/ns-smc/run_phase_transition.py``. It uses the 10D
uniform-ball example from Salomone et al., section 6.1, and compares the same
coordinate random-walk proposal on both paths, plus an exact constrained-prior
control. Its Figure 1(a)-style plot shows the analytic likelihood--volume curve;
a separate plot reports the actual finite-budget sampler trajectories. Run
with ``--plot-paths-only`` to regenerate just the reference-path PNG and PDF.
Unlike the easy benchmarks, this example supplies a known likelihood maximum
for remaining-evidence stopping, avoiding reliance on a potentially undiscovered
spike in the current live population.
