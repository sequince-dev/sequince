Pyro NUTS mutation
==================

Install ``pip install 'sequince[pyro]'``. The ``pyro`` extra adds ``pyro-ppl``;
Torch is supplied by Pyro or the existing CPU/CUDA Pixi backend environment.
Importing Sequince does not import Pyro or require it to be installed.
``PyroNUTSMutation`` is exported from ``sequince.steps`` and
``sequince.steps.mutation``. Sampler factories accept ``mutation="pyro-nuts"``.

``PyroNUTSMutation(target, n_steps=1, step_size=0.1, max_tree_depth=10,
use_multinomial_sampling=True)`` consumes the existing Target or scalar
log-density callback. ``n_steps`` means complete NUTS transitions per particle,
not leapfrog steps. The callback receives one vector; Target component callbacks
must also support the final population evaluation. Positions must be floating
Torch tensors of shape ``(n_particles, n_dimensions)`` on CPU or CUDA.
CPU float32 and float64 are covered by tests. CUDA tests are conditional on
hardware availability; a CPU-only run does not validate CUDA.

Target and coordinates
----------------------

Sequince does not impose prior-unit coordinates. The caller's Target defines
the density in the stored particle coordinates. With ``TemperedPosterior`` and
no proposal override, that density is ``log_prior + beta * log_likelihood``;
``beta`` lives in ``StepContext.metadata``. When a proposal is configured, the
existing proposal-to-posterior bridge is preserved instead. The kernel wraps
that same target evaluation, without adding another prior or reconstructing
the tempering formula. Every trajectory position receives a fresh evaluation.
Final cached components are recomputed through the normal Target path.

An optional existing ``ParticleTransform`` maps particles to mutation
coordinates with ``forward``. Its ``inverse`` returns particles and
``log|dx/dz|``. The potential is the negative of the existing target plus this
inverse Jacobian, included exactly once. The transform is fitted before the
first particle moves and then held fixed. With prior-unit coordinates, the
caller must supply the density appropriate to those coordinates; this kernel
does not add a physical-space prior automatically.

All target operations must preserve Torch autograd. Python/NumPy scalar targets,
detached tensors, non-scalar single-particle targets, and changed output device
or dtype are rejected. Initial potentials and gradients are validated for every
particle. Transitions explicitly enable gradients even inside ``torch.no_grad``.
``torch.inference_mode`` is not supported. Use differentiable bijections and
continuous parameters only. Sequince's generic Target interface has no parameter
schema: floating-point encodings of discrete parameters, partial detachments,
stochastic callbacks, and incorrect user Jacobians cannot in general be detected
automatically. Users must exclude these components. Known periodic wrapping
transforms are rejected, including inside composite transforms. A zero derivative at a
stationary point is valid and is not evidence of a missing gradient.

Execution and diagnostics
-------------------------

Particles are processed independently and sequentially, with a new Pyro kernel
for each particle. Neither multiprocessing nor a joint population chain is
used. Step size, identity mass matrix, context metadata (including beta), and
transform are fixed throughout a call. Step-size adaptation, mass adaptation,
and warmup are disabled. User callbacks must not mutate their target, transform,
or context. No diagnostic from an earlier particle tunes a later particle.

The result preserves particle order, shape, dtype/device, weights, ancestry,
evidence, and unrelated state metadata, using the existing state replacement
method. Final particles and cached components do not retain trajectory graphs.
``store_chains=True`` includes the complete chain in particle coordinates.
The implementation retains latent trajectories during a call to update existing
population metrics; memory scales with particles times transitions times
dimensions. Existing repeated-mutation and zero-step block protocols work.
The fitted transform uses the existing mutation checkpoint state convention.

Diagnostics are returned in the usual step output, with ``metrics`` finalized
at block boundaries. Fields include ``n_particles``, ``steps``, ``step_size``,
``max_tree_depth``, ``divergences`` (total divergent transitions), and
``pyro_acceptance_rate`` (mean of Pyro's public per-chain "acceptance rate").
That Pyro statistic counts transitions in which its NUTS tree-selection logic
accepts a proposal; it is not mean acceptance probability or a conventional
single-proposal Metropolis rate. Sequince's ``metrics.acceptance`` instead
measures the fraction of particles that move. Tree depth actually reached,
leapfrog counts, and mean acceptance probabilities are omitted because the
public kernel diagnostics do not expose them. Actual target work is accumulated
in ``StepContext.evaluation_counts``, including validation and final refresh.

An ORNG generator, NumPy generator, integer seed, or None is accepted through
Sequince's ``resolve_rng``. One seed is drawn per nonempty mutation block, and
the resulting Torch stream advances across particles. CPU and the active CUDA
device generators are saved and restored by ``torch.random.fork_rng``, including
on exceptions. Other CUDA devices are not seeded. Python and NumPy global RNGs
are untouched. This narrow use of global Torch state is not thread safe; do not
run concurrent mutations or unrelated random draws in the same process.
Reproducibility is within the same backend/device/software configuration, not
across CPU and CUDA or Pyro versions.

Examples
--------

Run the Gaussian example with ``python docs/examples/pyro_gaussian.py``.

.. literalinclude:: examples/pyro_gaussian.py
   :language: python

Bilby v3 Torch example
~~~~~~~~~~~~~~~~~~~~~~

The inspected checkouts contain no dedicated Sequince--Bilby adapter. The example
uses existing ``TemperedPosterior`` callbacks, passing tensor views in parameter
dictionaries directly to Bilby's public prior and likelihood APIs. Particles
are physical parameters here, so no additional coordinate Jacobian is needed.
The mutation itself neither imports Bilby nor handles its parameter conversion.

Install the local Bilby development checkout in editable mode. In a fresh
process, run::

    BILBY_ARRAY_API=1 SCIPY_ARRAY_API=1 python docs/examples/pyro_bilby.py

These environment variables activate backend inference at import time;
``bilby.compat.utils.array_module`` obtains the namespace from Torch inputs.
This checkout has no global Torch backend setter. Tests use separate processes
to leave the parent's backend and environment unchanged. The integration test
requires the sibling ``external/bilby`` checkout and asserts its resolved import
path; it skips when that optional checkout is absent.

.. literalinclude:: examples/pyro_bilby.py
   :language: python

A differentiable Torch waveform likelihood can replace ``TorchLikelihood``
without changing the mutation API. Its conversions, constraints, waveform
implementation and cache behavior still require independent gradient checks.
In particular, Bilby's interpolation-dependent priors are not generally Torch
compatible, and hard/discrete constraints are outside this implementation's
supported smooth targets. No full GW likelihood or waveform cache path is
validated by the toy example. Non-Torch Bilby configuration raises a clear
error in the example's target boundary instead of converting a NumPy target.

Efficient parallel particle-wise NUTS would require independent per-particle
momenta, tree building, stopping rules, proposal selection and RNG streams,
plus batched differentiable target evaluation. A common frozen tuning estimate
could be computed from the pre-mutation population in a future implementation.
Pyro multiprocessing or one joint NUTS trajectory over all particles does not
provide these semantics.
