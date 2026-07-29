Checkpointing
=============

Checkpointing records the state of a Sequential Monte Carlo run so
that it can continue in a later process. A checkpoint is captured after a
complete iteration: step processing has finished, weights have been normalised,
the new particle state has been added to the result, and callbacks have
finished. Consequently, resuming starts at the next iteration.

Configuring a checkpoint writer
-------------------------------

Pass one or more checkpoint writers to
:class:`~sequince.smc.SequentialMonteCarlo`. The ``every`` argument is measured
in completed iterations, so ``every=10`` writes after iterations 10, 20, and so
on.

.. code-block:: python

   from sequince import SequentialMonteCarlo
   from sequince.checkpoint import HDF5CheckpointWriter

   writer = HDF5CheckpointWriter(
       "run.h5",
       every=10,
       save_rng=True,
       include_history=True,
   )
   smc = SequentialMonteCarlo(
       steps=steps,
       callbacks=callbacks,
       checkpoint_writers=[writer],
   )
   result = smc.run(
       initial_state,
       observations=observations,
       rng=123,
       shared_context=shared_context,
   )

The available writers are:

``HDF5CheckpointWriter``
   Stores arrays and supported values as native HDF5 datasets, and mappings and
   sequences as groups. This format is suitable when the checkpoint should be
   inspectable or when its arrays may be useful independently. Install the
   optional dependency with ``pip install "sequince[hdf5]"``.

``PickleCheckpointWriter``
   Stores the checkpoint as one opaque pickle payload. This is the simpler
   format when inspecting the file outside Python is not important.

Both writers replace the latest runtime checkpoint. The HDF5 writer appends new
entries to the result history rather than rewriting existing entries on every
iteration. The pickle writer writes a temporary file and then replaces the
previous file.

What is saved
-------------

An :class:`~sequince.checkpoint.SMCCheckpoint` contains:

* the current :class:`~sequince.state.ParticleState`, including its particles,
  log-weights, log-evidence, and metadata;
* the next iteration and observation indices;
* mutable context metadata and recorded outputs;
* the observation from the most recently completed iteration;
* optional state supplied by each configured step;
* the random-number-generator state, unless ``save_rng=False``;
* the particle-state history, unless ``include_history=False``.

A checkpoint deliberately does not contain:

* the configured steps, callbacks, or their callable dependencies;
* immutable ``shared_context`` values;
* the complete observation sequence.

The sampler must therefore be reconstructed with the same ordered step
pipeline before loading a checkpoint. Shared inputs and observations must also
be supplied again when resuming.

Resuming a run
--------------

Load the checkpoint, reconstruct the sampler, and use
:meth:`~sequince.smc.SequentialMonteCarlo.resume_from` to prepare a runtime.

.. code-block:: python

   import h5py

   from sequince import SequentialMonteCarlo
   from sequince.checkpoint import SMCCheckpoint

   with h5py.File("run.h5", "r") as checkpoint_file:
       checkpoint = SMCCheckpoint.load_hdf5(checkpoint_file)

   smc = SequentialMonteCarlo(
       steps=steps,
       callbacks=callbacks,
       checkpoint_writers=[writer],
   )
   runtime = smc.resume_from(
       checkpoint,
       shared_context=shared_context,
   )
   result = smc.run(
       runtime=runtime,
       observations=observations,
   )

When observations were consumed before the checkpoint, ``observations`` must
be the original indexable sequence. Sequince resumes at the saved observation
index. A one-shot iterator cannot be repositioned and is therefore rejected.
Runs that do not use observations can continue with ``observations=None``.

If the checkpoint contains RNG state, it takes precedence over the ``rng``
argument to ``resume_from``. If ``save_rng=False`` was used, provide an RNG or
seed to ``resume_from`` instead:

.. code-block:: python

   runtime = smc.resume_from(
       checkpoint,
       rng=456,
       shared_context=shared_context,
   )

Once the runtime has been prepared, RNG and context arguments belong to the
runtime. Do not pass ``rng``, ``shared_context``, ``metadata``, or ``outputs``
again to ``run(runtime=...)``.

Stateful steps
--------------

Steps are stateless by default. A step that learns or accumulates information
needed by later iterations should implement ``state_dict()`` and
``load_state_dict()``. The state dictionary should contain mutable learned
state, not constructor configuration or immutable shared inputs.

.. code-block:: python

   from sequince.steps.base import SMCStep

   class AdaptiveStep(SMCStep):
       def __init__(self):
           self.scale = 1.0

       def state_dict(self):
           return {"scale": self.scale}

       def load_state_dict(self, state):
           self.scale = float(state["scale"])

       def apply(self, particle_state, context, rng):
           # Update and use self.scale here.
           return particle_state

Sequince records the index, module, and qualified class name for every step. On
resume, it validates the complete ordered pipeline before loading any step
state. This prevents state from silently being restored into a different step
configuration. Normalizing flows or adaptive proposals owned by a step should
be included in that step's state dictionary when their fitted parameters are
required by subsequent iterations.

Result history
--------------

With ``include_history=True``, resuming restores all recorded particle states.
This allows history-dependent diagnostics and plots to continue across the
checkpoint boundary. :meth:`~sequince.checkpoint.SMCCheckpoint.result` can also
reconstruct an :class:`~sequince.result.SMCResult` directly from a checkpoint.

With ``include_history=False``, only the current particle state is available.
The reconstructed result therefore starts with that state. This reduces file
size but cannot recreate plots or diagnostics that require earlier states.

HDF5 layout
-----------

The default HDF5 path is ``/checkpoint``. The latest runtime and optional
history are stored separately:

.. code-block:: text

   /checkpoint
   ├── runtime
   │   ├── version
   │   ├── next_iteration
   │   ├── next_observation_index
   │   ├── state
   │   │   ├── particles
   │   │   ├── log_weights
   │   │   ├── log_evidence
   │   │   └── metadata
   │   ├── metadata
   │   ├── outputs
   │   ├── rng_state
   │   └── step_states
   │       ├── 00000000
   │       └── 00000001
   └── history
       ├── 00000000
       ├── 00000001
       └── 00000002

String-keyed mappings become groups, lists and tuples become numbered groups,
and NumPy values and ordinary scalars become native datasets. Strings and bytes
also use datasets, while ``None`` is represented by an empty tagged group.
Unsupported values are stored as tagged pickle datasets. Mapping keys that are
not valid HDF5 path components are escaped and restored when reading. A mapping
with non-string keys uses the pickle fallback as a single value.

The ``history`` group is absent when history is disabled. A temporary runtime
group may be present if a process is interrupted during a write; a successful
write moves it into place as ``runtime``.

Loading trusted files
---------------------

Pickle can execute arbitrary code while loading. This applies both to pickle
checkpoint files and to unsupported values stored through the HDF5 pickle
fallback. Only load checkpoint files from trusted sources.
