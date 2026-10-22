Built-in callbacks
==================

Callbacks inspect the state and accumulated result after each completed SMC
iteration. Pass a list through ``callbacks`` when constructing a pre-built
sampler or a ``SequentialMonteCarlo`` runner.

Available callbacks
-------------------

The following callbacks are available from :mod:`sequince.callbacks`:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Callback
     - Purpose
   * - :class:`~sequince.callbacks.TracePlot`
     - Plot particle coordinates across the iteration history.
   * - :class:`~sequince.callbacks.IntermediateParticlesPlot`
     - Save a corner-style plot of the current particle population.
   * - :class:`~sequince.callbacks.StatePlot`
     - Plot selected metadata histories, ESS, or the normalizing constant.
   * - :class:`~sequince.callbacks.AdaptiveMutationPlot`
     - Plot completed mutation budgets, acceptance rates, and accepted moves.
   * - :class:`~sequince.callbacks.MutationChainPlot`
     - Plot stored chains from mutation step outputs.
   * - :class:`~sequince.callbacks.TransformedMutationChainPlot`
     - Plot stored mutation chains in the active transform's coordinates.
   * - :class:`~sequince.callbacks.ParticleTransformPlot`
     - Save a corner-style plot of the current particles in transformed
       coordinates.

Configuring plot callbacks
--------------------------

For example, configure a trace plot and histories of ESS and temperature:

.. code-block:: python

    from sequince.callbacks import StatePlot, TracePlot

    callbacks = [
        TracePlot(outdir="plots", every=5),
        StatePlot(metadata_keys=["ess", "beta"], outdir="plots", every=5),
    ]

Pass this list as ``callbacks=callbacks`` to the sampler constructor.
``outdir`` selects the output directory, which defaults to the current
directory. ``every`` controls the interval between callback invocations;
callbacks run on their first invocation and then at the requested interval.

``TracePlot``, ``StatePlot``, and ``AdaptiveMutationPlot`` update their
configured ``filename``. The other plot callbacks use ``base_filename`` and
append the iteration number to save separate images.

Chain plots require mutation steps configured with ``store_chains=True``.
``MutationChainPlot`` can select a step with ``step_index``; by default it
searches all step outputs. ``AdaptiveMutationPlot`` selects the mutation output
at ``step_index=2`` by default, so adjust this for your pipeline.
Transform plots require a fitted transform in context metadata, selected by
``metadata_key``. Corner-style plots require the ``corner`` extra; see
:doc:`../../installation` for optional dependencies.

For your own reporting or run-control logic, see
:doc:`../extending_sequince/custom_callbacks`. Checkpoint writers are configured
separately; see :doc:`../advanced_workflows/checkpointing`.
