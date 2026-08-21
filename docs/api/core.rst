Core modules
============

Top-level API
-------------

.. automodule:: sequince
   :members:
   :undoc-members:
   :show-inheritance:

Sequential Monte Carlo driver
-----------------------------

Reconstruct a runner with ``SequentialMonteCarlo.from_config(smc.config)``.
This preserves runner settings and shares the referenced steps, callbacks, and
checkpoint writers. It does not clone mutable component state or resume a run;
use the checkpoint APIs to restore runtime state.

.. automodule:: sequince.smc
   :members:
   :undoc-members:
   :show-inheritance:

Markov chain Monte Carlo driver
-------------------------------

.. automodule:: sequince.mcmc
   :members:
   :undoc-members:
   :show-inheritance:

Particle state and results
--------------------------

.. automodule:: sequince.state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sequince.result
   :members:
   :undoc-members:
   :show-inheritance:

Particle transforms
-------------------

.. automodule:: sequince.transforms
   :members:
   :undoc-members:
   :show-inheritance:

Checkpointing
-------------

.. automodule:: sequince.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

Posterior utilities
-------------------

.. automodule:: sequince.posterior
   :members:
   :undoc-members:
   :show-inheritance:

Pre-built samplers
------------------

.. automodule:: sequince.prebuilt_samplers
   :members:
   :undoc-members:
   :show-inheritance:
