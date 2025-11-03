Examples
========

The repository ships with a handful of executable scripts under ``examples/``.
They demonstrate how to assemble SMC pipelines for different scenarios.

.. csv-table::
   :header: "Script", "Highlights"
   :widths: 25, 75

   ``adaptive_schedule.py``, "Adaptive tempering starting from the prior distribution."
   ``adaptive_schedule_proposal.py``, "Tempering with an informative proposal distribution."
   ``gmm_transdimensional.py``, "Adaptive annealing with on-the-fly component expansion for a Gaussian mixture model."

Run any script with ``python examples/<name>.py`` after installing the project
in a virtual environment. Each script exposes constants at the top that can be
tweaked to explore different behaviours.
