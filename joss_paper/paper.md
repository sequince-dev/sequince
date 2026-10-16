---
title: 'Sequince: a modular sequential Monte Carlo toolkit'
tags:
  - Python
  - Bayesian inference
  - Sequential Monte Carlo
  - Particle filtering
authors:
- name: Michael J. Williams
  orcid: 0000-0003-2198-2974
  affiliation: "1, 2"
- name: Christian E. A. Chapman-Bird
  orcid: 0000-0002-2728-9612
  affiliation: 3
affiliations:
- name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Potsdam, Germany
  index: 1
- name: Institute of Cosmology and Gravitation, University of Portsmouth, United Kingdom
  index: 2
- name: Institute for Gravitational Wave Astronomy, University of Birmingham, United Kingdom
  index: 3
date: 15 October 2026
bibliography: paper.bib
---

# Summary

Sequential Monte Carlo (SMC) methods are a flexible family of algorithms for
approximating sequences of probability distributions using a population of weighted samples, known as particles
[@delmoral2006smc; @chopin2020introduction].
Such sequences arise naturally in applications including Bayesian inference,
and particle filtering, for example through likelihood tempering or observations arriving over time.

`sequince` is a Python toolkit for constructing SMC methods from reusable and extensible
components. It provides abstractions for defining sequences of target
distributions and for composing the operations used to evolve a particle population between them.
These include weighting, resampling, mutation, transformation, and control steps.
It is designed for researchers developing new SMC methods or adapting existing methods to scientific applications, including astrophysical inference, with checkpointing to support long-running analyses. Its core components follow the Python Array API standard, supporting NumPy, JAX, PyTorch, and CuPy arrays and enabling compatible workflows to run on CPUs or GPUs.

# Statement of need

Scientific applications often require SMC methods tailored to a particular problem. Researchers may need to combine adaptive tempering, mutation kernels, transformations, stopping rules, or multi-stage procedures, and to define problem-specific sequences of target distributions. When this logic is embedded in application-specific code, it becomes harder to compare methods, reuse components, and move prototype algorithms into stable analysis software.

`sequince` addresses these needs through a general-purpose, step-oriented framework with interchangeable algorithmic components. Researchers can start from pre-built samplers for common inference problems or construct custom pipelines with explicit control over the operations and their execution order. The same framework therefore supports both established inference methods and methodological research into new SMC variants.

Scientific models are also increasingly implemented using different numerical libraries to exploit automatic differentiation and hardware acceleration. Coupling an SMC implementation to a single array library can restrict its integration with these models. `sequince` uses the Python Array API standard [@python_array_api] to support reusable core operations across multiple numerical backends, while allowing specialised components to target particular ecosystems.

# State of the field

The Python ecosystem already contains several useful SMC-related tools. The `particles` package accompanying @chopin2020introduction provides implementations of many established SMC and particle-filtering methods and is particularly useful for teaching and methodological prototyping. `SMCPy` focuses on uncertainty quantification and Bayesian calibration using parallel likelihood-tempering SMC samplers, including fixed and adaptive tempering variants [@smcpy_repo]. `pocomc` implements persistent sampling with normalizing flows for posterior inference [@karamanis2025persistent].

Broader Bayesian software ecosystems also expose SMC functionality. `PyMC` includes a sequential Monte Carlo sampler within a general probabilistic programming framework [@abrilpla2023pymc], while `BlackJAX` provides composable inference components in the JAX ecosystem and includes SMC methods alongside other Monte Carlo algorithms [@cabezas2024blackjax]. Recent domain-specific frameworks such as SHARPy target gravitational-wave inference with JAX-accelerated SMC and gradient-based mutation kernels [@demasi2026sharpy].

`sequince` combines support for multiple SMC variants, including tempering, persistent sampling [@karamanis2025persistent], nested sampling via SMC (NS-SMC) [@salomone2025unbiased], and waste-free SMC [@dau2022wastefree], with a modular architecture for constructing bespoke algorithms. Shared interfaces allow users to reconfigure target sequences, transformations, mutation kernels, and control flow, supporting both the comparison of existing methods and the development of new variants. Support for multiple array backends and integrations with different sampling ecosystems further enable these components to be reused across scientific applications and numerical libraries. This combination makes `sequince` useful as research infrastructure for methodological work and as a foundation for application-specific downstream packages.

# Software design

`sequince` separates target distributions, particle populations, and the operations used to evolve them. The following outlines the main design choices; the documentation and runnable examples provide detailed descriptions of the interfaces and their use.

The distributions approximated during an SMC run are represented by target objects. A target evaluates named components of a probability distribution for a population of particles, while sequential targets describe how that distribution changes between stages. Separating the target from the operations that evolve the particles allows weighting and mutation components to be reused across different inference procedures. For example, tempered SMC and nested-sampling SMC use different target sequences while sharing resampling and compatible mutation components.

Target evaluations are stored as particle-aligned collections of named quantities. These can be propagated through resampling and updated during mutation, allowing previously computed quantities to be reused where valid. This is particularly useful for scientific applications in which likelihood evaluations dominate the computational cost. A `ParticleState` stores the particles, their log weights, accumulated normalising-constant information, and associated target evaluations. Specialised states provide additional information where required, such as the archive of previous populations used by persistent SMC.

An SMC algorithm is expressed as an ordered collection of steps. Each step receives the current particle state, a runtime context, and a random number generator, and returns an updated state, updated context, and structured outputs. Weighting, resampling, mutation, transformation, and control operations can therefore be implemented independently and combined into different algorithms. Higher-level steps can conditionally execute or compose other steps, keeping algorithm-specific control flow outside the runner. For example, waste-free SMC combines resampling with a mutation component that retains intermediate chain states.

A `SequentialMonteCarlo` runner repeatedly applies the configured steps. The runtime context contains the current iteration and observation, shared user-provided information, metadata used for communication between steps, and stopping controls. The runner accepts an iterable of observations, supporting both static inference problems and procedures in which targets change as new data become available. Separating particle state, runtime context, and algorithm configuration allows the execution machinery to remain common across different SMC constructions.

Each step can return structured outputs containing quantities such as effective sample sizes, tempering parameters, acceptance statistics, and other diagnostics. These outputs and the recorded particle states are collected in an `SMCResult` for subsequent analysis. New components can therefore introduce diagnostics without requiring changes to the runner or coupling their implementation to plotting and logging code. Callbacks provide logging, plotting, and other monitoring functionality during execution.

The core abstractions follow the Python Array API standard where possible [@python_array_api], allowing backend-native arrays to pass through the SMC procedure. Random number generation is similarly abstracted through `orng` [@orng]. This design supports reuse across numerical libraries while allowing specialised components to depend on a particular ecosystem. Alongside conventional eager execution, `sequince` provides a functional execution path that passes random state explicitly between steps. Compatible iterations can be compiled using facilities such as `jax.jit`, while the runner manages observations, callbacks, and result recording. Mutation integrations include ecosystem-specific packages such as `BlackJAX` [@cabezas2024blackjax].

Checkpointing is separate from result recording. Checkpoints capture the particle state, runtime progress, random-number state, and mutable step state needed to continue sampling with a reconstructed sampler configuration. Results provide the analysis record of a completed or partially completed run. Both support HDF5 storage, making numerical data and supported metadata available for inspection independently of the running process.

# Research impact statement

`sequince` is being developed as reusable infrastructure for both
methodological research and problem-specific inference workflows. Current
applications include customised sampling strategies for astrophysical inference
problems relevant to ongoing LIGO-Virgo-KAGRA analyses and future observations
with the Laser Interferometer Space Antenna (LISA) [@williams2026inprep], as well as integration with
broader Bayesian inference ecosystems such as `bilby` \[@ashton2019bilby,@bilby\] via `sequince-bilby` [@sequince_bilby]. The
same methodology is also relevant to future ground-based observatories such as
the Einstein Telescope (ET) and Cosmic Explorer (CE).

In these applications, the principal benefit of `sequince` is not a single
sampling algorithm but a shared framework in which new sequential inference
methods can be implemented, compared, and reused. The package has been used to
explore alternative tempering strategies, persistent SMC, specialised mutation
schemes, problem-specific transformations, and hardware-accelerated execution.
The common result interface also provides a consistent way to record and
compare algorithm-specific diagnostics across these different methods.

The modular design of `sequince` additionally supports downstream packages that
add method- or domain-specific functionality without requiring those
dependencies in the core library. For example, `sequince-density` [@sequince_density] provides
density-estimation components based on Gaussian mixture models and normalizing
flows. These can be combined with the core SMC machinery for learned
preconditioning, proposal construction, and other density-assisted SMC methods,
including approaches related to `pocomc` and `aspire` \[@karamanis2025persistent;
@williams2026accelerated\]. The wider sequince ecosystem also includes
`sequince-examples`, which provides runnable examples demonstrating common
workflows and more specialised uses of the package [@sequince_examples].

# AI usage disclosure

Generative AI tools, including OpenAI Codex and GitHub Copilot, were used to assist with code implementation and refactoring. ChatGPT was also used to assist with the organisation and editing of the manuscript. The authors made the core scientific and software design decisions and reviewed, edited, and validated all AI-assisted contributions. The authors retain full responsibility for the software and manuscript.

# Acknowledgements

M.J.W. acknowledges support from ST/X002225/1, ST/Y004876/1, and the University
of Portsmouth. C.E.A.C-B. is supported by UKSA Space Agency grant UKRI971.

Numerical computations were carried out on the SCIAMA high-performance
computing cluster, which is supported by the Institute of Cosmology and
Gravitation and the University of Portsmouth. Additional computations were
performed using the University of Birmingham's BlueBEAR High Performance
Computing facility.

# References
