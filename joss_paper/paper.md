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
  affiliation: 1
- name: Christian E. A. Chapman-Bird
  orcid: 0000-0002-2728-9612
  affiliation: 2
affiliations:
- name: Institute of Cosmology and Gravitation, University of Portsmouth, United Kingdom
  index: 1
- name: Institute for Gravitational Wave Astronomy, University of Birmingham, United Kingdom
  index: 2
date: 18 June 2026
bibliography: paper.bib
---

# Summary

Sequential Monte Carlo (SMC) methods are a flexible family of algorithms for
approximating sequences of probability distributions using weighted particles,
resampling, and mutation steps [@delmoral2006smc; @chopin2020introduction].
Such sequences arise naturally in applications including Bayesian computation,
particle filtering, model comparison, and sequential inference, for example
through likelihood tempering or observations arriving over time.

`sequince` is a Python toolkit for constructing SMC methods from reusable
components rather than exposing only a small number of fixed algorithms. It
provides abstractions for defining sequences of target distributions and for
composing the weighting, resampling, mutation, transformation, and control
steps used to evolve a particle population between them. It is designed for
users who need to adapt SMC to new scientific problems, including
domain-specific workflows in astrophysics and other research areas.

# Statement of need

SMC is often most useful when it is customised. Researchers may need to combine
adaptive tempering, resampling strategies, mutation kernels, deterministic
transformations, stopping rules, or multi-stage procedures in ways that differ
from standard textbook pipelines. The sequence of target distributions itself
may also be problem dependent, for example when targets are constructed through
annealing or updated as new observations become available. In many projects,
this logic ends up embedded in application-specific code, making it harder to
compare methods, reuse components, and move prototype algorithms into stable
analysis software.

`sequince` addresses this gap by providing a general-purpose, step-oriented
framework for SMC. Its central abstraction is the evolution of a particle
population through a sequence of target distributions using interchangeable
algorithmic components. Users can start from pre-built samplers for common
problems, such as tempered Bayesian inference and persistent sampling, or
construct SMC pipelines manually when they require full control over weighting,
resampling, mutation, transformations, and control flow. This makes the package
suitable both for routine inference and for methodological work in which new
SMC variants are developed and tested.

The package is also designed around backend-aware core abstractions so that
particle states, target evaluations, weight updates, and standard SMC
operations can use arrays from multiple Python array libraries where possible.
In particular, `sequince` aligns these core abstractions with the Python Array
API standard [@python_array_api], while allowing specialised components to
target particular ecosystems when required. This facilitates integration with
hardware-accelerated numerical libraries that are increasingly common in
scientific computing.

# State of the field

The Python ecosystem already contains several useful SMC-related tools. The
`particles` package accompanying @chopin2020introduction provides
implementations of many established SMC and particle-filtering methods and is
particularly useful for teaching and methodological prototyping. `SMCPy`
focuses on uncertainty quantification and Bayesian calibration using parallel
likelihood-tempering SMC samplers, including fixed and adaptive tempering
variants [@smcpy_repo]. `pocomc` implements persistent sampling with
normalizing flows for posterior inference [@karamanis2025persistent].

Broader Bayesian software ecosystems also expose SMC functionality. `PyMC`
includes a sequential Monte Carlo sampler within a general probabilistic
programming framework [@abrilpla2023pymc], while `BlackJAX` provides composable
inference components in the JAX ecosystem and includes SMC methods alongside
other Monte Carlo algorithms [@cabezas2024blackjax]. Recent domain-specific
frameworks such as SHARPy target gravitational-wave inference with
JAX-accelerated SMC and gradient-based mutation kernels [@demasi2026sharpy].

`sequince` differs from these tools by prioritising a general, modular
architecture for constructing bespoke SMC algorithms. Rather than defining a
single inference procedure, it separates the target distribution, particle
state, and operations that evolve the population. Users can therefore
reconfigure the sequence of SMC operations, introduce custom target sequences
and transformations, use mutation kernels from different sampling ecosystems,
attach problem-specific diagnostics, or compose higher-level SMC procedures
from the same underlying components. This emphasis on composability is intended
to make `sequince` useful both as research infrastructure for developing new
methods and as a foundation for application-specific downstream packages.

# Software design

The distributions approximated during an SMC run are represented in `sequince`
by target objects. A target evaluates named components of a probability
distribution for a population of particles, while sequential targets describe
how that distribution changes between stages of an SMC procedure. For example,
a tempered Bayesian target constructs a sequence between a prior and posterior
using an inverse-temperature parameter, whereas other targets may depend on
observations supplied sequentially during a run.

Target evaluations are stored as particle-aligned collections of named
quantities. These evaluations can be propagated through operations such as
resampling and mutation, allowing previously computed quantities to be reused
where possible. Separating target evaluation from the SMC operations that
consume it also provides a common interface for weighting and mutation steps
across different forms of sequential inference.

Particle populations are represented by a `ParticleState`, which stores the
particles, their log weights, accumulated normalising-constant information, and
associated target evaluations. Specialised states extend this representation
where an algorithm requires additional information. For example, persistent SMC
retains particles from previous populations in a persistent particle state.

The SMC algorithm itself is expressed as an ordered collection of steps. Each
step receives the current particle state, a runtime context, and a random
number generator and returns an updated state, updated context, and structured
outputs. These outputs can contain algorithm-specific quantities such as
effective sample sizes, tempering parameters, acceptance statistics, or other
diagnostics. The runner accumulates these outputs during execution so that the
trajectory of the SMC procedure can be inspected after sampling.

The small step interface allows weighting, resampling, mutation, proposal,
transformation, and control operations to be implemented independently and
combined into different algorithms. Higher-level steps can conditionally
execute or compose other steps, allowing algorithmic control flow to be
represented within the same framework rather than being embedded directly in
the runner. This also makes it possible to reuse individual components across
otherwise different SMC constructions.

A `SequentialMonteCarlo` runner repeatedly applies the configured sequence of
steps. The runtime context records the current iteration, the current
observation when processing sequential data, shared user-provided information,
and metadata used for communication between steps. It also exposes explicit
stopping controls. The runner accepts an iterable of observations and advances
through them as the SMC procedure progresses, allowing the same execution model
to support both static target sequences and problems in which the target
changes as new data become available.

The output of a run is represented by an `SMCResult`, which separates
analysis-facing results from the mutable objects used while the sampler is
executing. In addition to the final particle state, the result records outputs
accumulated throughout the SMC procedure. This allows quantities such as
tempering schedules, effective sample sizes, mutation diagnostics, evidence
estimates, and other step-specific information to be inspected after sampling.
Because individual steps can return structured outputs, new algorithms can
extend the information recorded by a result without requiring corresponding
changes to the runner.

This distinction between runtime state and stored results is important for
methodological work. The particle state represents the current approximation to
a target distribution, the runtime context contains information required while
the sampler is executing, and the result provides the persistent record used
for subsequent analysis. In this way, diagnostics produced by experimental
weighting, mutation, or control steps can be recorded systematically without
coupling them to plotting or logging code.

The core particle-state, target, weighting, transformation, and runner
abstractions are written around the Python Array API standard where possible
[@python_array_api], allowing backend-native arrays to flow through the SMC
procedure. Random number generation is similarly abstracted through the `orng`
interface. In addition to conventional eager execution, `sequince` provides a
functional execution path in which random state is explicitly threaded through
each step. Compatible step pipelines can therefore be transformed using
facilities such as `jax.jit`, while specialised mutation methods can still
integrate with ecosystem-specific packages including `BlackJAX`.

Callbacks provide observational functionality such as logging, plotting, and
diagnostics without coupling these tasks to the SMC steps themselves.
Checkpointing is handled separately from result recording. Checkpoint writers
capture the information required to resume an analysis, including the current
particle state, runtime progress, random-number state, and mutable state
belonging to individual SMC steps. By contrast, `SMCResult` provides the
analysis product of a completed or partially completed run. Results and
checkpoints can be serialised using portable HDF5 representations, avoiding
reliance on serialising live Python objects and allowing analyses to be
inspected or resumed independently of the original process.

Together, the target, particle-state, step, context, result, and runner
abstractions provide a compact foundation from which both standard and
experimental SMC algorithms can be constructed. Pre-built configurations cover
common Bayesian workflows, while the same interfaces are used to implement more
specialised procedures such as persistent SMC and sequential inference with
observations arriving over time.

# Research impact statement

`sequince` is being developed as reusable infrastructure for both
methodological research and problem-specific inference workflows. Current
applications include customised sampling strategies for astrophysical inference
problems relevant to ongoing LIGO-Virgo-KAGRA analyses and future observations
with the Laser Interferometer Space Antenna (LISA), as well as integration with
broader Bayesian inference ecosystems such as bilby [@ashton2019bilby]. The
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
dependencies in the core library. For example, `sequince-density` provides
density-estimation components based on Gaussian mixture models and normalizing
flows. These can be combined with the core SMC machinery for learned
preconditioning, proposal construction, and other density-assisted SMC methods,
including approaches related to `pocomc` and ASPIRE \[@karamanis2025persistent;
@Williams:2025aar\]. The wider sequince ecosystem also includes
`sequince-examples`, which provides runnable examples demonstrating common
workflows and more specialised uses of the package.

# AI usage disclosure

Generative AI tools were used to assist with code implementation and
refactoring from targeted prompts. OpenAI Codex, via ChatGPT 5.3, 5.4 and 5.6,
and GitHub Copilot were used during software development. The authors reviewed,
edited, and validated all AI-assisted outputs and retain responsibility for the
design and implementation decisions.

# Acknowledgements

M.J.W. acknowledges support from ST/X002225/1, ST/Y004876/1, and the University
of Portsmouth. C.E.A.C-B. is supported by UKSA Space Agency grant UKRI971.

Numerical computations were carried out on the SCIAMA high-performance
computing cluster, which is supported by the Institute of Cosmology and
Gravitation and the University of Portsmouth. Additional computations were
performed using the University of Birmingham's BlueBEAR High Performance
Computing facility.

# References
