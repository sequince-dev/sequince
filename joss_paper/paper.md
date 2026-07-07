---
title: 'Sequince: a modular sequential Monte Carlo toolkit for extensible sequential inference'
tags:
  - Python
  - Bayesian inference
  - Sequential Monte Carlo
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
They are used in Bayesian computation, particle filtering, model comparison,
and other sequential inference problems. `sequince` is a Python toolkit for
constructing these methods from reusable components rather than exposing only a
small number of fixed algorithms. It is designed for users who need to adapt
SMC to new scientific problems, including domain-specific workflows in
astrophysics and other research areas.

# Statement of need

SMC is often most useful when it is customised. Researchers may need to combine
adaptive tempering, resampling strategies, mutation kernels, deterministic
transformations, stopping rules, or nested procedures in ways that differ from
standard textbook pipelines. In many projects this logic ends up embedded in
application-specific code, which makes it harder to compare methods, reuse
components, and move prototype algorithms into stable analysis software.

`sequince` addresses this gap by providing a general-purpose, step-oriented SMC
implementation. Its main contribution is a software architecture for composing
SMC pipelines from interchangeable building blocks. Users can start from a
pre-built Bayesian recipe, or construct samplers manually when they need full
control over weighting, resampling, mutation, and control flow. This makes the
package suitable both for routine sequential inference and for methodological
work where new SMC variants are being developed and tested. The package is also
designed around backend-aware core abstractions so that particle states, weight
updates, and standard SMC workflows can operate on arrays from multiple Python
array libraries where possible. In particular, `sequince` aims to align these
core abstractions with the Python Array API standard [@python_array_api], while
still allowing specialised components to target particular ecosystems when
needed.

# State of the field

The Python ecosystem already contains several useful SMC-related tools. The
`particles` package accompanying @chopin2020introduction is valuable for
teaching and prototyping standard methods. `SMCPy` focuses on uncertainty
quantification and Bayesian calibration with parallel likelihood-tempering SMC
samplers, including fixed and adaptive tempering variants [@smcpy_repo].
`pocomc` focuses on persistent sampling for posterior inference
[@karamanis2025persistent]. Broader Bayesian software ecosystems also expose
SMC functionality. `PyMC` includes a sequential Monte Carlo sampler within a
general probabilistic programming framework [@abrilpla2023pymc], while
`BlackJAX` provides composable inference components in the JAX ecosystem and
includes SMC methods alongside other Monte Carlo algorithms
[@cabezas2024blackjax]. Recent domain-specific frameworks such as SHARPy
target gravitational-wave inference with JAX-accelerated SMC and
gradient-based mutation kernels [@demasi2026sharpy].

`sequince` differs from these tools by prioritising a general, modular
architecture for building bespoke SMC workflows. In particular, it is aimed at
problems where users want to reconfigure the sequence of SMC operations, attach
custom diagnostics, couple SMC to problem-specific transformations, or embed
one SMC procedure inside another. This emphasis on composability is intended to
make the package useful both as research infrastructure and as a foundation for
application-specific downstream packages.

# Software design

The core object in `sequince` is a runner that executes an ordered sequence of
steps on a particle state. Each step receives the current particles, a runtime
context, and a random number generator, then returns an updated state. This
keeps the interface between SMC components small and explicit, which makes it
straightforward to implement custom weighting, resampling, mutation, proposal,
control, or transformation steps.

The runtime context carries the current iteration, optional observations,
shared user-provided configuration, mutable metadata for communication between
steps, and recorded outputs for later inspection. This design supports both
static posterior inference and genuinely sequential settings in which new data
arrive over time. It also allows algorithm-specific state, such as temperatures
or diagnostic summaries, to be passed through the SMC pipeline without
hard-coding that logic into the runner itself.

The core particle-state, weighting, transform, and runner abstractions are
written around the Python Array API standard where possible
[@python_array_api], allowing backend-native arrays to flow through the main
SMC loop. This support is deliberately layered: generic components aim to
preserve backend compatibility, while some specialised mutation steps and
proposal mechanisms remain library-specific because they depend on external
packages with narrower execution models. Backend-aware execution also extends to
random number generation through the `orng` interface used across the main SMC
steps, helping preserve a common runtime model across array backends.

Callbacks are executed at the end of each iteration and provide functionality
such as logging, plotting, diagnostics, and checkpointing. Keeping these tasks
separate from the SMC steps helps users inspect or persist a run without
entangling analysis-side concerns with the core inference algorithm. Together,
the runner, step interface, context object, and callback system provide a
compact foundation for both standard workflows and more experimental SMC
designs.

# Research impact statement

`sequince` is being used as reusable infrastructure for problem-specific
inference workflows. Current applications include the development of customised
sampling strategies for astrophysical inference problems related to the Laser
Interferometer Space Antenna (LISA), as well as integration work with broader
Bayesian inference ecosystems such as `bilby` [@ashton2019bilby]. In these
settings, the main benefit of `sequince` is not a single fixed algorithm, but a
shared framework for expressing and testing new sequential inference ideas. This
includes companion-package development such as `sequince-flows`, which extends
the ecosystem with flow-based components needed for normalizing-flow-assisted
analyses in the style of `aspire`.

# AI usage disclosure

Generative AI tools were used to assist with code refactoring and rewriting
from targeted prompts. OpenAI Codex, via ChatGPT 5.3 and 5.4, and GitHub
Copilot were used during software development. The authors reviewed, edited,
and validated all AI-assisted outputs and retained responsibility for the
design and implementation decisions.

# Acknowledgements

M.J.W. acknowledges support from ST/X002225/1, ST/Y004876/1, and the
University of Portsmouth. Numerical computations were carried out on the SCIAMA
high-performance computing cluster, which is supported by the Institute of
Cosmology and Gravitation and the University of Portsmouth.

# References
