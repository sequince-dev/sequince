# Changelog

Notable changes to Sequince are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Independence Metropolis-Hastings proposals using `IndependentProposalKernel`
  and `ProposalDistribution`, with cached proposed log densities. Proposals must
  describe mutation coordinates; physical-space priors require physical-space
  mutation.

## [0.1.0b1] - 2026-10-23

First public beta release.

### Added

- Composable Sequential Monte Carlo pipelines for Bayesian inference and
  particle filtering.
- Prebuilt samplers for standard Bayesian SMC, persistent Bayesian SMC, and
  nested-sampling SMC.
- Array API support for NumPy, JAX, PyTorch, and CuPy backends.
- Posterior sampling, evidence estimates, and diagnostics.
- Checkpoint saving and resuming.

[Unreleased]: https://github.com/sequince-dev/sequince/compare/v0.1.0b1...HEAD
[0.1.0b1]: https://github.com/sequince-dev/sequince/releases/tag/v0.1.0b1
