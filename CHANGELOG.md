# Changelog

Notable changes to Sequince are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Documentation sidebar links to sequince-density and sequince-examples under
  Related Packages.

### Changed

- Add Gaussian noise to differential evolution proposals (`noise_scale=1e-5` in
  mutation coordinates) to escape collapsed ensembles.

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
