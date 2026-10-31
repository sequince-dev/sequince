# Changelog

Notable changes to Sequince are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Allow adaptive SMC to set `beta_tolerance` independently of the minimum
  beta increment, including through `make_standard_bayesian_smc`'s
  `weighting_kwargs`. The default continues to use `min_delta` as the tolerance.

## [0.1.0] - 2026-10-29

First stable release. Changes below are relative to 0.1.0b1.

### Added

- JOSS paper draft for arXiv describing Sequince v0.1.0 and its design.
  ([#46](https://github.com/sequince-dev/sequince/pull/46))
- Software archive DOIs and algorithm references for SMC variants and flow
  preconditioning. ([#143](https://github.com/sequince-dev/sequince/pull/143))
- Documentation sidebar links to sequince-density and sequince-examples under
  Related Packages. ([#140](https://github.com/sequince-dev/sequince/pull/140))

### Changed

- Add Gaussian noise to differential evolution proposals (`noise_scale=1e-5` in
  mutation coordinates) to escape collapsed ensembles.
  ([#135](https://github.com/sequince-dev/sequince/pull/135))
- Make MiniPCN optional.
  ([#142](https://github.com/sequince-dev/sequince/pull/142))
- Default all sampling helpers to differential evolution.
  ([#142](https://github.com/sequince-dev/sequince/pull/142))

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

[Unreleased]: https://github.com/sequince-dev/sequince/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sequince-dev/sequince/compare/v0.1.0b1...v0.1.0
[0.1.0b1]: https://github.com/sequince-dev/sequince/releases/tag/v0.1.0b1
