"""Sequince: a modular Sequential Monte Carlo framework."""
# ruff: noqa: E402

import importlib.metadata
import logging

from ._scipy_compat import enable_scipy_array_api

enable_scipy_array_api()

from .exceptions import DegenerateWeightsError
from .mcmc import MarkovChainMonteCarlo, MCMCResult, make_mcmc
from .posterior import TemperedPosterior
from .prebuilt_samplers import (
    make_persistent_bayesian_smc,
    make_standard_bayesian_smc,
)
from .proposals import (
    ProposalDistribution,
    TrainableProposal,
    draw_initial_state,
    draw_mcmc_initial_state,
)
from .result import SMCResult
from .smc import SequentialMonteCarlo
from .state import ParticleState, PersistentParticleState
from .target import SequentialTarget, Target, TargetEvaluation, TemperedTarget

try:
    __version__ = importlib.metadata.version("sequince")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

__all__ = [
    "DegenerateWeightsError",
    "ParticleState",
    "PersistentParticleState",
    "ProposalDistribution",
    "MCMCResult",
    "MarkovChainMonteCarlo",
    "SequentialMonteCarlo",
    "MetaSequentialMonteCarlo",
    "SMCResult",
    "SequentialTarget",
    "Target",
    "TemperedTarget",
    "TemperedPosterior",
    "TargetEvaluation",
    "TrainableProposal",
    "make_persistent_bayesian_smc",
    "make_standard_bayesian_smc",
    "make_mcmc",
    "draw_initial_state",
    "draw_mcmc_initial_state",
]
