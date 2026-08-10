"""Sequince: a modular Sequential Monte Carlo framework."""
# ruff: noqa: E402

import importlib.metadata
import logging

from ._scipy_compat import enable_scipy_array_api

enable_scipy_array_api()

from .exceptions import DegenerateWeightsError
from .mcmc import MarkovChainMonteCarlo, MCMCResult, make_mcmc
from .posterior import TemperedPosterior
from .prebuilt_samplers import make_standard_bayesian_smc
from .proposals import draw_initial_state, draw_mcmc_initial_state
from .result import SMCResult
from .smc import SequentialMonteCarlo
from .state import ParticleState
from .target import SequentialTarget, TargetEvaluation, TemperedTarget

try:
    __version__ = importlib.metadata.version("sequince")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

__all__ = [
    "DegenerateWeightsError",
    "ParticleState",
    "MCMCResult",
    "MarkovChainMonteCarlo",
    "SequentialMonteCarlo",
    "MetaSequentialMonteCarlo",
    "SMCResult",
    "SequentialTarget",
    "TemperedTarget",
    "TemperedPosterior",
    "TargetEvaluation",
    "make_standard_bayesian_smc",
    "make_mcmc",
    "draw_initial_state",
    "draw_mcmc_initial_state",
]
