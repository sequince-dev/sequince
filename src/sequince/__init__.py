"""Sequince: a modular Sequential Monte Carlo framework."""

import importlib.metadata
import logging

from .posterior import TemperedPosterior
from .prebuilt_samplers import make_standard_bayesian_smc
from .result import SMCResults
from .smc import SequentialMonteCarlo
from .state import ParticleState

try:
    __version__ = importlib.metadata.version("sequince")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

__all__ = [
    "ParticleState",
    "SequentialMonteCarlo",
    "SMCResults",
    "TemperedPosterior",
    "make_standard_bayesian_smc",
]
