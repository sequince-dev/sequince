"""Sequince: a modular Sequential Monte Carlo framework."""

import importlib.metadata
import logging

from .checkpoint import SMCCheckpoint
from .posterior import TemperedPosterior
from .prebuilt_samplers import make_standard_bayesian_smc
from .result import SMCResult
from .smc import SequentialMonteCarlo, SMCRuntime
from .state import ParticleState

try:
    __version__ = importlib.metadata.version("sequince")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

__all__ = [
    "ParticleState",
    "SMCCheckpoint",
    "SMCRuntime",
    "SequentialMonteCarlo",
    "MetaSequentialMonteCarlo",
    "SMCResult",
    "TemperedPosterior",
    "make_standard_bayesian_smc",
]
