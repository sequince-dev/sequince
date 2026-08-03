"""Sequince: a modular Sequential Monte Carlo framework."""

import importlib.metadata
import logging

from .posterior import TemperedPosterior
from .prebuilt_samplers import make_standard_bayesian_smc
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
    "ParticleState",
    "SequentialMonteCarlo",
    "MetaSequentialMonteCarlo",
    "SMCResult",
    "SequentialTarget",
    "TemperedTarget",
    "TemperedPosterior",
    "TargetEvaluation",
    "make_standard_bayesian_smc",
]
