"""Sequince: a modular Sequential Monte Carlo framework."""

import importlib.metadata
import logging

from .posterior import PosteriorEvaluation, TemperedPosterior
from .prebuilt_samplers import make_standard_bayesian_smc
from .proposals import draw_initial_state
from .result import SMCResult
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
    "MetaSequentialMonteCarlo",
    "SMCResult",
    "TemperedPosterior",
    "PosteriorEvaluation",
    "make_standard_bayesian_smc",
    "draw_initial_state",
]
