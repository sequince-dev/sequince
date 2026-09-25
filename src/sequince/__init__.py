"""Sequince: a modular Sequential Monte Carlo framework."""
# ruff: noqa: E402

import importlib.metadata
import logging

from .array_api import enable_scipy_array_api

enable_scipy_array_api()

from .density import SequentialDensity, TemperedDensity
from .exceptions import DegenerateWeightsError
from .mcmc import MarkovChainMonteCarlo, MCMCResult, make_mcmc
from .posterior import TemperedPosterior
from .prebuilt_samplers import (
    make_nested_sampling_smc,
    make_persistent_bayesian_smc,
    make_standard_bayesian_smc,
)
from .proposals import (
    ProposalDistribution,
    TrainableProposal,
    draw_initial_state,
    draw_mcmc_initial_state,
)
from .result import NestedSamplingLevel, NestedSamplingSMCResult, SMCResult
from .smc import SequentialMonteCarlo
from .state import ParticleState, PersistentParticleState
from .steps.constraints import (
    FixedLikelihoodThresholdSchedule,
    LikelihoodConstrainedPrior,
    LogPriorMassStoppingCriterion,
    QuantileLikelihoodThresholdSchedule,
    WorstParticleLikelihoodThresholdSchedule,
)
from .target import (
    EvaluationCounts,
    SequentialTarget,
    Target,
    TargetEvaluation,
    TemperedTarget,
)

try:
    __version__ = importlib.metadata.version("sequince")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

__all__ = [
    "NestedSamplingLevel",
    "FixedLikelihoodThresholdSchedule",
    "LikelihoodConstrainedPrior",
    "LogPriorMassStoppingCriterion",
    "QuantileLikelihoodThresholdSchedule",
    "WorstParticleLikelihoodThresholdSchedule",
    "make_nested_sampling_smc",
    "NestedSamplingSMCResult",
    "DegenerateWeightsError",
    "EvaluationCounts",
    "ParticleState",
    "PersistentParticleState",
    "ProposalDistribution",
    "MCMCResult",
    "MarkovChainMonteCarlo",
    "SequentialMonteCarlo",
    "MetaSequentialMonteCarlo",
    "SMCResult",
    "SequentialTarget",
    "SequentialDensity",
    "TemperedDensity",
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
