from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Sequence, Tuple

from ._typing import MetaData, MutationStrategy
from .checkpoint import CheckpointWriter
from .posterior import TemperedPosterior
from .proposals import ProposalDistribution
from .result import SMCResult
from .smc import SequentialMonteCarlo
from .steps import (
    AdaptiveTemperedWeighting,
    FixedTemperedWeighting,
    LoggingStep,
    MultinomialResampling,
    PersistentAdaptiveTemperedWeighting,
    PersistParticles,
    StratifiedResampling,
    SystematicResampling,
)
from .steps.base import SMCStep, StepContext
from .steps.mutation import get_mutation_step
from .steps.resampling import ResamplingStep
from .steps.weighting import TargetESSRatio
from .target import TemperedTarget
from .transforms import ParticleTransform, get_prebuilt_transform

if TYPE_CHECKING:
    from .state import ParticleState

ResamplingStrategy = Literal["systematic", "stratified", "multinomial"]

__all__ = ["make_persistent_bayesian_smc", "make_standard_bayesian_smc"]


def make_standard_bayesian_smc(
    log_prior: Callable[..., Any],
    log_likelihood: Callable[..., Any],
    *,
    log_proposal: Callable[..., Any] | ProposalDistribution | None = None,
    target_ess_ratio: TargetESSRatio = 0.7,
    max_temperature: float = 1.0,
    min_delta: float = 1e-8,
    max_delta: float | None = None,
    adaptive: bool = True,
    num_steps: int | None = None,
    weighting_criterion: Literal["ess", "cess"] = "ess",
    weighting_kwargs: Dict[str, Any] | None = None,
    resampling: ResamplingStrategy = "systematic",
    resampling_ess_threshold: float | None = 0.5,
    always_resample: bool | None = None,
    resampling_kwargs: Dict[str, Any] | None = None,
    mutation: MutationStrategy | SMCStep = "minipcn",
    mutation_n_steps: int = 5,
    mutation_kwargs: Dict[str, Any] | None = None,
    particle_transform: ParticleTransform | str | None = "affine",
    particle_transform_kwargs: Dict[str, Any] | None = None,
    logging_frequency: int = 1,
    auto_normalize: bool = True,
    callbacks: Sequence[
        Callable[[ParticleState, StepContext, SMCResult], None]
    ]
    | None = None,
    checkpoint_writers: Sequence[CheckpointWriter] | None = None,
) -> Tuple[SequentialMonteCarlo, TemperedPosterior, MetaData]:
    """Return a SequentialMonteCarlo instance configured with standard SMC steps.

    The returned sampler applies adaptive tempering, ESS-triggered resampling,
    and mutation moves tailored for standard Bayesian inference workflows.
    A ``TemperedPosterior`` instance and a shared context dictionary tracking
    the current temperature are returned alongside the sampler for convenience.

    Parameters
    ----------
    log_prior:
        Callable evaluating the log prior density ``log p(x)``.
    log_likelihood:
        Callable evaluating the log likelihood ``log l(x)``.
    log_proposal:
        Optional callable or proposal object providing ``log q(x)`` when
        importance sampling proposals are used.
    target_ess_ratio:
        Target effective sample size ratio used by adaptive tempering, or a
        callable of the current temperature and optional ``StepContext`` that
        returns the target for each iteration.
    max_temperature:
        Final tempering temperature; typically ``1.0`` for full posterior.
    min_delta:
        Minimum temperature increment allowed in adaptive tempering.
    max_delta:
        Optional maximum temperature increment allowed in adaptive tempering.
    adaptive:
        Whether to adapt the temperature schedule using the target ESS.
    num_steps:
        Number of fixed temperature steps. Required when ``adaptive=False``.
    weighting_criterion:
        Criterion for adaptive tempering; one of ``"ess"`` (default) or ``"cess"``.
    weighting_kwargs:
        Extra keyword arguments forwarded to the weighting step constructor.
    resampling:
        Resampling strategy to use; one of ``"systematic"``, ``"stratified"``,
        or ``"multinomial"``.
    resampling_ess_threshold:
        ESS ratio threshold triggering resampling. Set to ``None`` to disable
        ESS-triggered resampling (useful with ``always_resample=True``).
    always_resample:
        Force resampling at every iteration regardless of ESS. If set to
        ``None``, defaults to ``True`` if ``weighting_criterion=="ess"`` and
        ``False`` if ``weighting_criterion=="cess"``.
    resampling_kwargs:
        Extra keyword arguments forwarded to the resampling step constructor.
    mutation:
        Mutation strategy to use. Accepts ``"blackjax"``, ``"emcee"``,
        ``"metropolis-hastings"``, ``"minipcn"`` (default), ``"none"``, or a
        pre-built ``SMCStep`` instance.
    mutation_n_steps:
        Number of mutation steps to run per iteration.
    mutation_kwargs:
        Extra keyword arguments forwarded to the mutation step constructor.
        ``"metropolis-hastings"`` requires ``kernel`` and accepts optional
        ``kernel_kwargs``. For ``"emcee"``, ``move`` may be a named move,
        weighted move configuration, or the ``"bilby-de"`` preset.
        For ``"blackjax"``, use ``algorithm`` and ``algorithm_kwargs`` to
        configure the JAX mutation kernel.
    logging_frequency:
        Frequency (in SMC iterations) at which to log sampler status.
        Defaults to 1 (log every iteration). If set to 0, logging is disabled.
    auto_normalize:
        Normalize log weights after every iteration (default ``True``).
    callbacks:
        Optional (initialised) callbacks to attach to the sampler.
    checkpoint_writers:
        Optional checkpoint writers attached to the sampler.
    """
    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_proposal=log_proposal,
    )

    if always_resample is None:
        if weighting_criterion == "ess":
            always_resample = True
        elif weighting_criterion == "cess":
            always_resample = False
        else:
            raise ValueError(
                "always_resample default not defined for weighting_criterion="
                f"'{weighting_criterion}'. Please set always_resample."
            )
    elif always_resample is False and weighting_criterion == "ess":
        raise ValueError(
            "always_resample cannot be False when weighting_criterion='ess' "
            "since ESS-based tempering relies on resampling to achieve the "
            "target ESS ratio."
        )

    weighting_step: SMCStep
    if adaptive:
        weighting_step = AdaptiveTemperedWeighting(
            posterior,
            target_ess_ratio=target_ess_ratio,
            max_temperature=max_temperature,
            min_delta=min_delta,
            max_delta=max_delta,
            criterion=weighting_criterion,
            **weighting_kwargs or {},
        )
    else:
        if num_steps is None or num_steps < 1:
            raise ValueError("num_steps must be positive when adaptive=False.")
        weighting_step = FixedTemperedWeighting(
            posterior,
            temperature_increment=max_temperature / num_steps,
            **weighting_kwargs or {},
        )

    resampling_map = {
        "systematic": SystematicResampling,
        "stratified": StratifiedResampling,
        "multinomial": MultinomialResampling,
    }
    try:
        resampling_cls = resampling_map[resampling]
    except KeyError as exc:
        valid = ", ".join(sorted(resampling_map))
        raise ValueError(
            f"Unknown resampling strategy '{resampling}'. "
            f"Valid options are: {valid}."
        ) from exc

    resampling_step = resampling_cls(
        ess_threshold=resampling_ess_threshold,
        always_resample=always_resample,
        **resampling_kwargs or {},
    )

    if isinstance(particle_transform, str):
        particle_transform_kwargs = particle_transform_kwargs or {}
        particle_transform = get_prebuilt_transform(
            particle_transform, **particle_transform_kwargs
        )
    elif (
        isinstance(particle_transform, ParticleTransform)
        or particle_transform is None
    ):
        pass
    else:
        raise ValueError(
            "particle_transform must be a string identifier, "
            "a ParticleTransform instance, or None."
        )

    mutation_step = get_mutation_step(
        mutation,
        target=posterior,
        transform=particle_transform,
        mutation_n_steps=mutation_n_steps,
        mutation_kwargs=mutation_kwargs,
    )

    steps: list[SMCStep] = [weighting_step, resampling_step]
    if mutation_step is not None:
        steps.append(mutation_step)

    if logging_frequency > 0:

        def logging_fn(state: ParticleState, context: StepContext) -> str:
            current_temp = context.metadata.get("temperature", 0.0)
            return (
                f"SMC iteration {context.t}: "
                f"Temperature={current_temp:.3g}, "
                f"Log Evidence={state.log_evidence:.3g}, "
                f"ESS={context.metadata.get('ess', math.nan):.1f}"
            )

        logging_step = LoggingStep(
            str_fn=logging_fn, frequency=logging_frequency
        )
        steps.append(logging_step)

    sampler = SequentialMonteCarlo(
        steps=steps,
        callbacks=callbacks,
        checkpoint_writers=checkpoint_writers,
        auto_normalize=auto_normalize,
    )

    meta_data = dict(
        max_temperature=max_temperature,
    )

    return sampler, posterior, meta_data


def make_persistent_bayesian_smc(
    target: TemperedTarget,
    *,
    n_particles: int = 256,
    target_ess: float = 512,
    target_total_ess: float | None = None,
    weighting_kwargs: Dict[str, Any] | None = None,
    resampling: ResamplingStrategy | ResamplingStep = "systematic",
    resampling_kwargs: Dict[str, Any] | None = None,
    mutation: MutationStrategy | SMCStep = "minipcn",
    mutation_n_steps: int = 5,
    mutation_kwargs: Dict[str, Any] | None = None,
    particle_transform: ParticleTransform | str | None = "affine",
    particle_transform_kwargs: Dict[str, Any] | None = None,
    auto_normalize: bool = True,
    callbacks: Sequence[
        Callable[[ParticleState, StepContext, SMCResult], None]
    ]
    | None = None,
    checkpoint_writers: Sequence[CheckpointWriter] | None = None,
) -> SequentialMonteCarlo:
    """Return an SMC runner configured for persistent sampling.

    ``n_particles`` is the active population size drawn from the complete
    persistent archive on every iteration. ``target_ess`` and
    ``target_total_ess`` are absolute effective sample sizes of that archive;
    they are not fractions of the active population size.

    The caller should initialize an ordinary :class:`ParticleState` containing
    ``n_particles`` draws from the initial target. The first persistence step
    archives that state at temperature zero, and the adaptive weighting step
    requests a normal runner stop once its terminal ESS condition is met.

    For more details see https://arxiv.org/abs/2407.20722.
    """
    if isinstance(n_particles, bool) or not isinstance(n_particles, Integral):
        raise TypeError("n_particles must be a positive integer.")
    if n_particles <= 0:
        raise ValueError("n_particles must be a positive integer.")
    n_particles = int(n_particles)

    persist_step = PersistParticles()
    weighting_step = PersistentAdaptiveTemperedWeighting(
        target,
        target_ess=target_ess,
        target_total_ess=target_total_ess,
        **weighting_kwargs or {},
    )

    if isinstance(resampling, ResamplingStep):
        if resampling_kwargs:
            raise ValueError(
                "resampling_kwargs cannot be used with a resampling step."
            )
        if not resampling._always_resample:  # noqa: SLF001
            raise ValueError(
                "Persistent sampling requires always_resample=True."
            )
        if resampling._n_samples != n_particles:  # noqa: SLF001
            raise ValueError(
                "Persistent sampling requires the resampling step to use "
                f"n_samples={n_particles}."
            )
        resampling_step = resampling
    else:
        resampling_map = {
            "systematic": SystematicResampling,
            "stratified": StratifiedResampling,
            "multinomial": MultinomialResampling,
        }
        try:
            resampling_cls = resampling_map[resampling]
        except KeyError as exc:
            valid = ", ".join(sorted(resampling_map))
            raise ValueError(
                f"Unknown resampling strategy '{resampling}'. "
                f"Valid options are: {valid}."
            ) from exc

        configured_resampling = dict(resampling_kwargs or {})
        configured_n_samples = configured_resampling.pop(
            "n_samples", n_particles
        )
        if configured_n_samples != n_particles:
            raise ValueError(
                "Persistent sampling requires resampling n_samples to equal "
                f"n_particles ({n_particles})."
            )
        configured_always = configured_resampling.pop("always_resample", True)
        if configured_always is not True:
            raise ValueError(
                "Persistent sampling requires always_resample=True."
            )
        resampling_step = resampling_cls(
            always_resample=True,
            n_samples=n_particles,
            **configured_resampling,
        )

    if isinstance(particle_transform, str):
        particle_transform = get_prebuilt_transform(
            particle_transform,
            **particle_transform_kwargs or {},
        )
    elif not (
        isinstance(particle_transform, ParticleTransform)
        or particle_transform is None
    ):
        raise ValueError(
            "particle_transform must be a string identifier, "
            "a ParticleTransform instance, or None."
        )

    mutation_step = get_mutation_step(
        mutation,
        target=target,
        transform=particle_transform,
        mutation_n_steps=mutation_n_steps,
        mutation_kwargs=mutation_kwargs,
    )

    steps: list[SMCStep] = [
        persist_step,
        weighting_step,
        resampling_step,
    ]
    if mutation_step is not None:
        steps.append(mutation_step)

    return SequentialMonteCarlo(
        steps=steps,
        callbacks=callbacks,
        checkpoint_writers=checkpoint_writers,
        auto_normalize=auto_normalize,
    )
