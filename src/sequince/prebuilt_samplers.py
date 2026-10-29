from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from numbers import Integral
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Sequence, Tuple

import numpy as np

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
from .steps.constraints import (
    FixedLikelihoodThresholdSchedule,
    LikelihoodConstrainedPrior,
    LikelihoodThresholdSchedule,
    LikelihoodThresholdSelection,
    LikelihoodThresholdWeighting,
    LogPriorMassStoppingCriterion,
    QuantileLikelihoodThresholdSchedule,
    ShellEvidenceStep,
    WorstParticleLikelihoodThresholdSchedule,
    _nested_sampling_metadata,
)
from .steps.mutation import MutateIfResampled, get_mutation_step
from .steps.resampling import ResamplingStep
from .steps.weighting import TargetESSRatio
from .transforms import ParticleTransform, get_prebuilt_transform
from .typing import MetaData, MutationStrategy

if TYPE_CHECKING:
    from .state import ParticleState

ResamplingStrategy = Literal["systematic", "stratified", "multinomial"]
logger = logging.getLogger(__name__)

__all__ = [
    "make_persistent_bayesian_smc",
    "make_standard_bayesian_smc",
    "make_nested_sampling_smc",
]


def make_standard_bayesian_smc(
    log_prior: Callable[..., Any],
    log_likelihood: Callable[..., Any],
    *,
    log_proposal: Callable[..., Any] | ProposalDistribution | None = None,
    target_ess_ratio: TargetESSRatio = 0.7,
    max_beta: float = 1.0,
    min_delta: float = 1e-8,
    max_delta: float | None = None,
    adaptive: bool = True,
    num_steps: int | None = None,
    weighting_criterion: Literal["ess", "cess"] = "ess",
    weighting_kwargs: Dict[str, Any] | None = None,
    resampling: ResamplingStrategy = "systematic",
    resampling_ess_threshold: float | None = 0.5,
    always_resample: bool | None = None,
    resample_at_end: bool = True,
    resampling_kwargs: Dict[str, Any] | None = None,
    mutation: MutationStrategy | SMCStep = "differential-evolution",
    mutation_n_steps: int = 5,
    mutation_kwargs: Dict[str, Any] | None = None,
    adaptive_mutation: Mapping[str, object] | None = None,
    always_mutate: bool | None = None,
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
    the current beta are returned alongside the sampler for convenience.

    Parameters
    ----------
    log_prior : callable
        Callable evaluating the log prior density ``log p(x)``.
    log_likelihood : callable
        Callable evaluating the log likelihood ``log l(x)``.
    log_proposal : callable or ProposalDistribution or None, optional
        Optional callable or proposal object providing ``log q(x)`` when
        importance sampling proposals are used.
    target_ess_ratio : float or callable, optional
        Target effective sample size ratio used by adaptive tempering, or a
        callable of the current beta and optional ``StepContext`` that
        returns the target for each iteration.
    max_beta : float, optional
        Final tempering beta; typically ``1.0`` for full posterior.
    min_delta : float, optional
        Minimum beta increment allowed in adaptive tempering.
    max_delta : float or None, optional
        Optional maximum beta increment allowed in adaptive tempering.
    adaptive : bool, optional
        Whether to adapt the beta schedule using the target ESS.
    num_steps : int or None, optional
        Number of fixed beta steps. Required when ``adaptive=False``.
    weighting_criterion : {"ess", "cess"}, optional
        Criterion for adaptive tempering; one of ``"ess"`` (default) or ``"cess"``.
    weighting_kwargs : dict[str, Any] or None, optional
        Extra keyword arguments forwarded to the weighting step constructor.
    resampling : {"systematic", "stratified", "multinomial"}, optional
        Resampling strategy to use; one of ``"systematic"``, ``"stratified"``,
        or ``"multinomial"``.
    resampling_ess_threshold : float or None, optional
        ESS ratio threshold triggering resampling. Set to ``None`` to disable
        ESS-triggered resampling (useful with ``always_resample=True``).
    always_resample : bool or None, optional
        Force resampling at every iteration regardless of ESS. If set to
        ``None``, defaults to ``True`` if ``weighting_criterion=="ess"`` and
        ``False`` if ``weighting_criterion=="cess"``.
    resample_at_end : bool, optional
        If ``True`` (default) will always resample when the inverse
        temperature β reaches 1.0, regardless of ESS. If ``False``, the final
        resampling may be skipped if ``always_resample`` is ``False`` and the
        ESS ratio is above ``ess_threshold``.
    resampling_kwargs : dict[str, Any] or None, optional
        Extra keyword arguments forwarded to the resampling step constructor.
    mutation : str or SMCStep, optional
        Mutation strategy to use. Accepts ``"blackjax"``, ``"emcee"``, ``"eryn"``,
        ``"differential-evolution"`` (default), ``"metropolis-hastings"``,
        ``"minipcn"``, ``"none"``, or a pre-built ``SMCStep`` instance.
    mutation_n_steps : int, optional
        Number of mutation steps to run per iteration.
    mutation_kwargs : dict[str, Any] or None, optional
        Extra keyword arguments forwarded to the mutation step constructor.
        ``"metropolis-hastings"`` requires ``kernel`` and accepts optional
        ``kernel_kwargs``. For ``"emcee"``, ``move`` may be a named move,
        weighted move configuration, or the ``"bilby-de"`` preset.
        For ``"blackjax"``, use ``algorithm`` and ``algorithm_kwargs`` to
        configure the JAX mutation kernel.
    adaptive_mutation : Mapping[str, object] or None, optional
        Acceptance-based budget configuration; see the user guide. Defaults
        to disabled. Uses ``mutation_n_steps`` as the initial total budget.
    always_mutate : bool or None, optional
        Force mutation at every iteration regardless of resampling. If set to
        ``None``, defaults to ``True`` if ``always_resample=True`` and ``False``
        if ``always_resample=False``.
    particle_transform : ParticleTransform or str or None, optional
        Transform applied to particles before mutation. Accepts a string
        identifier for a prebuilt transform, a ``ParticleTransform`` instance,
        or ``None`` to skip transformation.
    particle_transform_kwargs : dict[str, Any] or None, optional
        Extra keyword arguments forwarded to the particle transform
        constructor.
    logging_frequency : int, optional
        Frequency (in SMC iterations) at which to log sampler status.
        Defaults to 1 (log every iteration). If set to 0, logging is disabled.
    auto_normalize : bool, optional
        Normalize log weights after every iteration (default ``True``).
    callbacks : sequence of callable or None, optional
        Optional (initialised) callbacks to attach to the sampler.
    checkpoint_writers : sequence of CheckpointWriter or None, optional
        Optional checkpoint writers attached to the sampler.

    Returns
    -------
    sampler : SequentialMonteCarlo
        Sampler configured with weighting, resampling, and mutation steps.
    posterior : TemperedPosterior
        Target shared by the sampler's steps.
    metadata : dict[str, Any]
        Initial run metadata containing ``max_beta``.
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

    if always_mutate is None:
        always_mutate = always_resample

    if mutation != "none" and always_resample and not always_mutate:
        raise ValueError(
            "always_mutate=False is incompatible with always_resample=True."
        )

    weighting_step: SMCStep
    if adaptive:
        weighting_step = AdaptiveTemperedWeighting(
            posterior,
            target_ess_ratio=target_ess_ratio,
            max_beta=max_beta,
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
            beta_increment=max_beta / num_steps,
            max_beta=max_beta,
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
        resample_at_end=resample_at_end,
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
        adaptive_mutation=adaptive_mutation,
        mutation_kwargs=mutation_kwargs,
    )

    steps: list[SMCStep] = [weighting_step, resampling_step]
    if mutation_step is not None:
        if not always_mutate:
            mutation_step = MutateIfResampled(mutation_step)
        steps.append(mutation_step)

    if logging_frequency > 0:

        def logging_fn(state: ParticleState, context: StepContext) -> str:
            current_beta = context.metadata.get("beta", 0.0)
            return (
                f"SMC iteration {context.t}: "
                f"Beta={current_beta:.3g}, "
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
        max_beta=max_beta,
    )

    return sampler, posterior, meta_data


def make_persistent_bayesian_smc(
    log_prior: Callable[..., Any],
    log_likelihood: Callable[..., Any],
    *,
    log_proposal: Callable[..., Any] | ProposalDistribution | None = None,
    n_particles: int = 256,
    target_ess: float | None = None,
    target_total_ess: float | None = None,
    dynamic: bool = False,
    weighting_kwargs: Dict[str, Any] | None = None,
    resampling: ResamplingStrategy | ResamplingStep = "systematic",
    resample_at_end: bool = True,
    resampling_kwargs: Dict[str, Any] | None = None,
    mutation: MutationStrategy | SMCStep = "differential-evolution",
    mutation_n_steps: int = 5,
    mutation_kwargs: Dict[str, Any] | None = None,
    adaptive_mutation: Mapping[str, object] | None = None,
    particle_transform: ParticleTransform | str | None = "affine",
    particle_transform_kwargs: Dict[str, Any] | None = None,
    auto_normalize: bool = True,
    logging_frequency: int = 1,
    callbacks: Sequence[
        Callable[[ParticleState, StepContext, SMCResult], None]
    ]
    | None = None,
    checkpoint_writers: Sequence[CheckpointWriter] | None = None,
) -> tuple[SequentialMonteCarlo, TemperedPosterior, MetaData]:
    """Return an SMC runner configured for persistent sampling.

    Mutation defaults to differential evolution; other mutation options follow
    :func:`make_standard_bayesian_smc`.

    ``n_particles`` is the active population size drawn from the complete
    persistent archive on every iteration. ``target_ess`` defaults to twice
    ``n_particles``. ``target_ess`` and ``target_total_ess`` are absolute
    effective sample sizes of that archive; they are not fractions of the active
    population size.

    With ``dynamic=True``, adapt the temperature ESS threshold using the
    multinomial expected unique fraction (a heuristic under systematic
    resampling). The initial threshold is truncated to an integer >= 1;
    ``target_total_ess`` never adapts. Updates take effect next iteration.

    The initial ``ParticleState`` may contain any positive multiple of
    ``n_particles`` equally weighted draws. Its size determines the initial
    population; the sampler neither draws particles nor chooses this size.
    Draws must follow ``log_proposal`` when configured, otherwise the prior.
    All initial batches are archived at beta zero on the first call, before
    tempering and resampling to the fixed active batch size. Nonmultiples are
    rejected rather than truncated. Initialization is atomic, so checkpoints
    resume after it, not between initial batches.

    A one-batch initial state preserves the existing initialization behavior.
    The first persistence step prepares evaluations and archives the state at beta
    zero. With ``resample_at_end=True`` (default), selecting beta one always
    completes resampling and the configured mutation. The next iteration archives
    that batch and reweights the complete archive before checking termination.
    If ``target_total_ess`` is set, beta-one batches accumulate until the returned
    weighted archive meets that threshold; no extra batch is generated afterward.
    The final archive weighting also updates the evidence estimate.
    With ``resample_at_end=False``, stop immediately after weighting and return
    the complete weighted archive without generating a final active batch.
    Progress is logged every
    ``logging_frequency`` iterations and once more for the terminal
    state. Set ``logging_frequency=0`` to disable it.

    ``adaptive_mutation`` optionally configures acceptance-based budgets for
    named miniPCN or differential-evolution mutation, starting at
    ``mutation_n_steps``. See the user guide for options and lagged behavior.

    For more details see https://arxiv.org/abs/2407.20722.
    """
    if isinstance(n_particles, bool) or not isinstance(n_particles, Integral):
        raise TypeError("n_particles must be a positive integer.")
    if n_particles <= 0:
        raise ValueError("n_particles must be a positive integer.")
    n_particles = int(n_particles)
    if target_ess is None:
        target_ess = 2 * n_particles

    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_proposal=log_proposal,
    )

    weighting_step = PersistentAdaptiveTemperedWeighting(
        posterior,
        target_ess=target_ess,
        target_total_ess=target_total_ess,
        dynamic=dynamic,
        resample_at_end=resample_at_end,
        **weighting_kwargs or {},
    )

    persist_step = PersistParticles(posterior, n_particles=n_particles)

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
        target=posterior,
        transform=particle_transform,
        mutation_n_steps=mutation_n_steps,
        adaptive_mutation=adaptive_mutation,
        mutation_kwargs=mutation_kwargs,
    )

    steps: list[SMCStep] = [
        persist_step,
        weighting_step,
        resampling_step,
    ]
    if mutation_step is not None:
        steps.append(mutation_step)

    configured_callbacks = list(callbacks or ())
    if logging_frequency > 0:

        def logging_callback(
            state: ParticleState,
            context: StepContext,
            result: SMCResult,
        ) -> None:
            if (
                context.t % logging_frequency != 0
                and not context.stop_requested
            ):
                return
            archive = persist_step.persistent_state
            # PersistParticles always runs before iteration callbacks.
            if archive is None:  # pragma: no cover
                n_batches = 0
                archive_size = 0
            else:
                n_batches = archive.n_batches
                archive_size = archive.n_particles
            beta = float(context.metadata.get("beta", 0.0))
            delta_beta = float(context.metadata.get("delta_beta", 0.0))
            outputs = result.outputs[-1]
            weighting_output = outputs[1]
            persistent_ess = float(weighting_output["effective_sample_size"])
            required_ess = (
                target_total_ess
                if beta >= 1.0 and target_total_ess is not None
                else float(weighting_output["target_ess"])
            )
            mutation = outputs[3] if len(outputs) > 3 else None
            acceptance = (
                f", Acceptance={float(mutation['metrics'].acceptance):.3f}"
                if isinstance(mutation, Mapping)
                and mutation.get("metrics") is not None
                else ""
            )
            logger.info(
                f"Persistent SMC iteration {context.t}: "
                f"Beta={beta:.6g} "
                f"(delta={delta_beta:.3g}), "
                f"Batches={n_batches}, "
                f"Archive size={archive_size}, "
                f"Persistent ESS={persistent_ess:.1f}/{required_ess:.1f}, "
                f"Log evidence={float(state.log_evidence):+.6g}"
                f"{acceptance}"
            )

        configured_callbacks.append(logging_callback)

    runner = SequentialMonteCarlo(
        steps=steps,
        callbacks=configured_callbacks,
        checkpoint_writers=checkpoint_writers,
        auto_normalize=auto_normalize,
    )
    return runner, posterior, {}


def make_nested_sampling_smc(
    log_prior: Callable[..., Any],
    log_likelihood: Callable[..., Any],
    *,
    threshold_schedule: (
        Literal["worst"] | Sequence[float] | LikelihoodThresholdSchedule | None
    ) = None,
    threshold_quantile: float | None = None,
    threshold_inclusive: bool = False,
    evidence_tolerance: float | None = 1e-3,
    log_likelihood_upper_bound: float | None = None,
    store_shell_particles: bool = True,
    min_log_prior_volume: float | None = None,
    resampling: ResamplingStrategy = "systematic",
    resampling_ess_threshold: float | None = None,
    always_resample: bool = True,
    resampling_kwargs: Dict[str, Any] | None = None,
    mutation: MutationStrategy | SMCStep | None = "differential-evolution",
    mutation_n_steps: int = 5,
    mutation_kwargs: Dict[str, Any] | None = None,
    particle_transform: ParticleTransform | str | None = "affine",
    particle_transform_kwargs: Dict[str, Any] | None = None,
    logging_frequency: int = 1,
    auto_normalize: bool = False,
    callbacks: Sequence[
        Callable[[ParticleState, StepContext, SMCResult], None]
    ]
    | None = None,
    checkpoint_writers: Sequence[CheckpointWriter] | None = None,
) -> Tuple[SequentialMonteCarlo, LikelihoodConstrainedPrior, MetaData]:
    """Build a nested-sampling SMC pipeline for a normalized prior.

    Parameters
    ----------
    log_prior, log_likelihood : callable
        Static vectorized log densities. Initial particles must represent the
        normalized prior; arbitrary importance proposals are not supported.
    threshold_schedule : {"worst"}, sequence of float, callable, or None
        Log-likelihood thresholds. None uses a weighted median. Sequences
        finalize on exhaustion; callables may raise StopIteration. Callables
        must implement schedule(log_likelihoods, state, context) -> float.
        Other user exceptions propagate unchanged.
    threshold_quantile : float or None
        Fraction to discard, in [0, 1). Overrides the default median; cannot
        be combined with an explicit threshold_schedule.
    threshold_inclusive : bool
        Retain particles on the threshold boundary. Adaptive schedules require
        a strict constraint to guarantee progress.
    evidence_tolerance : float or None
        Stop when the estimated remaining evidence is this fraction of the
        accumulated evidence. This is not an upper bound. Set to None to use
        only fixed-schedule or prior-volume stopping.
    log_likelihood_upper_bound : float or None
        Known global upper bound on log likelihood. If supplied, evidence
        stopping uses X * Lmax rather than X * mean(L), protecting against
        unseen higher-likelihood regions. X remains a Monte Carlo estimate.
    min_log_prior_volume : float or None
        Optional negative log prior-volume stopping limit.
    store_shell_particles : bool
        Retain discarded particles for posterior reconstruction. Disable for
        evidence-only runs with lower memory use.
    checkpoint_writers : sequence or None
        Writers used by the standard SMC checkpoint machinery.

    Returns
    -------
    sampler : SequentialMonteCarlo
        Runner using the current step/context APIs. Wrap its result in
        NestedSamplingSMCResult for evidence and posterior samples. Reaching
        run(num_steps=...) suspends the run without integrating its live tail;
        resume it to completion before requesting posterior samples.
    target : LikelihoodConstrainedPrior
        Mutation target with cached prior and likelihood components.
    metadata : dict
        Initial runtime bookkeeping.

    Notes
    -----
    Other mutation, transformation, resampling, and callback options follow
    make_standard_bayesian_smc. auto_normalize must be False. State normalizing
    constants track constrained-prior mass, not Bayesian evidence. The latter
    is accumulated separately over disjoint shells, including the live tail.
    Adaptive thresholds do not inherit fixed-schedule unbiasedness guarantees.
    Likelihood ties are discarded together; sampled plateaus with no survivors
    are integrated as a terminal shell. This cannot detect unobserved modes.

    Functional execution requires JAX arrays, unconditional resampling with
    unchanged population size, logging_frequency=0, and traceable mutation
    (BlackJAX or MiniPCN). Select one explicitly with ``mutation``; the default
    differential-evolution mutation is not supported for functional nested
    sampling. Use callbacks for host-side logging. Initialize with
    run_functional or initialize_functional before compiling iterations.
    Compiled schedules must use backend operations and return a scalar array;
    positive infinity finalizes the live tail. Python StopIteration is supported
    by eager and uncompiled schedules, not as a data-dependent compiled branch.
    Custom target functions and transforms must also be JAX traceable.
    """

    if auto_normalize:
        raise ValueError(
            "make_nested_sampling_smc requires auto_normalize=False because "
            "the constraint weighting step handles normalization explicitly."
        )

    if threshold_quantile is not None and threshold_schedule is not None:
        raise ValueError(
            "Specify threshold_quantile or threshold_schedule, not both."
        )
    if threshold_quantile is not None:
        resolved_threshold_schedule: LikelihoodThresholdSchedule = (
            QuantileLikelihoodThresholdSchedule(threshold_quantile)
        )
    elif threshold_schedule is None:
        resolved_threshold_schedule = QuantileLikelihoodThresholdSchedule()
    elif isinstance(threshold_schedule, str):
        if threshold_schedule == "worst":
            resolved_threshold_schedule = (
                WorstParticleLikelihoodThresholdSchedule()
            )
        else:
            raise ValueError("threshold_schedule string must be 'worst'.")
    elif isinstance(threshold_schedule, np.ndarray) or isinstance(
        threshold_schedule,
        SequenceABC,
    ):
        resolved_threshold_schedule = FixedLikelihoodThresholdSchedule(
            tuple(float(value) for value in threshold_schedule)
        )
    elif callable(threshold_schedule):
        resolved_threshold_schedule = threshold_schedule
    else:
        raise ValueError(
            "threshold_schedule must be 'worst', a sequence of thresholds, "
            "or a callable."
        )

    constrained_prior = LikelihoodConstrainedPrior(
        log_prior,
        log_likelihood,
        inclusive=threshold_inclusive,
    )
    if threshold_inclusive and isinstance(
        resolved_threshold_schedule, QuantileLikelihoodThresholdSchedule
    ):
        raise ValueError(
            "Adaptive schedules require threshold_inclusive=False."
        )
    threshold_step = LikelihoodThresholdSelection(
        constrained_prior,
        threshold_schedule=resolved_threshold_schedule,
        evidence_tolerance=evidence_tolerance,
        log_likelihood_upper_bound=log_likelihood_upper_bound,
    )
    shell_evidence_step = ShellEvidenceStep(
        constrained_prior, store_shell_particles=store_shell_particles
    )
    weighting_step = LikelihoodThresholdWeighting(constrained_prior)

    resampling_map = {
        "systematic": SystematicResampling,
        "stratified": StratifiedResampling,
        "multinomial": MultinomialResampling,
    }
    try:
        resampling_cls = resampling_map[resampling]
    except KeyError as exc:
        raise ValueError(
            f"Unknown resampling strategy '{resampling}'."
        ) from exc
    resampling_step = resampling_cls(
        ess_threshold=resampling_ess_threshold,
        always_resample=always_resample,
        **resampling_kwargs or {},
    )

    if isinstance(particle_transform, str):
        particle_transform = get_prebuilt_transform(
            particle_transform, **(particle_transform_kwargs or {})
        )
    elif particle_transform is not None and not isinstance(
        particle_transform, ParticleTransform
    ):
        raise ValueError(
            "particle_transform must be a string, ParticleTransform, or None."
        )
    mutation_step = get_mutation_step(
        mutation,
        target=constrained_prior,
        transform=particle_transform,
        mutation_n_steps=mutation_n_steps,
        mutation_kwargs=mutation_kwargs,
    )

    steps: list[SMCStep] = [
        threshold_step,
        shell_evidence_step,
        weighting_step,
    ]
    if min_log_prior_volume is not None:
        steps.append(
            LogPriorMassStoppingCriterion(
                min_log_prior_volume,
                constrained_prior,
                store_shell_particles=store_shell_particles,
            )
        )
    steps.append(resampling_step)
    if mutation_step is not None:
        steps.append(mutation_step)

    if logging_frequency > 0:

        def logging_fn(state: ParticleState, context: StepContext) -> str:
            runtime = context.metadata["nested_sampling"]
            threshold = runtime["threshold"]
            log_x = runtime["log_prior_volume"]
            log_z = runtime["log_nested_evidence"]
            n_surviving = runtime["n_surviving"]
            n_particles = state.particles.shape[0]
            return (
                f"NS-SMC iteration {context.t}: "
                f"Threshold={threshold:.3g}, "
                f"Log X={log_x:.3g}, "
                f"Log Z={log_z:.3g}, "
                f"Survivors={n_surviving}/{n_particles}, "
                f"ESS={context.metadata.get('ess', math.nan):.1f}"
            )

        steps.append(
            LoggingStep(str_fn=logging_fn, frequency=logging_frequency)
        )

    sampler = SequentialMonteCarlo(
        steps=steps,
        callbacks=callbacks,
        auto_normalize=False,
        checkpoint_writers=checkpoint_writers,
    )
    metadata = {"nested_sampling": _nested_sampling_metadata({})}
    return sampler, constrained_prior, metadata
