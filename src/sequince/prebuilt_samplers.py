from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Sequence, Tuple

from ._typing import MetaData, MutationStrategy
from .posterior import TemperedPosterior
from .result import SMCResult
from .smc import SequentialMonteCarlo
from .steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MultinomialResampling,
    StratifiedResampling,
    SystematicResampling,
)
from .steps.base import SMCStep, StepContext
from .steps.mutation import get_mutation_step
from .transforms import ParticleTransform, get_prebuilt_transform

if TYPE_CHECKING:
    from .state import ParticleState

ResamplingStrategy = Literal["systematic", "stratified", "multinomial"]


def make_standard_bayesian_smc(
    log_prior: Callable[..., Any],
    log_likelihood: Callable[..., Any],
    *,
    log_proposal: Callable[..., Any] | None = None,
    target_ess_ratio: float = 0.7,
    max_temperature: float = 1.0,
    min_delta: float = 1e-8,
    resampling: ResamplingStrategy = "systematic",
    resampling_ess_threshold: float | None = 0.5,
    always_resample: bool = False,
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
        Optional callable for ``log q(x)`` when importance sampling proposals
        are used.
    target_ess_ratio:
        Target effective sample size ratio used by adaptive tempering.
    max_temperature:
        Final tempering temperature; typically ``1.0`` for full posterior.
    min_delta:
        Minimum temperature increment allowed in adaptive tempering.
    resampling:
        Resampling strategy to use; one of ``"systematic"``, ``"stratified"``,
        or ``"multinomial"``.
    resampling_ess_threshold:
        ESS ratio threshold triggering resampling. Set to ``None`` to disable
        ESS-triggered resampling (useful with ``always_resample=True``).
    always_resample:
        Force resampling at every iteration regardless of ESS.
    mutation:
        Mutation strategy to use. Accepts ``"emcee"`` (default),
        ``"minipcn"``, ``"none"``, or a pre-built ``SMCStep`` instance.
    mutation_n_steps:
        Number of mutation steps to run per iteration.
    mutation_kwargs:
        Extra keyword arguments forwarded to the mutation step constructor.
        For ``"emcee"`` these are passed to ``emcee.EnsembleSampler``; for
        ``"minipcn"`` they are provided to ``MiniPCNMutation``.
    logging_frequency:
        Frequency (in SMC iterations) at which to log sampler status.
        Defaults to 1 (log every iteration). If set to 0, logging is disabled.
    auto_normalize:
        Normalize log weights after every iteration (default ``True``).
    callbacks:
        Optional (initialised) callbacks to attach to the sampler.
    """
    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_proposal=log_proposal,
    )

    weighting_step = AdaptiveTemperedWeighting(
        posterior,
        target_ess_ratio=target_ess_ratio,
        max_temperature=max_temperature,
        min_delta=min_delta,
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
        posterior=posterior,
        transform=particle_transform,
        mutation_n_steps=mutation_n_steps,
        mutation_kwargs=mutation_kwargs,
    )

    steps = [weighting_step, resampling_step]
    if mutation_step is not None:
        steps.append(mutation_step)

    if logging_frequency > 0:

        def logging_fn(state: ParticleState, context: StepContext) -> str:
            current_temp = context.metadata.get("temperature", 0.0)
            return (
                f"SMC iteration {context.t}: "
                f"Temperature={current_temp:.4f}, "
                f"Log Evidence={state.log_evidence:.2f}, "
            )

        logging_step = LoggingStep(
            str_fn=logging_fn, frequency=logging_frequency
        )
        steps.append(logging_step)

    sampler = SequentialMonteCarlo(
        steps=steps, callbacks=callbacks, auto_normalize=auto_normalize
    )

    meta_data = dict(
        max_temperature=max_temperature,
    )

    return sampler, posterior, meta_data
