from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Tuple

from ._typing import MutationStrategy
from .posterior import TemperedPosterior
from .smc import SequentialMonteCarlo
from .steps import (
    AdaptiveTemperedWeighting,
    MultinomialResampling,
    StratifiedResampling,
    SystematicResampling,
)
from .steps.base import SMCStep
from .steps.mutation import get_mutation_step

ResamplingStrategy = Literal["systematic", "stratified", "multinomial"]


def make_standard_bayesian_smc(
    log_prior: Callable[..., Any],
    log_likelihood: Callable[..., Any],
    *,
    log_proposal: Callable[..., Any] | None = None,
    target_ess_ratio: float = 0.7,
    max_temperature: float = 1.0,
    min_delta: float = 1e-4,
    resampling: ResamplingStrategy = "systematic",
    resampling_ess_threshold: float | None = 0.5,
    always_resample: bool = False,
    mutation: MutationStrategy | SMCStep = "emcee",
    mutation_n_steps: int = 5,
    mutation_move: Any = None,
    mutation_run_kwargs: Dict[str, Any] | None = None,
    mutation_kwargs: Dict[str, Any] | None = None,
    auto_normalize: bool = True,
) -> Tuple[SequentialMonteCarlo, TemperedPosterior, Dict[str, Any]]:
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
    mutation_move:
        Optional emcee move kernel (defaults to ``StretchMove``). Only used
        when ``mutation="emcee"``.
    mutation_run_kwargs:
        Additional keyword arguments passed to ``sampler.run_mcmc``.
        Only used when ``mutation="emcee"``.
    mutation_kwargs:
        Extra keyword arguments forwarded to the mutation step constructor.
        For ``"emcee"`` these are passed to ``emcee.EnsembleSampler``; for
        ``"minipcn"`` they are provided to ``MiniPCNMutation``.
    auto_normalize:
        Normalize log weights after every iteration (default ``True``).
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

    mutation_step = get_mutation_step(
        mutation,
        posterior,
        n_steps=mutation_n_steps,
        move=mutation_move,
        run_kwargs=mutation_run_kwargs,
        mutation_kwargs=mutation_kwargs,
    )

    steps = [weighting_step, resampling_step]
    if mutation_step is not None:
        steps.append(mutation_step)

    sampler = SequentialMonteCarlo(steps=steps, auto_normalize=auto_normalize)

    shared_context: Dict[str, Any] = {
        "temperature": 0.0,
        "temperature_delta": 0.0,
        "max_temperature": max_temperature,
    }

    return sampler, posterior, shared_context
