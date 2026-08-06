"""Public exceptions raised by Sequince."""


class DegenerateWeightsError(RuntimeError):
    """Raised when particle log weights have no finite normalizer."""

    def __init__(self, log_normalizer: float) -> None:
        self.log_normalizer = log_normalizer
        if log_normalizer == float("-inf"):
            reason = "all particles have zero weight"
        elif log_normalizer == float("inf"):
            reason = "one or more particle weights are positive infinity"
        else:
            reason = "the particle weights contain NaN or are undefined"
        super().__init__(
            "Particle log weights cannot be normalized because "
            f"{reason} (log normalizer={log_normalizer!r})."
        )
