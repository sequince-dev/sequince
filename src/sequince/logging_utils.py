from __future__ import annotations

import logging
from typing import Any

LOGGER_NAME = "sequince"
DEFAULT_FORMAT = "%(asctime)s sequince [%(levelname)s] %(message)s"


def configure_logging(
    *,
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    stream: Any | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure Sequince logging with sensible defaults.

    Parameters
    ----------
    level:
        Logging level to apply. Defaults to ``logging.INFO``.
    fmt:
        Logging format string supplied to ``basicConfig``.
    stream:
        Optional stream object for log output (passed to ``basicConfig``).
    force:
        When ``True``, reconfigure logging even if handlers already exist on the Sequince logger.
    """

    logger = logging.getLogger(LOGGER_NAME)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=stream)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
