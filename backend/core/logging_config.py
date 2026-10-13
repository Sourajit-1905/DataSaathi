"""
Logging setup
-------------
One handler on the root logger, writing to stdout (Docker and most hosts collect
that). Every line passes through a formatter that redacts API keys and tokens,
including inside tracebacks, so a secret that ends up in an exception message
or a log argument never reaches the log.

Paths are deliberately not redacted here: they are useful when debugging and the
log is server-side. Only text sent to clients is path-scrubbed (core.security).
"""

import logging
import sys

from core.config import settings
from core.security import scrub_secrets

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Libraries that are chatty at INFO, or at DEBUG when LOG_LEVEL=DEBUG.
NOISY_LOGGERS = [
    "uvicorn.access",        # we log one line per request ourselves
    "httpx", "httpcore", "urllib3",
    "asyncio",
    "multipart", "python_multipart",
    "sentence_transformers", "transformers", "huggingface_hub", "filelock",
    "google", "google_genai",
    "matplotlib", "PIL", "numba",
]


class RedactingFormatter(logging.Formatter):
    """Formats the whole record, traceback included, then redacts secrets."""

    def format(self, record: logging.LogRecord) -> str:
        return scrub_secrets(super().format(record))


def _resolve_level(name) -> tuple[int, bool]:
    """Return (level, valid). An unknown name falls back to INFO."""
    value = getattr(logging, str(name).upper(), None)
    if isinstance(value, int):
        return value, True
    return logging.INFO, False


def setup_logging() -> None:
    level, valid = _resolve_level(settings.LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    log = logging.getLogger(__name__)
    if not valid:
        log.warning("Unknown LOG_LEVEL %r; using INFO.", settings.LOG_LEVEL)
    log.info("Logging configured | level=%s", logging.getLevelName(level))





