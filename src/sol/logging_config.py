import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure structlog with optional file logging and rotation.

    When log_dir is provided, logs are written to a rotating file at
    ``log_dir/sol.log`` (10 MB, 5 backups). Console output is added
    when stderr is a TTY.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / "sol.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            ),
        )
        root.addHandler(file_handler)

    if sys.stderr.isatty():
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(),
                ],
            ),
        )
        root.addHandler(console_handler)

    # If no handlers at all (no log_dir, not a TTY), add a basic stderr handler
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(log_level)
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            ),
        )
        root.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
