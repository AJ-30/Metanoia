"""Logger setup for the Reconstruction package.

Uses the standard-library ``logging`` module (no extra dependencies) with one
console handler and one file handler. Every log line carries the run id so
logs from different runs can be told apart in a shared log file.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT: str = "[%(run_id)s] %(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


class _RunIdFilter(logging.Filter):
    """Logging filter that stamps every record with the current run id."""

    def __init__(self, run_id: str) -> None:
        """Store the run id.

        Args:
            run_id (str): Identifier added to each log record.
        """
        super().__init__()
        self.run_id: str = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach ``run_id`` to the record; never drops records.

        Args:
            record (logging.LogRecord): Record being emitted.

        Returns:
            bool: Always True.
        """
        record.run_id = self.run_id
        return True


def make_run_id(prefix: str) -> str:
    """Build a sortable, human-readable run id.

    Args:
        prefix (str): Short name of the job, e.g. "dataprep".

    Returns:
        str: ``<prefix>-YYYYmmdd-HHMMSS``.
    """
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def setup_logger(name: str, log_file: Path, run_id: str, debug: bool) -> logging.Logger:
    """Create a logger writing to stdout and to ``log_file``.

    Args:
        name (str): Logger name (usually the entry-point module name).
        log_file (Path): File the log is appended to; parent dirs are created.
        run_id (str): Run identifier stamped on every line.
        debug (bool): True for DEBUG level, False for INFO.

    Returns:
        logging.Logger: Configured logger.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    run_filter = _RunIdFilter(run_id)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setFormatter(formatter)
        handler.addFilter(run_filter)
        logger.addHandler(handler)
    return logger
