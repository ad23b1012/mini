"""Logging utilities."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "mmer-xai",
    log_dir: str = "./results/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / "training.log")
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
