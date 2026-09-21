"""
Logging Configuration for Crowd Anomaly Detection System.
Provides formatted console and file logging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# Windows-safe colored formatting
class ColoredFormatter(logging.Formatter):
    """Adds ANSI color codes to log levels for clear terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname_color = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "crowd_anomaly",
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: Optional[str | Path] = "logs/crowd_anomaly.log",
) -> logging.Logger:
    """
    Sets up and configures a standardized logger.

    Args:
        name: Logger name (defaults to 'crowd_anomaly').
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_to_file: Whether to save logs to a file.
        log_file: Relative or absolute path to log file.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_format = ColoredFormatter(
        "[%(asctime)s] [%(levelname_color)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_format = logging.Formatter(
                "[%(asctime)s] [%(levelname)-8s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"[WARNING] Could not initialize file logging to {log_file}: {e}")

    return logger
