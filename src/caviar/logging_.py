"""Logging utilities."""

import logging
import pathlib


def configure_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: str | pathlib.Path | None = None,
) -> logging.Logger:
    """Configure a basic logger.

    Args:
        name (str): Name of the logger.
        level (int | str, optional): Level below which logs won't be
            produced. Defaults to logging.INFO.
        log_file (str | pathlib.Path | None, optional): Path to the log file to
            produce. Defaults to None.

    Returns:
        logging.Logger: Resulting logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = pathlib.Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
