"""Logging setup for Promptify."""

import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the promptify logger."""
    logger = logging.getLogger("promptify")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = logging.getLogger("promptify")
