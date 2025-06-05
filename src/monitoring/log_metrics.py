import logging

logger = logging.getLogger(__name__)


def log_accuracy(score: float) -> None:
    """Log model accuracy using the standard logger."""
    logger.info("Model accuracy: %.4f", score)
