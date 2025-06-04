import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_accuracy(accuracy: float):
    logger.info("Model accuracy: %.4f", accuracy)
