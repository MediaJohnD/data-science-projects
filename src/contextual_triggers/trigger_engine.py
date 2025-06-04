
def should_trigger_retrain(accuracy: float, threshold: float = 0.95) -> bool:
    """Determine whether a model retraining should be triggered."""
    return accuracy < threshold
