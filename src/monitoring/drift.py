"""Simple utilities for detecting drift and deciding when to retrain."""


def detect_drift(stat: float, reference_stat: float, threshold: float = 0.1) -> bool:
    """Return ``True`` if the absolute difference exceeds ``threshold``."""
    return abs(stat - reference_stat) > threshold


def should_retrain(drift_detected: bool, performance_drop: bool = False) -> bool:
    """Return ``True`` if either drift is detected or performance dropped."""
    return drift_detected or performance_drop
