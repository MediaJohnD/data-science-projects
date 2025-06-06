from src.monitoring.drift import detect_drift, should_retrain


def test_detect_drift_negative():
    assert detect_drift(0.05, 0.0, threshold=0.1) is False


def test_should_retrain_negative():
    assert should_retrain(False, performance_drop=False) is False
