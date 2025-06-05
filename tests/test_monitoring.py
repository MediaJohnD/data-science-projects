import numpy as np

from src.monitoring.drift_detection import monitor_drift


def test_monitor_drift(tmp_path):
    data1 = np.random.rand(100)
    data2 = np.random.rand(100)
    baseline = tmp_path / "base.npy"
    stat1, drift1 = monitor_drift(data1, baseline_path=str(baseline))
    assert stat1 == 0.0 and not drift1
    stat2, _ = monitor_drift(data2, baseline_path=str(baseline))
    assert stat2 >= 0.0
