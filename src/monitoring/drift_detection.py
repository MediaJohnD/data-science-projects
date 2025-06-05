import os
import numpy as np
from scipy.stats import ks_2samp


def monitor_drift(
    data,
    baseline_path: str = "data/baseline.npy",
    threshold: float = 0.1,
):
    """Return KS statistic and drift flag comparing to baseline.

    If baseline does not exist it will be created using the provided data.
    """
    data = np.array(data).ravel()
    if os.path.exists(baseline_path):
        baseline = np.load(baseline_path)
        stat, _ = ks_2samp(baseline, data)
        drift = bool(stat > threshold)
    else:
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        np.save(baseline_path, data)
        stat, drift = 0.0, False
    return stat, drift
