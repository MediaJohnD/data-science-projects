import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from orchestrator import main_flow


def test_flow_runs(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "target": [1, 2]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    # Execution should complete without raising exceptions
    main_flow(str(csv_path))
