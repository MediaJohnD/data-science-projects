import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import main_flow


def test_flow_callable():
    assert callable(main_flow)
