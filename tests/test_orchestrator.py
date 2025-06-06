import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import main_flow  # noqa: E402


def test_flow_callable():
    assert callable(main_flow)


def test_flow_runs():
    """Flow should execute without raising exceptions."""
    app = main_flow()
    assert app is not None
