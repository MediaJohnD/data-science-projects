import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator import main_flow  # noqa: E402
from ingestion.load_data import run as ingest  # noqa: E402
from features.engineer_features import run as engineer  # noqa: E402
from modeling.opti_shift import train  # noqa: E402
from scoring.api import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from prefect.testing.utilities import prefect_test_harness  # noqa: E402


def test_flow_callable():
    assert callable(main_flow)


def test_flow_runs():
    """Flow should execute without raising exceptions."""
    with prefect_test_harness():
        app = main_flow()
    assert app is not None


def test_individual_tasks():
    visits = ingest()
    assert {"device_id", "timestamp", "poi_id"}.issubset(visits.columns)

    feats = engineer(visits)
    assert {"device_id", "visit_count", "unique_pois"}.issubset(feats.columns)

    model, metrics = train(
        feats[["visit_count", "unique_pois"]],
        (feats["unique_pois"] > 1).astype(int),
    )
    assert "accuracy" in metrics


def test_predict_endpoint():
    visits = ingest()
    feats = engineer(visits)
    model, _ = train(
        feats[["visit_count", "unique_pois"]],
        (feats["unique_pois"] > 1).astype(int),
    )
    app = create_app(model)
    client = TestClient(app)
    resp = client.post("/predict", json={"features": [1, 1]})
    assert resp.status_code == 200
    assert "probability" in resp.json()
