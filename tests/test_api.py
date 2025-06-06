from fastapi.testclient import TestClient

from src.scoring.api import app


def test_predict_endpoint():
    client = TestClient(app)
    response = client.post("/predict", json={"value": 3})
    assert response.status_code == 200
    assert response.json() == {"prediction": 6}
