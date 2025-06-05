from fastapi.testclient import TestClient
from src.scoring.api import app


def test_read_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_endpoint():
    client = TestClient(app)
    response = client.post("/score", json={"features": [1, 2, 3]})
    assert response.status_code == 200
    assert response.json() == {"score": 6}
