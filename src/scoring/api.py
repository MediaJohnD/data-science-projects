from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/score")
def score(request: ScoreRequest):
    """Return a dummy score based on the sum of the input features."""
    return {"score": sum(request.features)}
