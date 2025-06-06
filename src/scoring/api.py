from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictRequest(BaseModel):
    value: float


class PredictResponse(BaseModel):
    prediction: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Return a simple double of the provided value."""
    return PredictResponse(prediction=req.value * 2)
