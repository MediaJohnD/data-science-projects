"""Scoring API using FastAPI."""

from __future__ import annotations

from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: List[float]


def create_app(model) -> FastAPI:
    """Create a FastAPI app that scores incoming feature vectors."""

    app = FastAPI()

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        df = pd.DataFrame([req.features])
        prob = float(model.predict_proba(df)[0, 1])
        return {"probability": prob}

    return app


def deploy(model) -> None:
    """Serve the model locally using Uvicorn."""

    import uvicorn

    app = create_app(model)
    uvicorn.run(app, host="0.0.0.0", port=8000)
