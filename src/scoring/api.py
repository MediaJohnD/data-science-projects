from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()


class PredictionRequest(BaseModel):
    data: list


@app.post("/predict")
async def predict_endpoint(payload: PredictionRequest):
    df = pd.DataFrame(payload.data)
    # Placeholder for model prediction
    return {"rows": len(df)}
