from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")


@app.post("/predict")
def predict(features: list):
    array = np.array(features).reshape(1, -1)
    prediction = model.predict(array)
    return {"prediction": int(prediction[0])}
