# FastAPI app — serves the LinearRegression model trained on the diabetes dataset
# Model source: 02_classical_ml/01_linear_regression/models/linear_regression_scratch.pkl

from fastapi import FastAPI, HTTPException
from schemas import PredictRequest

import joblib
import numpy as np

# Create a FastAPI instance
app = FastAPI()

# Define a simple route to check if the API is working
@app.get("/health")
async def root():
    return {"status": "ok"}

# Load the model at startup (so it's ready for predictions when the API receives requests)
model = joblib.load("linear_regression_scratch.pkl")

# /predict endpoint to make predictions
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Convert the request data into a format suitable for the model
        features = np.array([[request.age, request.sex, request.bmi, request.bp, request.s1, request.s2, request.s3, request.s4, request.s5, request.s6]]) # Size (1, 10): 1 sample, 10 features
        prediction = model.predict(features) # Size (1,): 1 prediction for 1
        return {"prediction": prediction[0]} # return the single prediction as a float
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))