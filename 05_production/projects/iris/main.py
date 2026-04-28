# FastAPI app — serves the SoftmaxRegression model trained on the iris dataset
# Softmax Regression Model source: 02_classical_ml/03_softmax_regression/models/softmax_regression_scratch.pkl

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from schemas import PredictRequest
from contextlib import asynccontextmanager

# Model variables to hold loaded models and scaler
model = None
scaler = None

# Load model and scaler at api startup (so it's ready for predictions when the API receives requests)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    # Load softmax regression model
    model = joblib.load("softmax_regression_scratch.pkl")
    # Load the scaler
    scaler = joblib.load("scaler.pkl")
    yield

# Create a FastAPI instance
app = FastAPI(lifespan=lifespan)

# Define a simple route to check if the API is working
@app.get("/health")
async def root():
    if model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {"status": "ok", 
            "models": type(model).__name__}

# /predict/ endpoint to make predictions
@app.post("/predict/")
async def predict(request: PredictRequest):
    try:
        # Convert the request data into a format suitable for the model
        features = np.array([
            [request.sepal_length_cm,
             request.sepal_width_cm,
             request.petal_length_cm,
             request.petal_width_cm]])
        scaled_features = scaler.transform(features) # Scale features with the loaded scaler
        prediction = model.predict(scaled_features) 
        prediction_proba = model.predict_proba(scaled_features) # Get the probabilities of obtaining each class
        return {"prediction": int(prediction[0]), # return the single prediction as an integer
                "prediction_probability": prediction_proba[0].tolist()} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        