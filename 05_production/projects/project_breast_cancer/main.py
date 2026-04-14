# FastAPI app — serves the LogisticRegression and NeuralNetwork Classifier models trained on the breast cancer dataset
# Logistic Regression Model source: 02_classical_ml/02_logistic_regression/models/logistic_regression_scratch.pkl
# Neural Network Classifier Model source: 03_deep_learning/01_neural_network/models/neural_network_classifier_scratch.pkl

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from schemas import PredictRequest
from contextlib import asynccontextmanager

# Model variables to hold loaded models and scaler
models = {}
scaler = None

# Load models and scaler at api startup (so it's ready for predictions when the API receives requests)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, scaler
    # Load logistic regression model
    models = {
        'logistic_regression': joblib.load("logistic_regression_scratch.pkl"),
        'neural_network': joblib.load("neural_network_classifier_scratch.pkl")
    }
    # Load the scaler
    scaler = joblib.load("scaler.pkl")
    yield

# Create a FastAPI instance
app = FastAPI(lifespan=lifespan)

# Define a simple route to check if the API is working
@app.get("/health")
async def root():
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {"status": "ok", 
            "models": list(models.keys())}


# /predict/logistic_regression endpoint to make predictions
@app.post("/predict/logistic_regression")
async def predict_logistic_regression(request: PredictRequest):
    try:
        # Convert the request data into a format suitable for the model
        features = np.array([
            [request.mean_radius, 
             request.mean_texture, 
             request.mean_perimeter, 
             request.mean_area, 
             request.mean_smoothness, 
             request.mean_compactness, 
             request.mean_concavity, 
             request.mean_concave_points, 
             request.mean_symmetry, 
             request.mean_fractal_dimension, 
             request.radius_error, 
             request.texture_error, 
             request.perimeter_error, 
             request.area_error, 
             request.smoothness_error, 
             request.compactness_error, 
             request.concavity_error, 
             request.concave_points_error, 
             request.symmetry_error, 
             request.fractal_dimension_error, 
             request.worst_radius, 
             request.worst_texture, 
             request.worst_perimeter, 
             request.worst_area, 
             request.worst_smoothness, 
             request.worst_compactness, 
             request.worst_concavity, 
             request.worst_concave_points, 
             request.worst_symmetry, 
             request.worst_fractal_dimension]]) # Size (1, 30): 1 sample, 30 features
        scaled_features = scaler.transform(features) # Scale features with the loaded scaler
        prediction = models.get("logistic_regression").predict(scaled_features) # Size (1,): 1 prediction for 1
        prediction_proba = models.get("logistic_regression").predict_proba(scaled_features) # Get the probability of obtaining the class 1
        return {"prediction": int(prediction[0]), # return the single prediction as an integer
                "prediction_probability": float(prediction_proba[0])} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# /predict/neural_network endpoint to make predictions
@app.post("/predict/neural_network")
async def predict_neural_network(request: PredictRequest):
    try:
        # Convert the request data into a format suitable for the model
        features = np.array([
            [request.mean_radius, 
             request.mean_texture, 
             request.mean_perimeter, 
             request.mean_area, 
             request.mean_smoothness, 
             request.mean_compactness, 
             request.mean_concavity, 
             request.mean_concave_points, 
             request.mean_symmetry, 
             request.mean_fractal_dimension, 
             request.radius_error, 
             request.texture_error, 
             request.perimeter_error, 
             request.area_error, 
             request.smoothness_error, 
             request.compactness_error, 
             request.concavity_error, 
             request.concave_points_error, 
             request.symmetry_error, 
             request.fractal_dimension_error, 
             request.worst_radius, 
             request.worst_texture, 
             request.worst_perimeter, 
             request.worst_area, 
             request.worst_smoothness, 
             request.worst_compactness, 
             request.worst_concavity, 
             request.worst_concave_points, 
             request.worst_symmetry, 
             request.worst_fractal_dimension]]) # Size (1, 30): 1 sample, 30 features
        scaled_features = scaler.transform(features) # Scale features with the loaded scaler
        prediction = models.get("neural_network").predict(scaled_features) # Size (1,): 1 prediction for 1
        prediction_proba = models.get("neural_network").predict_proba(scaled_features) # Get the probability of obtaining the class 1
        return {"prediction": int(prediction.flatten()[0]), # return the single prediction as an integer
                "prediction_probability": float(prediction_proba[0])} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
