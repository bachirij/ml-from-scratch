# FastAPI app — serves the LogisticRegression model trained on the breast cancer dataset
# Model source: 02_classical_ml/02_logistic_regression/models/logistic_regression_scratch.pkl

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
model = joblib.load("logistic_regression_scratch.pkl")

# /predict endpoint to make predictions
@app.post("/predict")
async def predict(request: PredictRequest):
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
        prediction = model.predict(features) # Size (1,): 1 prediction for 1
        return {"prediction": int(prediction[0])} # return the single prediction as an integer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))