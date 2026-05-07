# =============================================================================
# FastAPI — Minimal Template
# =============================================================================
# This file is a generic template for serving an ML model via a REST API.
# It is NOT meant to be run directly — adapt it to your project.
#
# To use this template:
# 1. Replace the model loading path with your own .pkl file
# 2. Update the PredictRequest schema in schemas.py to match your features
# 3. Update the feature array in the /predict endpoint to match your schema
# 4. Run with: uvicorn main:app --reload
# =============================================================================

from fastapi import FastAPI, HTTPException
from schemas import PredictRequest  # Import the input schema from schemas.py

import joblib
import numpy as np

# -----------------------------------------------------------------------------
# Create the FastAPI application
# This object is what uvicorn looks for when you run: uvicorn main:app
# -----------------------------------------------------------------------------
app = FastAPI()

# -----------------------------------------------------------------------------
# Load the model once at startup
# Loading here (outside any function) means it happens once when the server
# starts — not on every request. This is important for performance.
#
# joblib also needs the class definition file (e.g. my_model.py) to be present
# in the same directory as the .pkl file, otherwise it raises ModuleNotFoundError.
# -----------------------------------------------------------------------------
model = joblib.load("your_model.pkl")  # ← replace with your model path

# -----------------------------------------------------------------------------
# GET /health
# A simple sanity check endpoint. Returns {"status": "ok"} if the API is running.
# Use this to verify the server is up before sending prediction requests.
# Test it by opening http://127.0.0.1:8000/health in your browser.
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# POST /predict
# The main endpoint. Receives features as JSON, runs the model, returns the result.
#
# FastAPI automatically:
# - Parses the incoming JSON into a PredictRequest object
# - Validates all fields (type, presence) using Pydantic
# - Returns a 422 error if validation fails (no need to handle this manually)
#
# The try/except block handles unexpected errors during prediction (e.g. model
# crash, shape mismatch) and returns a clean 500 error instead of crashing.
# -----------------------------------------------------------------------------
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Build a NumPy array from the incoming features
        # Shape must be (1, n_features) — one sample, n features
        # ↓ Replace with your own features in the correct order
        features = np.array([[
            request.feature_1,
            request.feature_2,
            # ... add all your features here
        ]])  # Shape: (1, n_features)

        # Run the model
        prediction = model.predict(features)  # Shape: (1,)

        # Return the single prediction value as JSON
        return {"prediction": prediction[0]}

    except Exception as e:
        # If anything goes wrong during prediction, return a 500 error
        # str(e) includes the actual error message — useful for debugging
        raise HTTPException(status_code=500, detail=str(e))