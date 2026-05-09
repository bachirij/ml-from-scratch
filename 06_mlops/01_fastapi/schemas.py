# =============================================================================
# schemas.py — Pydantic Input Schema Template
# =============================================================================
# This file defines the shape of the data the client must send to /predict.
# Pydantic validates every incoming request automatically — if a field is
# missing or the wrong type, FastAPI returns a 422 error before your function
# is even called.
#
# To use this template:
# 1. Rename the class if you want (PredictRequest is a good convention)
# 2. Replace feature_1, feature_2, etc. with your actual feature names
# 3. Set the correct type for each field (float, int, str, bool...)
# 4. Import this class in main.py: from schemas import PredictRequest
# =============================================================================

from pydantic import BaseModel

# -----------------------------------------------------------------------------
# PredictRequest
# Describes the JSON body the client must send to POST /predict.
#
# Example of valid request body:
# {
#     "feature_1": 0.5,
#     "feature_2": 1.2,
#     "feature_3": -0.3
# }
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    feature_1: float   # ← replace with your actual feature name and type
    feature_2: float
    feature_3: float
    # ... add as many fields as your model expects