# Pydantic schemas for the /predict endpoint
# Features match the sklearn diabetes dataset (10 float features)
# Model source: 02_classical_ml/01_linear_regression/models/linear_regression_scratch.pkl

from pydantic import BaseModel

class PredictRequest(BaseModel):
    age : float
    sex : float
    bmi : float
    bp : float
    s1 : float
    s2 : float
    s3 : float
    s4 : float
    s5 : float
    s6 : float

