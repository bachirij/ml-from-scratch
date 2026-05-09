# Pydantic schemas for the /predict endpoint
# Features match the sklearn iris dataset (4 float features)
# Model source: 02_classical_ml/03_softmax_regression/models/softmax_regression_scratch.pkl

from pydantic import BaseModel

class PredictRequest(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float