FROM python:3.12-slim 
WORKDIR /app 
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py schemas.py logistic_regression.py logistic_regression_scratch.pkl neural_network.py neural_network_classifier_scratch.pkl scaler.pkl /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]