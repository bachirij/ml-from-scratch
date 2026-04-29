# Breast Cancer Detection API — FastAPI + Docker

> Part of the `ml-from-scratch` project  
> Path: `05_production/projects/project_breast_cancer/project.md`

---

## Project Goal

The goal of this project is to serve three from-scratch classifiers as a REST API using FastAPI, and to containerize it with Docker. Rather than running predictions inside a notebook, the models are exposed via HTTP endpoints, meaning any client (a browser, a script, another application) can send tumor measurements and receive a diagnosis prediction in return.

The API serves three models trained on the same dataset:

- A `LogisticRegression` model built from scratch
- A `NeuralNetwork` model built from scratch
- A `DecisionTree` model built from scratch

All three predict whether a tumor is **malignant (0)** or **benign (1)** based on 30 numeric features extracted from digitized images of breast mass biopsies.

---

## Model Sources

### Logistic Regression

Built from scratch in:

```
02_classical_ml/02_logistic_regression/
```

Trained on the **sklearn breast cancer dataset** and saved at:

```
02_classical_ml/02_logistic_regression/models/logistic_regression_scratch.pkl
```

### Neural Network

Built from scratch in:

```
03_deep_learning/01_neural_network/
```

Architecture: `[30, 16, 8, 1]` - 3 hidden layers with ReLU activations, sigmoid output.  
Trained with `learning_rate=0.001`, `n_iterations=10000`, `random_seed=42`.  
Test accuracy: 96.5% — AUC: 0.991.

Saved at:

```
03_deep_learning/01_neural_network/models/neural_network_classifier_scratch.pkl
```

### Decision Tree

Built from scratch in:

```
02_classical_ml/04_decision_tree/
```

Trained with `max_depth=5`, `min_samples_split=2`, `criterion='gini'`.  
Test accuracy: 93% - AUC: 0.922.

Saved at:

```
02_classical_ml/04_decision_tree/models/decision_tree_classifier_scratch.pkl
```

All three models were trained on the same dataset with the same `train_test_split` (`random_state=42`), so a single `StandardScaler` (fit on `X_train`) applies to all.

---

## Project Structure

```
05_production/projects/project_breast_cancer/
├── main.py                                    ← FastAPI app, defines the endpoints
├── schemas.py                                 ← Pydantic input schema
├── logistic_regression.py                     ← Custom LogisticRegression class
├── neural_network.py                          ← Custom NeuralNetwork class
├── decision_tree.py                           ← Custom DecisionTreeClassifier class
├── logistic_regression_scratch.pkl            ← Trained logistic regression model
├── neural_network_classifier_scratch.pkl      ← Trained neural network model
├── decision_tree_classifier_scratch.pkl       ← Trained decision tree model
├── scaler.pkl                                 ← StandardScaler fit on X_train
├── Dockerfile                                 ← Instructions to build the Docker image
├── requirements.txt                           ← Python dependencies for the container
└── project.md                                 ← This file
```

---

## What Each File Does

### `main.py`

The core of the API. It creates the FastAPI application, loads all three models and the scaler once at startup using the `lifespan` context manager, and defines four endpoints:

- `GET /health` : returns status and the list of available models
- `POST /predict/logistic_regression` : runs the logistic regression model
- `POST /predict/neural_network` : runs the neural network model
- `POST /predict/decision_tree` : runs the decision tree model

All predict endpoints scale the input features with the shared scaler before running inference, and return both a predicted class and a probability.

The three models are stored in a single `models` dictionary loaded at startup:

```python
models = {
    'logistic_regression': joblib.load("logistic_regression_scratch.pkl"),
    'neural_network': joblib.load("neural_network_classifier_scratch.pkl"),
    'decision_tree': joblib.load("decision_tree_classifier_scratch.pkl")
}
```

### `schemas.py`

Defines the `PredictRequest` class using Pydantic. This class describes exactly what the client must send: 30 float fields matching the breast cancer dataset feature names.

FastAPI uses this schema to automatically validate incoming requests, if a field is missing or the wrong type, the API rejects the request with a clear error.

### `logistic_regression.py`

The custom `LogisticRegression` class. joblib needs this file to reconstruct the model object when loading the `.pkl`. Without it, loading raises a `ModuleNotFoundError`.

### `neural_network.py`

The custom `NeuralNetwork` class. Same requirement as above, joblib needs the class definition at load time. The class uses named functions for all activation derivatives (no lambdas) to ensure joblib serialization works correctly.

### `decision_tree.py`

The custom `DecisionTreeClassifier` class, along with the `Node` class it depends on. joblib needs both class definitions at load time to reconstruct the model object. The file also contains the `Node` class which represents each node in the tree.

### `logistic_regression_scratch.pkl` / `neural_network_classifier_scratch.pkl` / `decision_tree_classifier_scratch.pkl`

The serialized trained models. All three are loaded once at startup using `joblib.load()` and reused for every prediction request.

### `scaler.pkl`

The `StandardScaler` fit on `X_train`. Shared by all three models since they were trained on the same split. Applied via `scaler.transform()` (not `fit_transform()`) at inference time to avoid data leakage.

### `Dockerfile`

Instructions for Docker to build a self-contained image of the API.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py schemas.py logistic_regression.py logistic_regression_scratch.pkl neural_network.py neural_network_classifier_scratch.pkl decision_tree.py decision_tree_classifier_scratch.pkl scaler.pkl /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `requirements.txt`

```
fastapi
uvicorn
joblib
numpy
scikit-learn
```

`scikit-learn` is required to deserialize the `StandardScaler` saved with joblib.

Note: the correct package name for pip is `scikit-learn`, not `sklearn`.

---

## Option 1 - Run the API Locally (without Docker)

### 1. Activate your conda environment

```bash
conda activate your_env_name
```

### 2. Install dependencies (first time only)

```bash
pip install fastapi uvicorn joblib numpy scikit-learn
```

### 3. Navigate to the project folder

```bash
cd path/to/ml-from-scratch/05_production/projects/project_breast_cancer
```

### 4. Start the API

```bash
uvicorn main:app --reload
```

### 5. Stop the API

Press `CTRL+C` in the terminal.

---

## Option 2 - Run the API with Docker

### 1. Make sure Docker is running

```bash
docker --version
```

### 2. Navigate to the project folder

```bash
cd path/to/ml-from-scratch/05_production/projects/project_breast_cancer
```

### 3. Build the Docker image

```bash
docker build -t breast-cancer-api .
```

### 4. Run the container

```bash
docker run -p 8000:8000 breast-cancer-api
```

### 5. Stop the container

Press `CTRL+C`, or run:

```bash
docker stop $(docker ps -q --filter ancestor=breast-cancer-api)
```

---

## Testing the API

Once running (locally or via Docker), the API is accessible at `http://localhost:8000`.

### Test `/health`

```
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "models": ["logistic_regression", "neural_network", "decision_tree"]
}
```

### Test `/predict/logistic_regression`, `/predict/neural_network` and `/predict/decision_tree` via `/docs`

Open your browser and go to:

```
http://localhost:8000/docs
```

Click on any predict endpoint, then `Try it out`, paste the following JSON body
(raw features - the API applies scaling internally), and click `Execute`:

```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001.0,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871,
  "radius_error": 1.095,
  "texture_error": 0.9053,
  "perimeter_error": 8.589,
  "area_error": 153.4,
  "smoothness_error": 0.006399,
  "compactness_error": 0.04904,
  "concavity_error": 0.05373,
  "concave_points_error": 0.01587,
  "symmetry_error": 0.03003,
  "fractal_dimension_error": 0.006193,
  "worst_radius": 25.38,
  "worst_texture": 17.33,
  "worst_perimeter": 184.6,
  "worst_area": 2019.0,
  "worst_smoothness": 0.1622,
  "worst_compactness": 0.6656,
  "worst_concavity": 0.7119,
  "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601,
  "worst_fractal_dimension": 0.1189
}
```

Expected response format:

```json
{
  "prediction": 0,
  "prediction_probability": 0.97
}
```

A prediction of `0` means the tumor is classified as **malignant**.  
A prediction of `1` means the tumor is classified as **benign**.

---

## Key Concepts Learned

- A REST API exposes functionality over HTTP using standard verbs (GET, POST)
- FastAPI defines endpoints as Python functions decorated with `@app.get` or `@app.post`
- Pydantic schemas validate incoming data automatically before the function is called
- Multiple models can be served from a single API using a `models` dictionary
- A single shared scaler applies to multiple models when they were trained on the same split
- The scaler must be applied with `transform()` at inference time, never `fit_transform()`
- The `lifespan` context manager is the modern FastAPI pattern for startup logic (replaces `@app.on_event("startup")`)
- Models and scalers are loaded once at startup and stored as global variables for performance
- joblib requires the custom class definition to be present at load time, always copy `.py` files alongside `.pkl` files, this applies to all helper classes too (e.g. `Node` alongside `DecisionTreeClassifier`)
- Lambdas cannot be serialized by joblib, use named functions instead
- NumPy types (e.g. `numpy.int64`, `numpy.float64`) must be cast to native Python types before returning a JSON response
- The `NeuralNetwork.predict()` output has shape `(1, n_samples)`, use `.flatten()[0]` to extract a scalar
- Docker packages the app and all its dependencies into a portable container
- Port mapping (`-p 8000:8000`) connects the container's internal port to your machine
- `0.0.0.0` is the server listen address inside the container, access the API via `localhost` from your browser
- The correct pip package name is `scikit-learn`, not `sklearn`

---

## How to Reuse This Pattern for Another Model

1. Train and save your model and scaler with joblib
2. Copy the `.pkl` files and class definitions into your API folder
3. Add the model to the `models` dictionary in `main.py`
4. Add a new predict endpoint following the same pattern
5. Update `requirements.txt` if you need additional dependencies
6. Run locally with `uvicorn main:app --reload`, or build and run with Docker

---

## References

- [FastAPI — Official Documentation](https://fastapi.tiangolo.com/)
- [Pydantic — Official Documentation](https://docs.pydantic.dev/)
- [joblib — Documentation](https://joblib.readthedocs.io/)
- [Docker — Official Documentation](https://docs.docker.com/)
- [Breast Cancer Dataset — sklearn](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
