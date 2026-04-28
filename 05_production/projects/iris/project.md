# Iris Species Classifier API — FastAPI + Docker

> Part of the `ml-from-scratch` project  
> Path: `05_production/projects/iris/project.md`

---

## Project Goal

The goal of this project is to serve a from-scratch softmax regression classifier as a REST API using FastAPI, and to containerize it with Docker. Rather than running predictions inside a notebook, the model is exposed via HTTP endpoints, meaning any client (a browser, a script, another application) can send iris flower measurements and receive a species prediction in return.

The API serves one model trained on the Iris dataset:

- A `SoftmaxRegression` model built from scratch

It predicts the **species** of an iris flower (0 = setosa, 1 = versicolor, 2 = virginica) based on 4 numeric features: sepal length, sepal width, petal length, and petal width.

---

## Model Source

Built from scratch in:

```
02_classical_ml/03_softmax_regression/
```

Trained on the **sklearn Iris dataset** and saved at:

```
02_classical_ml/03_softmax_regression/models/softmax_regression_scratch.pkl
```

Trained with `learning_rate=0.01`, `num_iterations=5000`.  
Test accuracy: 100% on a 80/20 train/test split (`random_state=42`).

---

## Project Structure

```
02_classical_ml/03_softmax_regression/api/
├── main.py                              ← FastAPI app, defines the endpoints
├── schemas.py                           ← Pydantic input schema
├── softmax_regression.py                ← Custom SoftmaxRegression class
├── softmax_regression_scratch.pkl       ← Trained softmax regression model
├── scaler.pkl                           ← StandardScaler fit on X_train
├── Dockerfile                           ← Instructions to build the Docker image
├── requirements.txt                     ← Python dependencies for the container
└── project.md                           ← This file
```

---

## What Each File Does

### `main.py`

The core of the API. It creates the FastAPI application, loads the model and the scaler once at startup using the `lifespan` context manager, and defines two endpoints:

- `GET /health` : returns status and the loaded model name
- `POST /predict/` : runs the softmax regression model

The predict endpoint scales the input features with the scaler before running inference, and returns both a predicted class (integer) and a probability vector (list of 3 floats, one per class).

### `schemas.py`

Defines the `PredictRequest` class using Pydantic. This class describes exactly what the client must send: 4 float fields matching the Iris dataset feature names.

FastAPI uses this schema to automatically validate incoming requests, if a field is missing or the wrong type, the API rejects the request with a clear error.

### `softmax_regression.py`

The custom `SoftmaxRegression` class. joblib needs this file to reconstruct the model object when loading the `.pkl`. Without it, loading raises a `ModuleNotFoundError`.

### `softmax_regression_scratch.pkl`

The serialized trained model. Loaded once at startup using `joblib.load()` and reused for every prediction request.

### `scaler.pkl`

The `StandardScaler` fit on `X_train`. Applied via `scaler.transform()` (not `fit_transform()`) at inference time to avoid data leakage.

### `Dockerfile`

Instructions for Docker to build a self-contained image of the API.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py schemas.py softmax_regression.py softmax_regression_scratch.pkl scaler.pkl /app/
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
cd path/to/ml-from-scratch/02_classical_ml/03_softmax_regression/api
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
cd path/to/ml-from-scratch/02_classical_ml/03_softmax_regression/api
```

### 3. Build the Docker image

```bash
docker build -t softmax-regression-api .
```

### 4. Run the container

```bash
docker run -p 8000:8000 softmax-regression-api
```

### 5. Stop the container

Press `CTRL+C`, or run:

```bash
docker stop $(docker ps -q --filter ancestor=softmax-regression-api)
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
  "model": "SoftmaxRegression"
}
```

### Test `/predict/` via `/docs`

Open your browser and go to:

```
http://localhost:8000/docs
```

Click on the predict endpoint, then `Try it out`, paste the following JSON body
(raw features — the API applies scaling internally), and click `Execute`:

```json
{
  "sepal_length_cm": 5.1,
  "sepal_width_cm": 3.5,
  "petal_length_cm": 1.4,
  "petal_width_cm": 0.2
}
```

Expected response:

```json
{
  "prediction": 0,
  "prediction_probability": [0.9859, 0.014, 0.000003]
}
```

A prediction of `0` means the flower is classified as **setosa**.  
A prediction of `1` means the flower is classified as **versicolor**.  
A prediction of `2` means the flower is classified as **virginica**.

---

## Key Concepts Learned

- Softmax regression generalizes logistic regression to K classes, the output layer produces K probabilities that sum to 1
- The `/predict` endpoint returns a probability vector (one value per class) instead of a single scalar
- `prediction_proba[0].tolist()` converts a NumPy array to a JSON-serializable Python list
- `type(model).__name__` retrieves the class name of a loaded joblib object as a string
- The `lifespan` context manager is the modern FastAPI pattern for startup logic (replaces `@app.on_event("startup")`)
- Models and scalers are loaded once at startup and stored as global variables for performance
- joblib requires the custom class definition to be present at load time, always copy `.py` files alongside `.pkl` files
- The scaler must be applied with `transform()` at inference time, never `fit_transform()`
- NumPy types must be cast to native Python types before returning a JSON response, use `.tolist()` for arrays and `int()` for scalars
- Docker packages the app and all its dependencies into a portable container
- Port mapping (`-p 8000:8000`) connects the container's internal port to your machine
- `0.0.0.0` is the server listen address inside the container, access the API via `localhost` from your browser
- The correct pip package name is `scikit-learn`, not `sklearn`

---

## How to Reuse This Pattern for Another Model

1. Train and save your model and scaler with joblib
2. Copy the `.pkl` files and class definitions into your API folder
3. Update `schemas.py` to match the new dataset's features
4. Update the predict endpoint in `main.py` to build the feature array from the new schema
5. Update `requirements.txt` if you need additional dependencies
6. Run locally with `uvicorn main:app --reload`, or build and run with Docker

---

## References

- [FastAPI — Official Documentation](https://fastapi.tiangolo.com/)
- [Pydantic — Official Documentation](https://docs.pydantic.dev/)
- [joblib — Documentation](https://joblib.readthedocs.io/)
- [Docker — Official Documentation](https://docs.docker.com/)
- [Iris Dataset — sklearn](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
