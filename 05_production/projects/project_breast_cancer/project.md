# Breast Cancer Detection API — FastAPI + Docker

> Part of the `ml-from-scratch` project  
> Path: `05_production/projects/project_breast_cancer/project.md`

---

## Project Goal

The goal of this project is to serve a logistic regression model as a REST API using
FastAPI, and to containerize it with Docker. Rather than running predictions inside a
notebook, the model is exposed via HTTP endpoints — meaning any client (a browser, a
script, another application) can send tumor measurements and receive a diagnosis
prediction in return.

The model predicts whether a tumor is **malignant (0)** or **benign (1)** based on 30
numeric features extracted from digitized images of breast mass biopsies.

---

## Model Source

The model served by this API is the `LogisticRegression` class built from scratch in:

```
02_classical_ml/02_logistic_regression/
```

It was trained on the **sklearn breast cancer dataset** (30 float features, binary target)
and saved using joblib at:

```
02_classical_ml/02_logistic_regression/models/logistic_regression_scratch.pkl
```

Both the `.pkl` file and the `logistic_regression.py` class definition were copied into
the API folder so the project is self-contained and does not depend on files outside its
own directory.

---

## Project Structure

```
05_production/projects/project_breast_cancer/
├── main.py                         ← FastAPI app — defines the endpoints
├── schemas.py                      ← Pydantic input schema — defines what the API expects
├── logistic_regression.py          ← Custom LogisticRegression class (required to load the .pkl)
├── logistic_regression_scratch.pkl ← Trained model
├── Dockerfile                      ← Instructions to build the Docker image
├── requirements.txt                ← Python dependencies for the container
├── project.md                      ← This file
└── docker_context.md               ← Context file for the Docker session
```

---

## What Each File Does

### `main.py`

The core of the API. It creates the FastAPI application, loads the model once at startup,
and defines two endpoints:

- `GET /health` — returns `{"status": "ok"}` to confirm the API is running
- `POST /predict` — receives 30 features as JSON, runs the model, returns the predicted class

### `schemas.py`

Defines the `PredictRequest` class using Pydantic. This class describes exactly what the
client must send: 30 float fields matching the breast cancer dataset feature names.
FastAPI uses this schema to automatically validate incoming requests, if a field is
missing or the wrong type, the API rejects the request with a clear error.

### `logistic_regression.py`

The custom `LogisticRegression` class. joblib needs this file to reconstruct the model
object when loading the `.pkl`. Without it, loading the model raises a
`ModuleNotFoundError`.

### `logistic_regression_scratch.pkl`

The serialized trained model. Loaded once at startup using `joblib.load()` and reused
for every prediction request.

### `Dockerfile`

Instructions for Docker to build a self-contained image of the API. It specifies the
base Python version, installs dependencies, copies the project files, and defines the
command to start the server.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py schemas.py logistic_regression.py logistic_regression_scratch.pkl /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `requirements.txt`

The list of Python packages installed inside the Docker container:

```
fastapi
uvicorn
joblib
numpy
```

---

## Option 1 — Run the API Locally (without Docker)

### 1. Activate your conda environment

```bash
conda activate your_env_name
```

Replace `your_env_name` with the name of your environment. You can list available
environments with:

```bash
conda env list
```

### 2. Install dependencies (first time only)

```bash
pip install fastapi uvicorn joblib numpy
```

### 3. Navigate to the project folder

```bash
cd path/to/ml-from-scratch/05_production/projects/project_breast_cancer
```

You must be in this folder when launching uvicorn — it looks for `main.py` in the
current directory.

### 4. Start the API

```bash
uvicorn main:app --reload
```

- `main` → the file `main.py`
- `app` → the FastAPI object defined inside it
- `--reload` → automatically restarts the server when you save changes (development only)

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 5. Stop the API

Press `CTRL+C` in the terminal.

---

## Option 2 — Run the API with Docker

### 1. Make sure Docker is running

Open Docker Desktop and verify it is running, or check in the terminal:

```bash
docker --version
```

### 2. Navigate to the project folder

```bash
cd path/to/ml-from-scratch/02_classical_ml/02_logistic_regression/api
```

### 3. Build the Docker image

```bash
docker build -t breast-cancer-api .
```

- `docker build` → builds an image from the Dockerfile
- `-t breast-cancer-api` → names the image `breast-cancer-api`
- `.` → tells Docker to look for the Dockerfile in the current folder

### 4. Run the container

```bash
docker run -p 8000:8000 breast-cancer-api
```

- `docker run` → starts a container from the image
- `-p 8000:8000` → maps port 8000 on your machine to port 8000 inside the container
- `breast-cancer-api` → the name of the image to run

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 5. Stop the container

Press `CTRL+C` in the terminal, or run:

```bash
docker stop $(docker ps -q --filter ancestor=breast-cancer-api)
```

---

## Testing the API

Both options expose the API on the same address. Once running:

### Test `/health`

Open your browser and go to:

```
http://127.0.0.1:8000/health
```

Expected response:

```json
{ "status": "ok" }
```

### Test `/predict` via `/docs`

Open your browser and go to:

```
http://127.0.0.1:8000/docs
```

Click on `POST /predict`, then `Try it out`, paste the following JSON body (first sample
from the breast cancer test set, scaled), and click `Execute`:

```json
{
  "mean_radius": 1.0993723318640178,
  "mean_texture": -2.0716633787617127,
  "mean_perimeter": 1.0993723318640178,
  "mean_area": 1.1151253987522385,
  "mean_smoothness": 0.3540213718095619,
  "mean_compactness": 1.0837244520772064,
  "mean_concavity": 1.1243109499623624,
  "mean_concave_points": 1.4819306419867418,
  "mean_symmetry": 0.1817553853971798,
  "mean_fractal_dimension": 0.5408764046960991,
  "radius_error": 1.1594824091482755,
  "texture_error": -0.2197879786098986,
  "perimeter_error": 1.1594824091482755,
  "area_error": 1.0872781913571953,
  "smoothness_error": -0.4606536861386064,
  "compactness_error": 0.6576510044785031,
  "concavity_error": 0.6576510044785031,
  "concave_points_error": 0.6576510044785031,
  "symmetry_error": -0.2197879786098986,
  "fractal_dimension_error": 0.6576510044785031,
  "worst_radius": 1.1151253987522385,
  "worst_texture": -1.7862703431017523,
  "worst_perimeter": 1.1151253987522385,
  "worst_area": 1.0872781913571953,
  "worst_smoothness": 0.2196399139930551,
  "worst_compactness": 0.8219091289404353,
  "worst_concavity": 0.8219091289404353,
  "worst_concave_points": 1.2153352289538573,
  "worst_symmetry": 0.2196399139930551,
  "worst_fractal_dimension": 0.6576510044785031
}
```

Expected response:

```json
{ "prediction": 0 }
```

A prediction of `0` means the tumor is classified as **malignant**.  
A prediction of `1` means the tumor is classified as **benign**.

---

## Key Concepts Learned

- A REST API exposes functionality over HTTP using standard verbs (GET, POST)
- FastAPI defines endpoints as Python functions decorated with `@app.get` or `@app.post`
- Pydantic schemas validate incoming data automatically before the function is called
- The model is loaded once at startup (not on every request) for performance
- uvicorn is the server that runs the FastAPI application on the network
- Docker packages the app and all its dependencies into a portable container
- A Docker image is built once from a Dockerfile; a container is a running instance of that image
- Port mapping (`-p 8000:8000`) connects the container's internal port to your machine
- The project is self-contained — all files it needs live in its own directory
- NumPy types (e.g. `numpy.int64`) must be cast to native Python types before returning a JSON response, FastAPI cannot serialize NumPy types directly

---

## How to Reuse This Pattern for Another Model

1. Train and save your model with joblib in your algorithm folder
2. Copy the `.pkl` and the class definition into your API folder
3. Update `schemas.py` to match your model's input features and types
4. Update `main.py` to load the new model and build the correct NumPy array
5. Update `requirements.txt` if you need additional dependencies
6. Run locally with `uvicorn main:app --reload`, or build and run with Docker

The structure stays the same — only the schema and the model change.

---

## References

- [FastAPI — Official Documentation](https://fastapi.tiangolo.com/)
- [Pydantic — Official Documentation](https://docs.pydantic.dev/)
- [joblib — Documentation](https://joblib.readthedocs.io/)
- [Docker — Official Documentation](https://docs.docker.com/)
- [Breast Cancer Dataset — sklearn](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
