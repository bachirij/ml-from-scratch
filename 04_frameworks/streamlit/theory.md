# Streamlit — Theory

## Intuition

Streamlit turns a Python script into a web application. The core proposition: if you already know Python and your libraries (pandas, matplotlib, sklearn, NumPy), you can build an interactive interface around your work without learning HTML, CSS, JavaScript, or any web framework.

The user sees a web page with widgets, charts, and tables. Under the hood, it is a Python script running on a server. There is no separation between frontend and backend — it is all one file.

This is the key difference from tools like Grafana or Tableau: those tools are built around a UI that consumes data. Streamlit is built around code that produces a UI. Anything you can do in Python, you can surface in Streamlit: load a `.pkl` model, read a `.parquet` file, call an API, run NumPy computations, feed data into a Plotly chart.

---

## The Execution Model

This is the most important concept to understand. Streamlit has an unconventional execution model that surprises most developers the first time they encounter it.

**When a user interacts with a widget (changes a slider, selects a date, clicks a button) Streamlit reruns the entire Python script from top to bottom.**

There is no event listener attached to individual widgets. There is no partial update. The whole script runs again.

This means:

```python
import streamlit as st

n = st.slider("Pick a number", 1, 100)
st.write(n * 2)
```

Every time the slider moves, Python executes both lines again. `n` gets its new value, and `st.write` renders the new result.

This simplicity is intentional. You do not need to think about state synchronisation, callbacks, or DOM updates. You write top-to-bottom procedural code, and Streamlit handles the rest.

### Implications of full reruns

1. **Expensive computations re-run on every interaction.** If loading a dataset or a model takes 3 seconds, that cost is paid on every widget change, unless you cache it explicitly (see `st.cache_data` and `st.cache_resource` below).

2. **Variables are reset on every rerun.** You cannot store state in a regular Python variable between interactions. If you need state that persists across reruns, you must use `st.session_state`.

3. **Widget values are the primary inputs.** The return value of any widget (`st.slider(...)`, `st.selectbox(...)`, etc.) is the current value selected by the user. Read it like a variable.

---

## Application Structure

A Streamlit app is a single Python file, conventionally named `app.py`. You run it with:

```bash
streamlit run app.py
```

Streamlit starts a local web server (default port 8501) and opens a browser tab automatically.

### Minimal working app

```python
import streamlit as st

st.title("My App")
st.write("Hello, world.")
```

That is a complete, running application.

---

## Core API

Streamlit's API is organized around three categories: **display**, **input**, and **layout**.

### Display

These functions write content to the page. They execute in order, top to bottom, which determines the visual order.

| Function                    | What it renders                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `st.title("text")`          | Large heading                                                                             |
| `st.header("text")`         | Section heading                                                                           |
| `st.subheader("text")`      | Subsection heading                                                                        |
| `st.write(anything)`        | Smart renderer — handles text, dataframes, dicts, Matplotlib/Plotly figures, NumPy arrays |
| `st.text("text")`           | Plain monospace text                                                                      |
| `st.markdown("**bold**")`   | Markdown string                                                                           |
| `st.dataframe(df)`          | Interactive table                                                                         |
| `st.metric("label", value)` | KPI-style number display                                                                  |
| `st.plotly_chart(fig)`      | Plotly figure                                                                             |
| `st.pyplot(fig)`            | Matplotlib figure                                                                         |
| `st.image(...)`             | Image (path, URL, or array)                                                               |

`st.write` is the versatile catch-all. It inspects the type of its argument and renders it appropriately. In practice, use `st.plotly_chart` for Plotly figures, it gives you better control over sizing.

### Input (widgets)

All widget functions return the current value. That value changes on every rerun when the user interacts.

| Function                           | Returns                | Notes                                           |
| ---------------------------------- | ---------------------- | ----------------------------------------------- |
| `st.slider("label", min, max)`     | `int` or `float`       | Add `value=` for default                        |
| `st.selectbox("label", options)`   | selected item          | `options` is a list                             |
| `st.multiselect("label", options)` | list of selected items |                                                 |
| `st.date_input("label")`           | `datetime.date`        |                                                 |
| `st.text_input("label")`           | `str`                  |                                                 |
| `st.number_input("label")`         | `int` or `float`       |                                                 |
| `st.checkbox("label")`             | `bool`                 |                                                 |
| `st.button("label")`               | `bool`                 | `True` only on the rerun triggered by the click |
| `st.radio("label", options)`       | selected item          |                                                 |

Example combining input and display:

```python
import streamlit as st
import pandas as pd

col = st.selectbox("Select column", ["temperature", "demand", "forecast"])
st.write(f"You selected: {col}")
```

### Layout

Streamlit provides three main layout primitives.

**Sidebar**: a collapsible panel on the left. Any widget placed in `st.sidebar` appears there instead of the main column. Commonly used for filters and configuration.

```python
with st.sidebar:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
```

**Columns**: split the horizontal space into N equal (or weighted) columns.

```python
col1, col2 = st.columns(2)

with col1:
    st.metric("R²", 0.91)

with col2:
    st.metric("MAE", 42.3)
```

**Expander**: a collapsible section.

```python
with st.expander("Show raw data"):
    st.dataframe(df)
```

---

## Caching

Because the script reruns on every interaction, caching is essential for any operation that is slow or reads from disk.

Streamlit provides two decorators:

### `st.cache_data`

For functions that return **data**: dataframes, arrays, processed results. The function runs once, subsequent calls with the same arguments return the cached result immediately.

```python
@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

df = load_data("realtime.parquet")
```

Use this for: reading files, API calls, data transformations, any computation that returns a serialisable result.

### `st.cache_resource`

For functions that return **shared resources**: models, database connections, objects that should not be serialised or duplicated. The resource is initialised once and shared across all reruns and all users.

```python
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model("models/best_model.pkl")
```

Use this for: loading ML models, initialising connections, anything heavyweight that should exist as a single instance.

**Rule of thumb:** data → `st.cache_data`, model or connection → `st.cache_resource`.

---

## Session State

Regular Python variables are reset on every rerun. To persist a value across reruns, use `st.session_state`.

`st.session_state` behaves like a dictionary, but values survive reruns within the same browser session.

```python
if "count" not in st.session_state:
    st.session_state["count"] = 0

if st.button("Increment"):
    st.session_state["count"] += 1

st.write(f"Count: {st.session_state['count']}")
```

Common use cases: tracking which page is active, storing user selections that span multiple widgets, accumulating a history of predictions.

---

## Streamlit vs FastAPI

These tools are not alternatives, they solve different problems.

|                | FastAPI                                       | Streamlit                                 |
| -------------- | --------------------------------------------- | ----------------------------------------- |
| **Audience**   | Developers consuming an API                   | Data scientists, analysts, internal users |
| **Interface**  | JSON over HTTP                                | Interactive web UI                        |
| **Use case**   | Production model serving, microservices       | Dashboards, exploration tools, demos      |
| **Frontend**   | None (caller handles it)                      | Built-in                                  |
| **Complexity** | Requires HTTP client, request/response design | Write Python, UI appears                  |

They compose well together. A typical architecture for the energy forecasting project:

```
ENTSO-E data → preprocessing → FastAPI model server
                                      ↓
                              Streamlit dashboard ← calls /predict endpoint
```

The Streamlit app can call your FastAPI endpoint via `requests`, display the returned forecast, and let users explore the results, without duplicating the model loading or inference logic.

---

## Plotly Integration

`st.plotly_chart(fig, use_container_width=True)` renders any Plotly figure inline.

`use_container_width=True` makes the chart fill the available column width, almost always the right choice.

```python
import plotly.graph_objects as go
import streamlit as st

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["demand"], name="Historical"))
fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["forecast"], name="Forecast"))
fig.update_layout(title="Electricity demand — France", xaxis_title="Time", yaxis_title="MW")

st.plotly_chart(fig, use_container_width=True)
```

For time series with two overlapping traces (historical demand + forecast), `go.Figure` with multiple `go.Scatter` traces is the correct approach, each trace has its own name, color, and can be toggled in the legend.

---

## App Architecture Pattern

For a non-trivial app like the energy forecasting dashboard, a clean file structure is:

```
project/
├── app.py              ← main Streamlit script
├── data/
│   └── realtime.parquet
├── models/
│   └── best_model.pkl
└── requirements.txt    ← streamlit, plotly, pandas, joblib, numpy
```

Inside `app.py`, the recommended structure:

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# 1. Page config (must be first Streamlit call)
st.set_page_config(page_title="Energy Forecast", layout="wide")

# 2. Load resources once (cached)
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_data
def load_data():
    return pd.read_parquet("data/realtime.parquet")

model = load_model()
df = load_data()

# 3. Sidebar — filters and configuration
with st.sidebar:
    ...

# 4. Main content — metrics, charts, tables
st.title("France electricity demand forecast")
...
```

`st.set_page_config` must be the first Streamlit call in the script. It controls the browser tab title, the layout (wide or centered), and the sidebar default state.

---

## Deployment (Overview)

For production beyond local use, two main options:

**Streamlit Community Cloud**: free hosting for public apps. Connect a GitHub repository, and Streamlit deploys automatically on every push. Suitable for demos and non-sensitive projects. No Docker required.

**Docker**: the same pattern as your FastAPI deployments. The Dockerfile installs Streamlit and your dependencies, copies the app, and exposes port 8501. Use `--server.address 0.0.0.0` (equivalent to FastAPI's `--host 0.0.0.0`) so the server accepts connections from outside the container.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

---

## Connections to Other Tools

- **Pandas / NumPy**: data manipulation happens in Python as usual; Streamlit only handles the display layer
- **Plotly**: the recommended charting library for interactive time series in Streamlit
- **joblib**: model loading follows the same pattern as the FastAPI deployments: load once at startup, use on every request (every rerun in Streamlit's terms)
- **FastAPI**: complementary, Streamlit calls FastAPI endpoints via `requests` when you want a clean separation between model serving and UI

---

## Review Questions

1. What happens in Streamlit when a user moves a slider? Describe the full execution cycle.

2. You have a function that reads a 500 MB parquet file. What decorator do you use and why? What would happen without it?

3. You have a function that loads a 200 MB scikit-learn model. What decorator do you use, and how does it differ from the previous answer?

4. A user clicks a button, and you want to store that a click happened so a later widget can react to it. Why can't you use a regular Python variable? What do you use instead?

5. What is the difference in use case between `st.write` and `st.plotly_chart`? When would you choose one over the other?

6. You want to display two metrics side by side (R² on the left, MAE on the right). What Streamlit layout primitive do you use?

7. In your energy forecasting app, you want to display a chart of historical demand and a separate chart of the forecast. What Plotly object type do you use, and how do you overlay two traces on the same figure?

8. Why must `st.set_page_config` be the first Streamlit call in the script?

9. You want to move the date range selector out of the main page and into a collapsible left panel. What Streamlit feature do you use?

10. You have a FastAPI endpoint at `http://localhost:8000/predict` that returns a forecast. How would you call it from a Streamlit app, and what library would you use?
