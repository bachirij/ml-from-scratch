"""
utils/data_loader.py - Centralised data loading for the dashboard.

Rules:
- All data loading happens here - never directly in tabs or dashboard.py
- Every function is cached with an appropriate TTL
- Functions raise FileNotFoundError on missing data — tabs handle the error

TTL guidelines:
- Real-time data (refreshes every few minutes): ttl=300
- Daily/historical data (stable within a session): ttl=3600
- Static files (model artifacts, config): ttl=0 (cached for the session lifetime)
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path configuration
# Resolve paths relative to this file so the app works from any working dir
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_main_dataset() -> pd.DataFrame:
    """Load the main dataset.

    Returns:
        DataFrame sorted by datetime ascending.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = DATA_DIR / "dataset.parquet"     # Adapt filename and format

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_realtime() -> pd.DataFrame:
    """Load a real-time rolling window file.
    Refreshes every 5 minutes (ttl=300).

    Returns:
        DataFrame with non-null values, sorted by datetime ascending.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = DATA_DIR / "realtime.parquet"    # Adapt path

    if not path.exists():
        raise FileNotFoundError(f"Realtime file not found: {path}")

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[df["value"].notna()].reset_index(drop=True)   # Adapt column name


@st.cache_resource
def load_model(filename: str):
    """Load a serialised model from the models directory.
    Cached as a shared resource: loaded once, never copied.

    Args:
        filename: Model filename (e.g. "best_model.pkl").

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    import joblib

    path = MODELS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)