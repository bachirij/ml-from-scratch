"""
utils/charts.py - Reusable Plotly chart builders for the dashboard.

Rules:
- Functions return a go.Figure - never call st.plotly_chart here
- Rendering belongs in tabs - chart builders only build the figure
- Each function accepts a DataFrame and column name arguments
  so they work with any dataset that follows the expected schema
"""

import pandas as pd
import plotly.graph_objects as go


def line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    y_label: str = "",
    name: str = "Value",
    color: str = "#1f77b4",
) -> go.Figure:
    """Single-trace line chart for time series data.

    Args:
        df:      DataFrame containing the data.
        x_col:   Column name for the x-axis (typically datetime).
        y_col:   Column name for the y-axis.
        title:   Chart title.
        y_label: Y-axis label.
        name:    Trace name shown in the legend.
        color:   Line color (hex string).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="lines",
        name=name,
        line=dict(color=color, width=1.5),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_label,
        hovermode="x unified",      # Single hover tooltip across all traces
        legend=dict(orientation="h", y=1.1),
    )

    return fig


def forecast_chart(
    df: pd.DataFrame,
    x_col: str,
    actual_col: str,
    forecast_col: str,
    title: str = "",
    y_label: str = "",
) -> go.Figure:
    """Two-trace chart overlaying actual values and forecast.
    Typical use: historical demand + model prediction on the same axis.

    Args:
        df:           DataFrame containing both columns.
        x_col:        Column name for the x-axis (datetime).
        actual_col:   Column name for the actual values trace.
        forecast_col: Column name for the forecast trace.
        title:        Chart title.
        y_label:      Y-axis label.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[actual_col],
        mode="lines",
        name="Actual",
        line=dict(color="#1f77b4", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[forecast_col],
        mode="lines",
        name="Forecast",
        line=dict(color="#ff7f0e", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )

    return fig


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    y_label: str = "",
    color: str = "#1f77b4",
) -> go.Figure:
    """Simple bar chart.

    Args:
        df:      DataFrame containing the data.
        x_col:   Column name for the x-axis (categories or dates).
        y_col:   Column name for the y-axis (values).
        title:   Chart title.
        y_label: Y-axis label.
        color:   Bar color (hex string).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        marker_color=color,
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_label,
    )

    return fig