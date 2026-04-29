"""
tabs/overview.py - Overview tab.

Each tab exposes a single public function: render().
render() is called by dashboard.py when this tab is active.

Responsibilities:
- Layout and widgets for this tab only
- Calls data loading functions from utils/data_loader.py
- Calls chart building functions from utils/charts.py
- No st.set_page_config here (belongs in dashboard.py only)
"""

import streamlit as st
from utils.data_loader import load_main_dataset
from utils.charts import line_chart


def render() -> None:
    """Render the overview page."""

    st.header("Overview")

    # ------------------------------------------------------------------
    # Load data (cached in data_loader - no performance cost on rerun)
    # ------------------------------------------------------------------
    try:
        df = load_main_dataset()
    except FileNotFoundError as e:
        st.error(f"Data unavailable: {e}")
        return  # Stop rendering this tab - avoid downstream errors

    # ------------------------------------------------------------------
    # Sidebar filters (scoped to this tab)
    # Place tab-specific filters inside the existing sidebar block
    # ------------------------------------------------------------------
    with st.sidebar:
        st.subheader("Filters")
        # Example: date range, category selector, threshold slider
        # selected_category = st.selectbox("Category", df["category"].unique())

    # ------------------------------------------------------------------
    # Metrics row
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total rows",
            value=f"{len(df):,}",
            help="Number of records in the dataset.",
        )
    with col2:
        st.metric(
            label="Metric B",
            value="...",
            help="Description of metric B.",
        )
    with col3:
        st.metric(
            label="Metric C",
            value="...",
            help="Description of metric C.",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Main chart
    # ------------------------------------------------------------------
    fig = line_chart(df, x_col="datetime", y_col="value", title="Overview")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Raw data (collapsed by default)
    # ------------------------------------------------------------------
    with st.expander("Show raw data"):
        st.dataframe(df, use_container_width=True)