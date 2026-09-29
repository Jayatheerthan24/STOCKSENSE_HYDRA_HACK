"""
Streamlit Data & Model Loader module for StockSense Round 3 Dashboard.
Implements robust caching for processed datasets and recommendations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


@st.cache_data(ttl=3600)
def load_master_dataset() -> pd.DataFrame:
    """Load master_dataset.csv from Round 1."""
    path = DATA_DIR / "master_dataset.csv"
    if not path.exists():
        st.error(f"Master dataset file not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(ttl=3600)
def load_modeling_dataset() -> pd.DataFrame:
    """Load modeling_dataset.csv from Round 2."""
    path = DATA_DIR / "modeling_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(ttl=3600)
def load_manager_recommendations() -> pd.DataFrame:
    """Load manager_recommendations.csv from Round 2."""
    path = DATA_DIR / "manager_recommendations.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(ttl=3600)
def load_data_quality_summary() -> pd.DataFrame:
    """Load data_quality_summary.csv."""
    path = DATA_DIR / "data_quality_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
