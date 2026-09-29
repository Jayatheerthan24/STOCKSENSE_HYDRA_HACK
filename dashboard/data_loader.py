"""
Streamlit Data & Model Loader module for StockSense Round 3 Dashboard.
Implements robust caching for processed datasets, trained models, metrics, and feature importances.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
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
        st.error(f"Modeling dataset file not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(ttl=3600)
def load_manager_recommendations() -> pd.DataFrame:
    """Load manager_recommendations.csv from Round 2."""
    path = DATA_DIR / "manager_recommendations.csv"
    if not path.exists():
        st.error(f"Manager recommendations file not found at {path}")
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


@st.cache_data(ttl=3600)
def load_model_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load demand_model_metrics.csv and stockout_model_metrics.csv."""
    dem_path = REPORTS_DIR / "demand_model_metrics.csv"
    so_path = REPORTS_DIR / "stockout_model_metrics.csv"

    dem_df = pd.read_csv(dem_path) if dem_path.exists() else pd.DataFrame()
    so_df = pd.read_csv(so_path) if so_path.exists() else pd.DataFrame()

    return dem_df, so_df


@st.cache_data(ttl=3600)
def load_feature_importances() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load demand_feature_importance.csv and stockout_feature_importance.csv."""
    dem_imp_path = REPORTS_DIR / "demand_feature_importance.csv"
    so_imp_path = REPORTS_DIR / "stockout_feature_importance.csv"

    dem_imp = pd.read_csv(dem_imp_path) if dem_imp_path.exists() else pd.DataFrame()
    so_imp = pd.read_csv(so_imp_path) if so_imp_path.exists() else pd.DataFrame()

    return dem_imp, so_imp


@st.cache_resource
def load_trained_models():
    """Load trained joblib pipelines for demand and stockout risk models."""
    dem_model_path = MODELS_DIR / "demand_forecasting_model.joblib"
    so_model_path = MODELS_DIR / "stockout_risk_model.joblib"

    dem_model = joblib.load(dem_model_path) if dem_model_path.exists() else None
    so_model = joblib.load(so_model_path) if so_model_path.exists() else None

    return dem_model, so_model
