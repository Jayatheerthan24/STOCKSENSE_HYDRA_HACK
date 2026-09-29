"""
Utility functions and path configurations for StockSense Round 1 Implementation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Define Project Paths using pathlib
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_directories():
    """Ensure all required project subdirectories exist."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_dataset(filename: str) -> pd.DataFrame:
    """
    Load a raw CSV dataset from data/raw/ using relative pathlib path.
    
    Parameters:
        filename (str): Name of the CSV file in data/raw/
        
    Returns:
        pd.DataFrame: Loaded raw DataFrame
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    return pd.read_csv(file_path)


def print_dataset_overview(name: str, df: pd.DataFrame, date_col: str = None, id_col: str = None) -> dict:
    """
    Inspect and return comprehensive data quality overview metrics for a dataset.
    
    Parameters:
        name (str): Dataset name label
        df (pd.DataFrame): Target DataFrame
        date_col (str, optional): Column name containing dates
        id_col (str, optional): Column name containing unique entity IDs
        
    Returns:
        dict: Diagnostic summary dictionary
    """
    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    date_range = "N/A"
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col].dropna())
        if not dates.empty:
            date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"

    unique_ids = df[id_col].nunique() if id_col and id_col in df.columns else "N/A"

    print(f"\n==========================================")
    print(f" DATASET OVERVIEW: {name}")
    print(f"==========================================")
    print(f"• Rows: {rows}")
    print(f"• Columns: {cols}")
    print(f"• Column Names: {list(df.columns)}")
    print(f"• Unique IDs ({id_col}): {unique_ids}")
    print(f"• Date Range: {date_range}")
    print(f"• Total Missing Values: {missing_total}")
    print(f"• Duplicate Rows: {duplicate_rows}")
    print("• Data Types:")
    for col, dtype in df.dtypes.items():
        missing_cnt = df[col].isna().sum()
        print(f"  - {col}: {dtype} (missing: {missing_cnt})")

    return {
        "dataset": name,
        "rows": rows,
        "columns": cols,
        "column_names": list(df.columns),
        "unique_ids": unique_ids,
        "date_range": date_range,
        "missing_total": missing_total,
        "duplicate_rows": duplicate_rows
    }
