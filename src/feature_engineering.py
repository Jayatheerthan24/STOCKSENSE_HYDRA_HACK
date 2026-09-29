"""
Feature engineering and dataset preparation module for StockSense Round 2.
Constructs time-based, lag, rolling, trend, promotion, inventory, price, and external features,
along with forward-looking targets without data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from src.utils import PROCESSED_DATA_DIR, load_raw_dataset


def build_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct time-aware features and targets on the Master Analytics Dataset.
    
    Master Grain: ONE ROW = ONE DATE x ONE STORE_ID x ONE PRODUCT_ID.
    
    Calculates:
    - Time-based features
    - Grouped chronologically shifted Lags & Rolling demand statistics
    - Demand trend & volatility (CV) metrics
    - Promotion & Pricing features
    - Inventory gap & ratio metrics
    - Targets: next_7_day_demand (forward 7-day sum) & stockout_flag (closing <= 0)
    
    Returns:
        pd.DataFrame: Engineered DataFrame
    """
    df = df.copy()

    # 1. Datetime conversion & Chronological Sorting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)

    # 2. Basic Imputation for Base Variables
    df['daily_units_sold'] = df['daily_units_sold'].fillna(0)
    if 'mrp' in df.columns:
        df['avg_selling_price'] = df['avg_selling_price'].fillna(df['mrp'])
    else:
        df['avg_selling_price'] = df['avg_selling_price'].fillna(df['avg_selling_price'].median())
    df['avg_discount_pct'] = df['avg_discount_pct'].fillna(0.0)

    # 3. Time-Based Features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # 4. Grouped Lags & Rolling Demand Features (Strict Chronological Shifting)
    grouped = df.groupby(['store_id', 'product_id'])['daily_units_sold']

    df['lag_1'] = grouped.shift(1)
    df['lag_2'] = grouped.shift(2)
    df['lag_3'] = grouped.shift(3)
    df['lag_7'] = grouped.shift(7)
    df['lag_14'] = grouped.shift(14)

    # Rolling statistics use shifted values (shift 1) to prevent leakage of current day
    df['rolling_mean_3'] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df['rolling_mean_7'] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_mean_14'] = grouped.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df['rolling_std_7'] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std()).fillna(0)
    df['rolling_std_14'] = grouped.transform(lambda s: s.shift(1).rolling(14, min_periods=1).std()).fillna(0)

    # 5. Demand Trend Features
    df['demand_change_1d'] = df['lag_1'] - df['lag_2']
    df['demand_change_7d'] = df['lag_1'] - df['lag_7']
    df['demand_cv_7'] = df['rolling_std_7'] / (df['rolling_mean_7'] + 1e-5)

    # 6. Promotion Features & Lags
    promo_grp = df.groupby(['store_id', 'product_id'])['promotion_flag']
    df['promotion_lag_1'] = promo_grp.shift(1).fillna(0).astype(int)
    df['promotion_lag_7'] = promo_grp.shift(7).fillna(0).astype(int)

    # 7. Inventory Features
    if 'closing' in df.columns and 'reorder_lvl' in df.columns:
        df['inventory_gap'] = df['closing'] - df['reorder_lvl']
        df['inventory_ratio'] = np.where(df['reorder_lvl'] > 0, df['closing'] / df['reorder_lvl'], 0.0)
    else:
        df['inventory_gap'] = 0.0
        df['inventory_ratio'] = 0.0

    # 8. Price Features
    if 'avg_selling_price' in df.columns and 'mrp' in df.columns:
        df['price_discount_ratio'] = np.where(df['mrp'] > 0, df['avg_selling_price'] / df['mrp'], 1.0)
    else:
        df['price_discount_ratio'] = 1.0

    # 9. TARGET CREATION
    # Target 1: next_7_day_demand (Forward 7-day sum of daily_units_sold from t+1 to t+7)
    # Using reverse rolling sum shifted by -7
    def calc_future_demand(s: pd.Series) -> pd.Series:
        # s.iloc[::-1] reverses series, rolling(7) sums backward, then reverse back and shift -1
        rev_sum = s.iloc[::-1].rolling(7, min_periods=7).sum().iloc[::-1]
        return rev_sum.shift(-1)

    df['next_7_day_demand'] = df.groupby(['store_id', 'product_id'])['daily_units_sold'].transform(calc_future_demand)

    # Target 2: stockout_flag (1 when closing inventory <= 0, else 0)
    if 'closing' in df.columns:
        df['stockout_flag'] = (df['closing'] <= 0).astype(int)
    else:
        df['stockout_flag'] = 0

    return df


def prepare_modeling_dataset(master_path: Path = PROCESSED_DATA_DIR / "master_dataset.csv") -> pd.DataFrame:
    """
    Load master_dataset.csv, build feature matrix, drop invalid target/lag window rows,
    and save to data/processed/modeling_dataset.csv.
    
    Returns:
        pd.DataFrame: Valid Modeling Dataset
    """
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found at {master_path}")

    raw_master = pd.read_csv(master_path)
    df_feat = build_features_and_targets(raw_master)

    # Drop rows where lag features (e.g. lag_14) or target (next_7_day_demand) are NaN
    # For initial lag_14, first 14 days per product are NaN; for next_7_day_demand, last 7 days are NaN.
    valid_modeling_df = df_feat.dropna(subset=['next_7_day_demand', 'lag_14']).copy()

    # Sort chronologically
    valid_modeling_df = valid_modeling_df.sort_values(by=['date', 'store_id', 'product_id']).reset_index(drop=True)

    output_path = PROCESSED_DATA_DIR / "modeling_dataset.csv"
    valid_modeling_df.to_csv(output_path, index=False)

    print("\n==========================================")
    print(" MODELING DATASET CREATION SUMMARY")
    print("==========================================")
    print(f"• Input Master Rows: {len(raw_master)}")
    print(f"• Valid Modeling Rows: {len(valid_modeling_df)}")
    print(f"• Total Features & Targets: {len(valid_modeling_df.columns)}")
    print(f"• Date Range: {valid_modeling_df['date'].min().strftime('%Y-%m-%d')} to {valid_modeling_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"• Target next_7_day_demand Range: {valid_modeling_df['next_7_day_demand'].min():.0f} to {valid_modeling_df['next_7_day_demand'].max():.0f}")
    print(f"• Target stockout_flag Distribution: {valid_modeling_df['stockout_flag'].value_counts().to_dict()}")
    print(f"[OK] Saved modeling dataset to: {output_path}")

    return valid_modeling_df


if __name__ == "__main__":
    prepare_modeling_dataset()
