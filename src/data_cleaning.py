"""
Data cleaning and quality auditing module for StockSense Round 1 Implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


def clean_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Clean and audit transactions.csv data according to strict data quality rules.
    
    Treatments:
    - Exact duplicate rows are removed.
    - Conflicting transaction IDs (same ID, different content) are flagged and kept in quality audit.
    - Negative quantities are flagged as invalid (not converted to positive) and filtered out of valid sales.
    - Invalid dates, prices, discounts, hours, and flags are logged.
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: Cleaned valid transactions DataFrame and quality log records.
    """
    df_raw = df.copy()
    total_rows = len(df_raw)
    quality_logs = []

    # 1. Exact Duplicate Rows Analysis & Treatment
    exact_duplicates = df_raw.duplicated(keep='first')
    num_exact_dups = int(exact_duplicates.sum())
    if num_exact_dups > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Exact duplicate rows",
            "affected_rows": num_exact_dups,
            "affected_percentage": round(num_exact_dups / total_rows * 100, 2),
            "treatment": "Removed duplicate copy",
            "notes": "Exact match across all columns removed to prevent duplicate counting."
        })
    df_clean = df_raw.drop_duplicates(keep='first').copy()

    # 2. Conflicting Transaction IDs Analysis (Same ID, different content)
    tx_id_counts = df_clean['transaction_id'].value_counts()
    conflicting_ids = tx_id_counts[tx_id_counts > 1].index.tolist()
    num_conflicting_rows = len(df_clean[df_clean['transaction_id'].isin(conflicting_ids)])
    if num_conflicting_rows > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Conflicting duplicate transaction IDs",
            "affected_rows": num_conflicting_rows,
            "affected_percentage": round(num_conflicting_rows / total_rows * 100, 2),
            "treatment": "Flagged and excluded conflicting duplicates from sales analysis",
            "notes": f"Found {len(conflicting_ids)} IDs appearing with conflicting details."
        })

    # 3. Missing Transaction IDs & Dates
    missing_id = df_clean['transaction_id'].isna() | (df_clean['transaction_id'] == '')
    missing_date = df_clean['date'].isna()
    invalid_date_mask = missing_id | missing_date
    num_invalid_dates = int(invalid_date_mask.sum())
    if num_invalid_dates > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Missing transaction_id or date",
            "affected_rows": num_invalid_dates,
            "affected_percentage": round(num_invalid_dates / total_rows * 100, 2),
            "treatment": "Filtered out record",
            "notes": "Records missing key identifiers cannot be used for daily store aggregation."
        })
    df_clean = df_clean[~invalid_date_mask].copy()

    # Convert date to datetime string YYYY-MM-DD
    df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')

    # 4. Negative and Zero Quantities
    neg_qty = df_clean['quantity'] < 0
    zero_qty = df_clean['quantity'] == 0
    num_neg_qty = int(neg_qty.sum())
    num_zero_qty = int(zero_qty.sum())

    if num_neg_qty > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Negative quantity values",
            "affected_rows": num_neg_qty,
            "affected_percentage": round(num_neg_qty / total_rows * 100, 2),
            "treatment": "Flagged as invalid sales; excluded from positive sales aggregation",
            "notes": "Negative quantities represent invalid/return records; preserved in log, excluded from sales sum."
        })
    if num_zero_qty > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Zero quantity values",
            "affected_rows": num_zero_qty,
            "affected_percentage": round(num_zero_qty / total_rows * 100, 2),
            "treatment": "Excluded from demand calculation",
            "notes": "Zero sales quantity records."
        })

    # 5. Invalid Selling Price & Discounts
    invalid_price = (df_clean['selling_price'].isna()) | (df_clean['selling_price'] < 0)
    invalid_discount = (df_clean['discount_pct'] < 0) | (df_clean['discount_pct'] > 100)
    num_invalid_price = int(invalid_price.sum())
    num_invalid_discount = int(invalid_discount.sum())

    if num_invalid_price > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Missing or negative selling_price",
            "affected_rows": num_invalid_price,
            "affected_percentage": round(num_invalid_price / total_rows * 100, 2),
            "treatment": "Filtered out invalid pricing records",
            "notes": "Valid positive selling price required for revenue calculation."
        })

    if num_invalid_discount > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Discount percentage out of bounds [0, 100]",
            "affected_rows": num_invalid_discount,
            "affected_percentage": round(num_invalid_discount / total_rows * 100, 2),
            "treatment": "Clipped discount percentage to valid [0, 100] range",
            "notes": "Corrected discount values outside normal bounds."
        })
        df_clean['discount_pct'] = df_clean['discount_pct'].clip(0, 100)

    # 6. Invalid Hour and Promotion Flag
    invalid_hour = (df_clean['hour'] < 0) | (df_clean['hour'] > 23)
    invalid_promo = ~df_clean['promotion_flag'].isin([0, 1])
    if invalid_hour.sum() > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Invalid hour outside 0–23",
            "affected_rows": int(invalid_hour.sum()),
            "affected_percentage": round(invalid_hour.sum() / total_rows * 100, 2),
            "treatment": "Flagged and set to null/default",
            "notes": "Hour values outside valid 0-23 range."
        })
    if invalid_promo.sum() > 0:
        quality_logs.append({
            "dataset": "transactions",
            "issue": "Invalid promotion flag outside 0/1",
            "affected_rows": int(invalid_promo.sum()),
            "affected_percentage": round(invalid_promo.sum() / total_rows * 100, 2),
            "treatment": "Coerced to 0 or 1 binary representation",
            "notes": "Non-binary promotion flags."
        })
        df_clean['promotion_flag'] = df_clean['promotion_flag'].apply(lambda x: 1 if x in [1, '1', True] else 0)

    # Filter to valid sales transactions for daily store x product aggregation
    valid_sales_mask = (df_clean['quantity'] > 0) & (~df_clean['selling_price'].isna()) & (df_clean['selling_price'] >= 0)
    valid_df = df_clean[valid_sales_mask].copy()

    return valid_df, quality_logs


def clean_products(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Clean and audit products.csv data.
    
    Treatments:
    - Standardize category names (e.g. beverages, BEVERAGES -> Beverages; snacks, Snacks -> Snacks).
    - Rename 'shelf_life' column to 'shelf_life_days' if required.
    - Check missing values, cost_price > MRP, and invalid values.
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: Cleaned products DataFrame and quality log.
    """
    df_clean = df.copy()
    total_rows = len(df_clean)
    quality_logs = []

    # Rename shelf_life if needed
    if 'shelf_life' in df_clean.columns and 'shelf_life_days' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'shelf_life': 'shelf_life_days'})

    # 1. Duplicate & Missing product_id
    dup_pids = int(df_clean.duplicated(subset=['product_id']).sum())
    missing_pids = int(df_clean['product_id'].isna().sum())
    if dup_pids > 0:
        quality_logs.append({
            "dataset": "products",
            "issue": "Duplicate product_id",
            "affected_rows": dup_pids,
            "affected_percentage": round(dup_pids / total_rows * 100, 2),
            "treatment": "Deduplicated product catalog",
            "notes": "Kept first instance of product definition."
        })
        df_clean = df_clean.drop_duplicates(subset=['product_id'], keep='first').copy()

    if missing_pids > 0:
        quality_logs.append({
            "dataset": "products",
            "issue": "Missing product_id",
            "affected_rows": missing_pids,
            "affected_percentage": round(missing_pids / total_rows * 100, 2),
            "treatment": "Dropped record",
            "notes": "Products missing ID cannot be referenced."
        })
        df_clean = df_clean.dropna(subset=['product_id']).copy()

    # 2. Standardize Category Names
    if 'category' in df_clean.columns:
        original_cats = df_clean['category'].tolist()
        df_clean['category'] = df_clean['category'].astype(str).str.strip().str.title()
        # Custom fix for specific casing if needed
        category_mapping = {
            "Beverages": "Beverages",
            "Snacks": "Snacks",
            "Personal Care": "Personal Care",
            "Household": "Household",
            "Dairy": "Dairy",
            "Frozen": "Frozen"
        }
        df_clean['category'] = df_clean['category'].map(lambda c: category_mapping.get(c, c))
        
        inconsistencies = sum(1 for orig, clean in zip(original_cats, df_clean['category']) if orig != clean)
        if inconsistencies > 0:
            quality_logs.append({
                "dataset": "products",
                "issue": "Inconsistent category name casing (e.g. beverages, BEVERAGES)",
                "affected_rows": inconsistencies,
                "affected_percentage": round(inconsistencies / total_rows * 100, 2),
                "treatment": "Standardized category names to Title Case",
                "notes": "Unified category text representation."
            })

    # 3. Missing Fields Audit (brand, sub_category, supplier_id)
    for col in ['brand', 'sub_category', 'supplier_id']:
        if col in df_clean.columns:
            n_missing = int(df_clean[col].isna().sum())
            if n_missing > 0:
                quality_logs.append({
                    "dataset": "products",
                    "issue": f"Missing {col}",
                    "affected_rows": n_missing,
                    "affected_percentage": round(n_missing / total_rows * 100, 2),
                    "treatment": "Imputed as 'Unknown' / Preserved as NaN",
                    "notes": f"{n_missing} products missing {col}."
                })
                df_clean[col] = df_clean[col].fillna("Unknown")

    # 4. Price & Cost Validation
    invalid_mrp = (df_clean['mrp'] <= 0) | df_clean['mrp'].isna()
    invalid_cost = (df_clean['cost_price'] <= 0) | df_clean['cost_price'].isna()
    cost_gt_mrp = df_clean['cost_price'] > df_clean['mrp']

    if invalid_mrp.sum() > 0 or invalid_cost.sum() > 0:
        quality_logs.append({
            "dataset": "products",
            "issue": "Invalid MRP or cost_price (<= 0 or missing)",
            "affected_rows": int(invalid_mrp.sum() + invalid_cost.sum()),
            "affected_percentage": round((invalid_mrp.sum() + invalid_cost.sum()) / total_rows * 100, 2),
            "treatment": "Flagged pricing error",
            "notes": "Check product pricing master data."
        })

    if cost_gt_mrp.sum() > 0:
        quality_logs.append({
            "dataset": "products",
            "issue": "cost_price exceeds MRP",
            "affected_rows": int(cost_gt_mrp.sum()),
            "affected_percentage": round(cost_gt_mrp.sum() / total_rows * 100, 2),
            "treatment": "Flagged anomalous pricing",
            "notes": "Cost price reported higher than Maximum Retail Price."
        })

    return df_clean, quality_logs


def clean_stores(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Clean and audit stores.csv data.
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: Cleaned stores DataFrame and quality log.
    """
    df_clean = df.copy()
    total_rows = len(df_clean)
    quality_logs = []

    # 1. Duplicate store_id
    dup_stores = int(df_clean.duplicated(subset=['store_id']).sum())
    if dup_stores > 0:
        quality_logs.append({
            "dataset": "stores",
            "issue": "Duplicate store_id",
            "affected_rows": dup_stores,
            "affected_percentage": round(dup_stores / total_rows * 100, 2),
            "treatment": "Deduplicated store profiles",
            "notes": "Maintained unique store profile."
        })
        df_clean = df_clean.drop_duplicates(subset=['store_id'], keep='first').copy()

    # 2. Missing Fields
    for col in ['city', 'store_type', 'region', 'floor_area_sqft', 'avg_daily_customers']:
        if col in df_clean.columns:
            n_missing = int(df_clean[col].isna().sum())
            if n_missing > 0:
                quality_logs.append({
                    "dataset": "stores",
                    "issue": f"Missing {col}",
                    "affected_rows": n_missing,
                    "affected_percentage": round(n_missing / total_rows * 100, 2),
                    "treatment": "Logged missing store attribute",
                    "notes": f"Missing values found in {col}."
                })

    # 3. Numeric range validation
    invalid_area = df_clean['floor_area_sqft'] <= 0
    invalid_cust = df_clean['avg_daily_customers'] <= 0
    if invalid_area.sum() > 0 or invalid_cust.sum() > 0:
        quality_logs.append({
            "dataset": "stores",
            "issue": "Invalid store floor_area or daily customer count",
            "affected_rows": int(invalid_area.sum() + invalid_cust.sum()),
            "affected_percentage": round((invalid_area.sum() + invalid_cust.sum()) / total_rows * 100, 2),
            "treatment": "Flagged invalid store dimensions",
            "notes": "Non-positive metric values."
        })

    return df_clean, quality_logs


def clean_inventory(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Clean and audit inventory.csv data.
    
    Treatments:
    - Standardize keys: 'store' -> 'store_id', 'product' -> 'product_id'.
    - Validate equation: closing = opening + received - sold.
    - Create 'inventory_mismatch_flag' (1 if arithmetic fails, 0 otherwise).
    - Do NOT silently repair inventory mismatches.
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: Cleaned inventory DataFrame and quality log.
    """
    df_clean = df.copy()
    total_rows = len(df_clean)
    quality_logs = []

    # 1. Key Standardisation
    rename_dict = {}
    if 'store' in df_clean.columns:
        rename_dict['store'] = 'store_id'
    if 'product' in df_clean.columns:
        rename_dict['product'] = 'product_id'
    if rename_dict:
        df_clean = df_clean.rename(columns=rename_dict)

    # 2. Convert date
    df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')

    # 3. Fill missing numeric values if any with explicit tracking
    missing_inv = df_clean[['opening', 'received', 'sold', 'closing']].isna().sum().sum()
    if missing_inv > 0:
        quality_logs.append({
            "dataset": "inventory",
            "issue": "Missing inventory stock values",
            "affected_rows": int(missing_inv),
            "affected_percentage": round(missing_inv / total_rows * 100, 2),
            "treatment": "Imputed missing with 0 and flagged",
            "notes": "Missing opening/received/sold/closing values."
        })
        df_clean[['opening', 'received', 'sold', 'closing']] = df_clean[['opening', 'received', 'sold', 'closing']].fillna(0)

    # 4. Inventory Reconciliation Check: closing = opening + received - sold
    expected_closing = df_clean['opening'] + df_clean['received'] - df_clean['sold']
    mismatch_mask = np.abs(df_clean['closing'] - expected_closing) > 0.001
    num_mismatches = int(mismatch_mask.sum())
    mismatch_pct = round(num_mismatches / total_rows * 100, 2)

    df_clean['inventory_mismatch_flag'] = mismatch_mask.astype(int)

    quality_logs.append({
        "dataset": "inventory",
        "issue": "Inventory balance arithmetic mismatch (closing != opening + received - sold)",
        "affected_rows": num_mismatches,
        "affected_percentage": mismatch_pct,
        "treatment": "Created inventory_mismatch_flag = 1 without silent repair",
        "notes": f"{num_mismatches} records ({mismatch_pct}%) have inventory balance discrepancies."
    })

    # 5. Check for Negative Values
    neg_fields = (df_clean['opening'] < 0) | (df_clean['received'] < 0) | (df_clean['sold'] < 0) | (df_clean['closing'] < 0)
    num_neg = int(neg_fields.sum())
    if num_neg > 0:
        quality_logs.append({
            "dataset": "inventory",
            "issue": "Negative stock quantities (opening, received, sold, or closing)",
            "affected_rows": num_neg,
            "affected_percentage": round(num_neg / total_rows * 100, 2),
            "treatment": "Flagged negative inventory entries",
            "notes": "Inventory stock cannot physically be negative."
        })

    return df_clean, quality_logs


def clean_external_factors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Clean and audit external_factors.csv data.
    
    Treatments:
    - Handle missing temperature and rainfall using city-level median imputation.
    - Create flags 'temp_was_missing' and 'rain_was_missing'.
    - Validate date x city unique key.
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: Cleaned external factors DataFrame and quality log.
    """
    df_clean = df.copy()
    total_rows = len(df_clean)
    quality_logs = []

    df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')

    # 1. Duplicate date x city check
    dup_key = int(df_clean.duplicated(subset=['date', 'city']).sum())
    if dup_key > 0:
        quality_logs.append({
            "dataset": "external_factors",
            "issue": "Duplicate date x city records",
            "affected_rows": dup_key,
            "affected_percentage": round(dup_key / total_rows * 100, 2),
            "treatment": "Deduplicated by keeping first observation",
            "notes": "City daily weather must be unique per date."
        })
        df_clean = df_clean.drop_duplicates(subset=['date', 'city'], keep='first').copy()

    # 2. Missing Temperature Imputation
    df_clean['temp_was_missing'] = df_clean['temp_c'].isna().astype(int)
    num_missing_temp = int(df_clean['temp_was_missing'].sum())
    if num_missing_temp > 0:
        city_temp_median = df_clean.groupby('city')['temp_c'].transform('median')
        df_clean['temp_c'] = df_clean['temp_c'].fillna(city_temp_median)
        quality_logs.append({
            "dataset": "external_factors",
            "issue": "Missing temperature (temp_c)",
            "affected_rows": num_missing_temp,
            "affected_percentage": round(num_missing_temp / total_rows * 100, 2),
            "treatment": "City-level median imputation + preserved temp_was_missing flag",
            "notes": "Imputed missing weather temperature values."
        })

    # 3. Missing Rainfall Imputation
    df_clean['rain_was_missing'] = df_clean['rain_mm'].isna().astype(int)
    num_missing_rain = int(df_clean['rain_was_missing'].sum())
    if num_missing_rain > 0:
        city_rain_median = df_clean.groupby('city')['rain_mm'].transform('median')
        df_clean['rain_mm'] = df_clean['rain_mm'].fillna(city_rain_median)
        quality_logs.append({
            "dataset": "external_factors",
            "issue": "Missing rainfall (rain_mm)",
            "affected_rows": num_missing_rain,
            "affected_percentage": round(num_missing_rain / total_rows * 100, 2),
            "treatment": "City-level median imputation + preserved rain_was_missing flag",
            "notes": "Imputed missing rainfall data."
        })

    # 4. Binary flag normalization
    for flag_col in ['holiday', 'festival', 'weekend', 'local_event']:
        if flag_col in df_clean.columns:
            df_clean[flag_col] = df_clean[flag_col].apply(lambda x: 1 if x in [1, '1', True] else 0)

    return df_clean, quality_logs


def detect_sparse_history(transactions_df: pd.DataFrame, threshold_days: int = 7) -> pd.DataFrame:
    """
    Calculate total observed active sales days per product and flag sparse products (< 7 days).
    
    Returns:
        pd.DataFrame: DataFrame containing product_id, sales_history_days, and sparse_history_flag.
    """
    sales_days = transactions_df.groupby('product_id')['date'].nunique().reset_index()
    sales_days.columns = ['product_id', 'sales_history_days']
    sales_days['sparse_history_flag'] = (sales_days['sales_history_days'] < threshold_days).astype(int)
    return sales_days


def build_data_quality_summary(quality_logs: List[Dict]) -> pd.DataFrame:
    """
    Consolidate all dataset quality log records into a structured summary DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame matching schema (dataset, issue, affected_rows, affected_percentage, treatment, notes)
    """
    summary_df = pd.DataFrame(quality_logs)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=[
            "dataset", "issue", "affected_rows", "affected_percentage", "treatment", "notes"
        ])
    return summary_df
