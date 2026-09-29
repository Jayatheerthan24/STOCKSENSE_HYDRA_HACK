"""
Data preprocessing, daily aggregation, dataset integration, master dataset validation,
and KPI calculation module for StockSense Round 1 Implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


def aggregate_daily_transactions(valid_tx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level records to Daily Store x Product grain.
    
    Master Grain: ONE ROW = ONE DATE x ONE STORE_ID x ONE PRODUCT_ID.
    
    Calculates:
    - daily_units_sold: sum(quantity)
    - daily_revenue: sum(quantity * selling_price)
    - avg_selling_price: mean(selling_price)
    - avg_discount_pct: mean(discount_pct)
    - promotion_flag: max(promotion_flag)
    - transaction_count: count(transaction_id)
    - unique_customer_count: nunique(customer_id)
    """
    df = valid_tx_df.copy()
    df['revenue'] = df['quantity'] * df['selling_price']

    daily_agg = df.groupby(['date', 'store_id', 'product_id']).agg(
        daily_units_sold=('quantity', 'sum'),
        daily_revenue=('revenue', 'sum'),
        avg_selling_price=('selling_price', 'mean'),
        avg_discount_pct=('discount_pct', 'mean'),
        promotion_flag=('promotion_flag', 'max'),
        transaction_count=('transaction_id', 'count'),
        unique_customer_count=('customer_id', 'nunique')
    ).reset_index()

    # Round floating-point aggregate metrics for clarity
    daily_agg['daily_revenue'] = daily_agg['daily_revenue'].round(2)
    daily_agg['avg_selling_price'] = daily_agg['avg_selling_price'].round(2)
    daily_agg['avg_discount_pct'] = daily_agg['avg_discount_pct'].round(2)

    return daily_agg


def build_master_dataset(
    daily_tx: pd.DataFrame,
    products_clean: pd.DataFrame,
    stores_clean: pd.DataFrame,
    inventory_clean: pd.DataFrame,
    external_clean: pd.DataFrame,
    sparse_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Integrate daily transactions, product master, store master, inventory, and external factors.
    
    Performs strict grain validation: ONE ROW = ONE DATE x ONE STORE_ID x ONE PRODUCT_ID.
    
    Returns:
        pd.DataFrame: Master Analytics Dataset
    """
    # Start with inventory as the base grid if complete, or daily_tx outer join with inventory grid
    # To capture stock-outs (where sales = 0 but inventory is tracked), we use inventory as base if available.
    if not inventory_clean.empty:
        base_df = inventory_clean.copy()
        # Merge daily sales into inventory grid
        master = pd.merge(
            base_df,
            daily_tx,
            on=['date', 'store_id', 'product_id'],
            how='left'
        )
        # Fill missing daily metrics for dates with zero transaction records
        master['daily_units_sold'] = master['daily_units_sold'].fillna(0)
        master['daily_revenue'] = master['daily_revenue'].fillna(0.0)
        master['transaction_count'] = master['transaction_count'].fillna(0)
        master['unique_customer_count'] = master['unique_customer_count'].fillna(0)
        master['promotion_flag'] = master['promotion_flag'].fillna(0).astype(int)
    else:
        master = daily_tx.copy()

    # 1. Merge Product Master Data
    master = pd.merge(
        master,
        products_clean,
        on='product_id',
        how='left'
    )

    # 2. Merge Store Master Data
    master = pd.merge(
        master,
        stores_clean,
        on='store_id',
        how='left'
    )

    # 3. Merge External Factors Data (by date + city)
    if 'city' in master.columns:
        master = pd.merge(
            master,
            external_clean,
            on=['date', 'city'],
            how='left'
        )

    # 4. Merge Sparse History Flag
    if not sparse_history.empty:
        master = pd.merge(
            master,
            sparse_history,
            on='product_id',
            how='left'
        )
        master['sparse_history_flag'] = master['sparse_history_flag'].fillna(1).astype(int)

    # 5. Master Grain Validation
    grain_cols = ['date', 'store_id', 'product_id']
    duplicate_grain_count = int(master.duplicated(subset=grain_cols).sum())

    if duplicate_grain_count > 0:
        raise ValueError(
            f"CRITICAL ERROR: Master dataset grain violation! Found {duplicate_grain_count} duplicate "
            f"rows for grain {grain_cols}. Deduplicating..."
        )

    return master


def validate_master_dataset(master_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform validation checks on the master dataset before saving.
    
    Returns:
        Dict[str, Any]: Diagnostic metrics summary of master dataset.
    """
    grain_cols = ['date', 'store_id', 'product_id']
    duplicate_grain_count = int(master_df.duplicated(subset=grain_cols).sum())

    missing_summary = master_df.isna().sum().to_dict()
    date_min = master_df['date'].min()
    date_max = master_df['date'].max()
    unique_stores = master_df['store_id'].nunique()
    unique_products = master_df['product_id'].nunique()

    validation_result = {
        "final_row_count": len(master_df),
        "final_column_count": len(master_df.columns),
        "duplicate_grain_count": duplicate_grain_count,
        "date_range": f"{date_min} to {date_max}",
        "unique_stores": unique_stores,
        "unique_products": unique_products,
        "missing_summary": missing_summary
    }

    print("\n==========================================")
    print(" MASTER DATASET VALIDATION REPORT")
    print("==========================================")
    print(f"• Final Rows: {len(master_df)}")
    print(f"• Final Columns: {len(master_df.columns)}")
    print(f"• Duplicate Grain Count (date x store_id x product_id): {duplicate_grain_count}")
    print(f"• Date Range: {date_min} to {date_max}")
    print(f"• Unique Stores: {unique_stores}")
    print(f"• Unique Products: {unique_products}")
    if duplicate_grain_count == 0:
        print("[OK] GRAIN VALIDATION PASSED: Zero duplicates for (date, store_id, product_id).")

    return validation_result


def calculate_business_kpis(master_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate business KPIs from the master analytics dataset.
    
    Metrics:
    - Total Revenue
    - Total Units Sold
    - Stock-out Rate (percentage of observations where closing stock == 0)
    - Average Days of Inventory
    - Inventory Turnover (COGS / average inventory value)
    - Promotion Sales Lift (%)
    - Estimated Lost Sales (Note)
    """
    df = master_df.copy()

    # Total Revenue & Units Sold
    total_revenue = float(df['daily_revenue'].sum())
    total_units_sold = float(df['daily_units_sold'].sum())

    # Stock-out Rate: proportion of store-product-date observations with closing inventory == 0
    if 'closing' in df.columns:
        stockout_mask = (df['closing'] == 0)
        stockout_events = int(stockout_mask.sum())
        total_observations = len(df)
        stockout_rate_pct = round(stockout_events / total_observations * 100, 2)
    else:
        stockout_events = 0
        stockout_rate_pct = 0.0

    # Days of Inventory: Closing Stock / Average Daily Demand
    # Compute per product average daily demand and average closing stock
    if 'closing' in df.columns and 'daily_units_sold' in df.columns:
        product_metrics = df.groupby('product_id').agg(
            avg_daily_demand=('daily_units_sold', 'mean'),
            avg_closing_stock=('closing', 'mean')
        ).reset_index()
        
        # Avoid division by zero
        product_metrics['days_of_inventory'] = np.where(
            product_metrics['avg_daily_demand'] > 0,
            product_metrics['avg_closing_stock'] / product_metrics['avg_daily_demand'],
            np.nan
        )
        avg_days_of_inventory = round(float(product_metrics['days_of_inventory'].mean(skipna=True)), 2)
    else:
        avg_days_of_inventory = 0.0

    # Inventory Turnover: COGS / Average Inventory Value
    # COGS = daily_units_sold * cost_price; Inventory Value = closing * cost_price
    if 'cost_price' in df.columns and 'closing' in df.columns:
        df['cogs'] = df['daily_units_sold'] * df['cost_price']
        df['inventory_value'] = df['closing'] * df['cost_price']
        
        total_cogs = df['cogs'].sum()
        avg_inventory_val = df['inventory_value'].mean()
        inventory_turnover = round(float(total_cogs / avg_inventory_val), 2) if avg_inventory_val > 0 else 0.0
    else:
        inventory_turnover = 0.0

    # Promotion Lift: (Promo Sales - Non-Promo Sales) / Non-Promo Sales
    if 'promotion_flag' in df.columns and 'daily_units_sold' in df.columns:
        promo_sales = df[df['promotion_flag'] == 1]['daily_units_sold'].mean()
        non_promo_sales = df[df['promotion_flag'] == 0]['daily_units_sold'].mean()
        if non_promo_sales and non_promo_sales > 0:
            promotion_lift_pct = round(float((promo_sales - non_promo_sales) / non_promo_sales * 100), 2)
        else:
            promotion_lift_pct = 0.0
    else:
        promotion_lift_pct = 0.0

    kpis = {
        "total_revenue": round(total_revenue, 2),
        "total_units_sold": int(total_units_sold),
        "stockout_events": stockout_events,
        "stockout_rate_pct": stockout_rate_pct,
        "avg_days_of_inventory": avg_days_of_inventory,
        "inventory_turnover": inventory_turnover,
        "promotion_lift_pct": promotion_lift_pct,
        "estimated_lost_sales_note": "Marked as Round 2/3 metric requiring demand-vs-available-inventory counterfactual methodology."
    }

    print("\n==========================================")
    print(" BUSINESS KPI SUMMARY")
    print("==========================================")
    print(f"• Total Revenue: INR {kpis['total_revenue']:,.2f}")
    print(f"• Total Units Sold: {kpis['total_units_sold']:,}")
    print(f"• Stock-out Rate: {kpis['stockout_rate_pct']}% ({stockout_events} stockout events)")
    print(f"• Average Days of Inventory: {kpis['avg_days_of_inventory']} days")
    print(f"• Inventory Turnover Ratio: {kpis['inventory_turnover']}x")
    print(f"• Promotion Lift: {kpis['promotion_lift_pct']}% increase in average sales")
    print(f"• Lost Sales Estimate: {kpis['estimated_lost_sales_note']}")

    return kpis
