"""
Managerial Recommendation Engine module for StockSense Round 2.
Translates Machine Learning demand forecasts and stock-out probabilities into actionable
reorder quantities, safety stock calculations, risk tiers, and natural language recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

from src.utils import PROCESSED_DATA_DIR, REPORTS_DIR


def generate_manager_recommendations(
    eval_df: pd.DataFrame,
    demand_preds: np.ndarray,
    stockout_probs: np.ndarray,
    z_service_level: float = 1.65
) -> pd.DataFrame:
    """
    Generate manager-friendly inventory recommendations for each evaluation observation.
    
    Formulas:
    - average_daily_demand = forecast_demand / 7.0
    - lead_time_demand = average_daily_demand * lead_days
    - safety_stock = z * demand_std * sqrt(lead_days)
    - reorder_point = lead_time_demand + safety_stock
    - recommended_reorder_qty = max(0, reorder_point - closing)
    
    Returns:
        pd.DataFrame: Structured recommendations DataFrame saved to data/processed/manager_recommendations.csv
    """
    df = eval_df.copy().reset_index(drop=True)
    df['forecast_demand'] = np.maximum(0, demand_preds).round(2)
    df['stockout_probability'] = np.clip(stockout_probs, 0.0, 1.0).round(4)

    # 1. Lead Time & Demand Calculations
    df['lead_days'] = df['lead_days'].fillna(3).clip(lower=1)
    df['avg_daily_demand'] = df['forecast_demand'] / 7.0
    df['lead_time_demand'] = df['avg_daily_demand'] * df['lead_days']

    # 2. Safety Stock & Reorder Point
    if 'rolling_std_7' in df.columns:
        demand_std = df['rolling_std_7'].fillna(1.0).apply(lambda x: max(1.0, x))
    else:
        demand_std = 1.0

    df['safety_stock'] = (z_service_level * demand_std * np.sqrt(df['lead_days'])).round(2)
    df['reorder_point'] = (df['lead_time_demand'] + df['safety_stock']).round(2)

    # 3. Recommended Reorder Quantity (Non-negative)
    closing_stock = df['closing'].fillna(0)
    df['recommended_reorder_qty'] = np.maximum(0, (df['reorder_point'] - closing_stock).round(0)).astype(int)

    # 4. Risk Level Stratification (Documented Configurable Thresholds)
    # HIGH: stockout_prob >= 0.50 OR closing <= 0
    # MEDIUM: 0.20 <= stockout_prob < 0.50 OR closing < reorder_lvl
    # LOW: stockout_prob < 0.20
    reorder_lvl = df['reorder_lvl'] if 'reorder_lvl' in df.columns else 0

    conditions = [
        (df['stockout_probability'] >= 0.50) | (closing_stock <= 0),
        ((df['stockout_probability'] >= 0.20) & (df['stockout_probability'] < 0.50)) | (closing_stock < reorder_lvl),
    ]
    choices = ['HIGH', 'MEDIUM']
    df['risk_level'] = np.select(conditions, choices, default='LOW')

    # 5. Top Reasons & Managerial Action Text
    reasons_1 = []
    reasons_2 = []
    actions = []

    for idx, row in df.iterrows():
        r1, r2, act = derive_manager_narrative(row)
        reasons_1.append(r1)
        reasons_2.append(r2)
        actions.append(act)

    df['top_reason_1'] = reasons_1
    df['top_reason_2'] = reasons_2
    df['manager_recommendation'] = actions

    # Select clean export columns
    export_cols = [
        'date', 'store_id', 'product_id', 'category', 'forecast_demand',
        'closing', 'reorder_lvl', 'lead_days', 'stockout_probability',
        'risk_level', 'safety_stock', 'reorder_point', 'recommended_reorder_qty',
        'top_reason_1', 'top_reason_2', 'manager_recommendation'
    ]
    # Filter existing columns
    export_cols = [c for c in export_cols if c in df.columns]
    recs_df = df[export_cols].copy()

    # Save to CSV
    output_path = PROCESSED_DATA_DIR / "manager_recommendations.csv"
    recs_df.to_csv(output_path, index=False)
    print(f"[OK] Saved manager recommendations to: {output_path}")

    # Generate recommendation logic report
    generate_recommendation_logic_report(z_service_level)

    return recs_df


def derive_manager_narrative(row: pd.Series) -> Tuple[str, str, str]:
    """Derive top 2 reasons and manager action string based on actual row values."""
    closing = row.get('closing', 0)
    forecast = row.get('forecast_demand', 0)
    promo = row.get('promotion_flag', 0)
    prob = row.get('stockout_probability', 0.0)
    reorder_qty = row.get('recommended_reorder_qty', 0)
    risk = row.get('risk_level', 'LOW')

    # Reason 1
    if closing <= 0:
        r1 = f"Stock-out active (Closing stock: {closing:.0f} units)"
    elif prob >= 0.50:
        r1 = f"High predicted stock-out probability ({prob:.1%})"
    elif forecast > 100:
        r1 = f"High 7-day predicted demand forecast ({forecast:.0f} units)"
    else:
        r1 = f"Stable closing inventory position ({closing:.0f} units)"

    # Reason 2
    if promo == 1:
        r2 = "Active promotional campaign boosting velocity"
    elif row.get('lead_days', 0) >= 4:
        r2 = f"Long lead time ({row.get('lead_days', 0)} days) requires safety buffer"
    else:
        r2 = f"Reorder trigger point set at {row.get('reorder_point', 0):.0f} units"

    # Recommendation action text
    if risk == 'HIGH':
        act = f"HIGH RISK: Reorder approximately {reorder_qty} units immediately before the next replenishment cycle."
    elif risk == 'MEDIUM':
        act = f"MEDIUM RISK: Monitor closely; projected demand is {forecast:.0f} units. Recommended buffer order: {reorder_qty} units."
    else:
        act = "LOW RISK: Stock position is adequate. No immediate reorder required."

    return r1, r2, act


def generate_recommendation_logic_report(z_val: float) -> str:
    """Generate documentation report saved to reports/recommendation_logic.md."""
    logic_md = f"""# StockSense - Recommendation Engine Logic & Risk Stratification

## Executive Summary
The StockSense Recommendation Engine translates machine learning demand forecasts and stock-out probabilities into actionable inventory replenishment decisions for store managers.

---

## 1. Safety Stock & Reorder Point Formulas

### Formulas
$$\\text{{Average Daily Demand}} = \\frac{{\\text{{Forecast 7-Day Demand}}}}{{7}}$$

$$\\text{{Lead Time Demand}} = \\text{{Average Daily Demand}} \\times \\text{{Lead Days}}$$

$$\\text{{Safety Stock}} = z \\times \\sigma_{\\text{{demand}}} \\times \\sqrt{{\\text{{Lead Days}}}}$$

$$\\text{{Reorder Point}} = \\text{{Lead Time Demand}} + \\text{{Safety Stock}}$$

$$\\text{{Recommended Reorder Qty}} = \\max\\left(0, \\lceil\\text{{Reorder Point}} - \\text{{Closing Inventory}}\\rceil\\right)$$

### Parameter Specifications
- **Service Level ($z$):** $z = {z_val}$ (corresponds to a 95% cycle service level target).
- **Demand Volatility ($\\sigma_{\\text{{demand}}}$):** Rolling 7-day demand standard deviation (`rolling_std_7`). Fallback value = 1.0.
- **Lead Days:** Store/product specific replenishment lead time.

---

## 2. Risk Level Stratification Thresholds

| Risk Level | Trigger Criteria | Managerial Priority | Action Description |
| :--- | :--- | :--- | :--- |
| **HIGH** | `stockout_prob >= 0.50` OR `closing <= 0` | Priority 1 (Urgent) | Reorder immediately to prevent / resolve stock-out. |
| **MEDIUM** | `0.20 <= stockout_prob < 0.50` OR `closing < reorder_lvl` | Priority 2 (Caution) | Monitor inventory position closely; place buffer order. |
| **LOW** | `stockout_prob < 0.20` | Priority 3 (Normal) | Stock levels adequate; no immediate reorder needed. |

---

## 3. Output Schema (`data/processed/manager_recommendations.csv`)

- `date`: Evaluation observation date
- `store_id`: Store profile identifier
- `product_id`: Product catalog identifier
- `category`: Product category
- `forecast_demand`: Machine learning 7-day predicted demand
- `closing`: Current closing inventory
- `stockout_probability`: Machine learning stock-out risk probability
- `risk_level`: Risk tier (`HIGH`, `MEDIUM`, `LOW`)
- `safety_stock`: Calculated safety stock buffer
- `reorder_point`: Calculated reorder trigger point
- `recommended_reorder_qty`: Actionable order quantity
- `top_reason_1`: Primary empirical driver
- `top_reason_2`: Secondary empirical driver
- `manager_recommendation`: Clear natural language action instructions
"""

    output_path = REPORTS_DIR / "recommendation_logic.md"
    output_path.write_text(logic_md, encoding="utf-8")
    print(f"[OK] Saved recommendation logic report to: {output_path}")

    return logic_md
