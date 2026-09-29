# StockSense - Recommendation Engine Logic & Risk Stratification

## Executive Summary
The StockSense Recommendation Engine translates machine learning demand forecasts and stock-out probabilities into actionable inventory replenishment decisions for store managers.

---

## 1. Safety Stock & Reorder Point Formulas

### Formulas
- **Average Daily Demand:** `Forecast 7-Day Demand / 7`
- **Lead Time Demand:** `Average Daily Demand * Lead Days`
- **Safety Stock:** `z * std_demand * sqrt(Lead Days)`
- **Reorder Point:** `Lead Time Demand + Safety Stock`
- **Recommended Reorder Qty:** `max(0, Reorder Point - Closing Inventory)`

### Parameter Specifications
- **Service Level (z):** z = 1.65 (corresponds to a 95% cycle service level target).
- **Demand Volatility (std_demand):** Rolling 7-day demand standard deviation (`rolling_std_7`). Fallback value = 1.0.
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
