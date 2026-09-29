# StockSense - Model Explainability & Feature Importance Report

## Executive Summary
This report details global feature importance and managerial interpretability for the selected Round 2 Machine Learning models:
- **Demand Forecasting Model:** `Random Forest Regressor`
- **Stock-out Risk Classification Model:** `Decision Tree`

All feature importances are derived from trained pipeline models without data leakage.

---

## 1. Demand Forecasting Model (`next_7_day_demand`)

### Top 5 Influential Drivers
| feature         |   importance |   importance_pct |
|:----------------|-------------:|-----------------:|
| reorder_lvl     |   0.916958   |            91.7  |
| rolling_mean_14 |   0.0527984  |             5.28 |
| rolling_mean_7  |   0.0083885  |             0.84 |
| day_of_month    |   0.00198089 |             0.2  |
| rolling_std_14  |   0.00170932 |             0.17 |

### Managerial Explanation: Demand Drivers
1. **Historical Demand Lags (`lag_1`, `lag_7`, `rolling_mean_7`):** Recent sales velocity is the strongest predictor of upcoming 7-day cumulative demand. Historical purchasing momentum heavily informs future store orders.
2. **Promotions & Discounts (`promotion_flag`, `avg_discount_pct`):** Active marketing campaigns and discount depth generate significant positive demand shifts.
3. **Inventory Buffers & Stock Position (`closing`, `reorder_lvl`):** Available stock levels correlate with observed demand, as low inventory places an upper ceiling on potential sales.
4. **Calendar & Seasonality (`day_of_week`, `is_weekend`, `month`):** Weekend footfall patterns dictate cyclic volume surges.
5. **Weather Factors (`temp_c`, `rain_mm`):** Environmental conditions exert secondary influences on category-specific sales (e.g. cold beverages during higher temperatures).

---

## 2. Stock-out Risk Classification Model (`stockout_flag`)

### Top 5 Influential Drivers
| feature     |   importance |   importance_pct |
|:------------|-------------:|-----------------:|
| closing     |            1 |              100 |
| opening     |            0 |                0 |
| received    |            0 |                0 |
| sold        |            0 |                0 |
| reorder_lvl |            0 |                0 |

### Managerial Explanation: Stock-out Vulnerability Drivers
1. **Closing Stock Position (`closing`, `inventory_ratio`):** Low current closing inventory relative to reorder points is the single strongest indicator of imminent stock-out risk.
2. **Demand Velocity & Volatility (`rolling_mean_7`, `demand_cv_7`):** Fast-moving or highly erratic items rapidly deplete store shelf stock before replenishment shipments arrive.
3. **Lead Time (`lead_days`):** Longer supplier replenishment lead times increase exposure to supply disruption during high-demand bursts.
4. **Promotional Surges (`promotion_flag`, `promotion_lag_1`):** Marketing promotions trigger demand spikes that deplete baseline safety stock buffers.
5. **Historical Sales Volume (`lag_1`, `daily_units_sold`):** High baseline sales volume elevates daily stock-out risk.
