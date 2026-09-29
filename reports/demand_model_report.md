# StockSense - Demand Forecasting Model Report

## 1. Executive Summary
The selected demand forecasting model is **Random Forest Regressor**, evaluated on an untouched chronological test set.

## 2. Model Performance Summary

### Validation Models Comparison
| Model                   | Partition   |   MAE |   RMSE |     R2 |   MAPE (%) |
|:------------------------|:------------|------:|-------:|-------:|-----------:|
| Random Forest Regressor | Validation  |  8.12 |  10.98 | 0.9742 |       8.91 |
| Linear Regression       | Validation  |  8.44 |  11.07 | 0.9738 |      25.95 |

### Final Selected Model (Random Forest Regressor) Test Performance
- **MAE:** 10.65 units
- **RMSE:** 13.74 units
- **R² Score:** 0.9597
- **MAPE:** 14.66% (Handling zero demand by clipping denominator to 1.0)

## 3. Target & Feature Specifications
- **Target Variable:** `next_7_day_demand` (Forward-looking 7-day cumulative sum of `daily_units_sold`).
- **Feature Engineering:** Includes 14-day lags, 7-day rolling statistics, demand trend volatility, pricing ratios, promotion indicators, and calendar features.
- **Data Split:** Chronological 70% Train, 15% Validation, 15% Test with zero temporal leakage.
