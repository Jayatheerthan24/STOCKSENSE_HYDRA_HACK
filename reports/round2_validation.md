# StockSense - Round 2 Quality & Compliance Validation Report

## Validation Checklist

| Verification Check | Status | Empirical Observation |
| :--- | :--- | :--- |
| **No duplicate date-store-product rows in modeling dataset** | **PASS** | Found 0 duplicate grain rows. |
| **Train/Val/Test Chronological Order (No temporal overlap)** | **PASS** | Train Max: 2026-08-11, Val Min: 2026-08-12, Val Max: 2026-08-17, Test Min: 2026-08-18 |
| **Saved models can be loaded successfully** | **PASS** | Joblib models loaded successfully. |
| **Predictions contain no unexpected NaN values** | **PASS** | NaN demand: 0, NaN stockout: 0. |
| **Recommendation quantities are non-negative** | **PASS** | Found 0 negative recommendation quantities. |
| **Stock-out probability bounded between 0 and 1** | **PASS** | All probabilities in [0.0, 1.0]. |
| **Selected model metrics saved to reports/** | **PASS** | Metrics CSV files present. |

---

## Technical Notes & Limitations
- **Lag Window Trimming:** The initial 14 calendar days were trimmed to construct `lag_14` and 14-day rolling statistics without data leakage.
- **Target Window Trimming:** The final 7 calendar days were trimmed from training targets to maintain exact 7-day forward demand sums.
- **Model Storage:** Trained scikit-learn/XGBoost pipelines saved to `models/demand_forecasting_model.joblib` and `models/stockout_risk_model.joblib`.
