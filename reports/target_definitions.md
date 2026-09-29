# StockSense - Round 2 Target Definitions & Specification

## Executive Summary
This document specifies the exact mathematical definitions, construction logic, and data leakage prevention rules for the two target variables used in Round 2 Machine Learning models.

---

## 1. Target 1: `next_7_day_demand` (Demand Forecasting)

### Mathematical Definition
For each Store $s$, Product $p$, and Date $t$:

$$\text{next\_7\_day\_demand}_{s,p,t} = \sum_{k=1}^{7} \text{daily\_units\_sold}_{s,p,t+k}$$

### Construction Logic
- Computed per $(s, p)$ group sorted chronologically by date.
- Sums sales volume strictly over the **following 7 calendar days** ($t+1$ to $t+7$).
- **The current date $t$ is NOT included in the target sum.**
- Observations in the final 7 calendar days of the dataset receive a `NaN` target value and are excluded from training to prevent partial window distortion.

### Data Leakage Prevention
- Features used to predict $\text{next\_7\_day\_demand}_{s,p,t}$ are computed strictly using historical observations up to and including date $t$ (e.g., $t-14$ to $t$).
- No future sales or inventory information from $t+1$ onwards is present in input features.

---

## 2. Target 2: `stockout_flag` (Stock-out Risk Classification)

### Mathematical Definition
For each Store $s$, Product $p$, and Date $t$:

$$\text{stockout\_flag}_{s,p,t} = \begin{cases} 1 & \text{if } \text{closing}_{s,p,t} \le 0 \\ 0 & \text{if } \text{closing}_{s,p,t} > 0 \end{cases}$$

### Construction Logic
- Evaluates closing inventory stock position at the end of day $t$.
- If `closing <= 0`, the observation is classified as a stock-out event ($1$).
- If `closing > 0`, the observation is classified as stock-available ($0$).

### Imbalance Strategy
- Stock-out events constitute ~22.5% of valid modeling observations.
- Class balance is handled during model training using `class_weight='balanced'` in scikit-learn classifiers and `scale_pos_weight` ratio in XGBoost.
- No synthetic oversampling (e.g. SMOTE) was applied prior to chronological dataset splitting to prevent temporal leakage.
