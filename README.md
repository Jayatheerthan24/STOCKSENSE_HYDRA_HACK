# StockSense – AI-Powered Retail Inventory Intelligence (NovaMart)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**StockSense** is an enterprise-grade AI-powered demand forecasting and stock-out risk management platform developed for the **IntelliData 2026 Data Science Hackathon**. Built specifically for retail operations managers (e.g. NovaMart), StockSense integrates time-aware machine learning models, statistical hypothesis testing, explainable AI, and an automated managerial recommendation engine to prevent stock-outs and optimize inventory turnover.

---

## 📌 Problem Statement & Business Objectives

Retail chains suffer massive revenue losses and customer dissatisfaction due to frequent stock-outs during demand surges and promotional events. Conversely, overstocking capital leads to high holding costs and inventory spoilage.

### Core Objectives
1. **Data Quality & Integration (Round 1):** Clean 5 raw datasets (`transactions`, `products`, `stores`, `inventory`, `external_factors`), reconcile inventory balance equations (`closing = opening + received - sold`), standardize product catalogs, and construct a Master Analytics Dataset at `ONE ROW = ONE DATE x ONE STORE_ID x ONE PRODUCT_ID` grain.
2. **Predictive Machine Learning (Round 2):**
   - **Model 1 (Demand Forecasting):** Predict cumulative 7-day forward sales demand (`next_7_day_demand`).
   - **Model 2 (Stock-out Risk Classification):** Predict probability of stock-out occurrence (`stockout_flag`).
3. **Managerial Action Engine (Round 2 & 3):** Calculate dynamic safety stock ($z=1.65$ for 95% service level), reorder points, and recommended reorder quantities with clear natural language guidance.
4. **Interactive Executive Prototype (Round 3):** Deploy a multi-page Streamlit intelligence dashboard featuring sidebar filtering, risk heatmaps, product drill-downs, model explainability, and executive CSV export capabilities.

---

## 🏗️ Project Architecture & Directory Structure

```
STOCKSENSE_HYDRA_HACK/
├── data/
│   ├── raw/                        # Original raw CSV files
│   └── processed/                  # Output datasets
│       ├── master_dataset.csv      # Unified Master Analytics Dataset (Grain: Date x Store x Product)
│       ├── data_quality_summary.csv # Consolidated data quality audit log
│       ├── modeling_dataset.csv    # Feature engineered & target creation dataset
│       └── manager_recommendations.csv # Actionable inventory reorder recommendations
├── src/                            # Modular Python package
│   ├── utils.py                    # Path management & data inspection utilities
│   ├── data_cleaning.py            # Quality audit, deduplication, weather imputation & sparse flags
│   ├── data_preprocessing.py       # Daily aggregation, master dataset integration & KPI calculation
│   ├── eda.py                      # 10 business-focused matplotlib/seaborn EDA chart generators
│   ├── statistical_analysis.py    # Hypothesis testing (Mann-Whitney U, Kruskal-Wallis, Chi-Square)
│   ├── feature_engineering.py     # Time-aware lags, rolling statistics, pricing & trend features
│   ├── train_models.py             # Master ML training, time split (70/15/15), eval & joblib saving
│   ├── model_explainability.py     # Feature importances & natural language managerial narratives
│   └── recommendation_engine.py   # Safety stock, reorder point & risk level stratification engine
├── dashboard/                      # Round 3 Streamlit Prototype Application
│   ├── app.py                      # Master Streamlit multi-page application
│   ├── data_loader.py              # Streamlit cached data & model loading utilities
│   ├── visualizations.py           # Interactive Plotly chart generators
│   └── dashboard_utils.py          # Dynamic sidebar filters & UI cell formatting helpers
├── notebooks/                      # Executable Jupyter Notebooks
│   ├── 01_data_understanding_eda.ipynb  # Round 1 Data Understanding, Cleaning & EDA
│   └── 02_feature_engineering_ml.ipynb   # Round 2 Feature Engineering, ML & Recommendations
├── models/                         # Serialized Scikit-Learn / XGBoost Joblib Pipelines
│   ├── demand_forecasting_model.joblib
│   └── stockout_risk_model.joblib
├── reports/                        # Executive markdown reports & figures
│   ├── figures/                    # Saved EDA & ML diagnostic PNG charts
│   ├── data_quality_report.md
│   ├── eda_insights.md
│   ├── statistical_analysis.md
│   ├── target_definitions.md
│   ├── model_split.md
│   ├── demand_model_report.md
│   ├── stockout_model_report.md
│   ├── model_explainability.md
│   ├── recommendation_logic.md
│   └── round2_validation.md
├── main.py                         # Round 1 Master Execution Script
├── requirements.txt                # Project dependencies
└── README.md                       # Master Documentation
```

---

## ⚡ Quick Start & Installation Guide

### Prerequisites
- Python 3.11 or higher
- Virtual environment (`.venv`) recommended

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Jayatheerthan24/STOCKSENSE_HYDRA_HACK.git
cd STOCKSENSE_HYDRA_HACK

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Pipelines

#### Run Round 1 Pipeline (Data Cleaning, Integration, EDA & Statistical Tests):
```bash
python main.py
```

#### Run Round 2 Pipeline (Feature Engineering, Time Split, ML Training & Recommendations):
```bash
python -m src.feature_engineering
python -m src.train_models
```

#### Run Round 3 Interactive Streamlit Dashboard:
```bash
streamlit run dashboard/app.py
```

---

## 📊 Model Evaluation Summary

All machine learning models were trained on a **chronological 70% Train partition**, tuned on a **15% Validation partition**, and evaluated ONCE on an untouched **15% Test partition** with zero temporal data leakage.

### Model 1: Demand Forecasting (`target = next_7_day_demand`)
- **Selected Model:** **Random Forest Regressor** (Tuned max_depth=12, n_estimators=100)
- **Test Performance:**
  - **MAE:** `10.65` units
  - **RMSE:** `13.74` units
  - **$R^2$ Score:** `0.9597` (95.97% variance explained)
  - **MAPE:** `14.66%`

### Model 2: Stock-out Risk Classification (`target = stockout_flag`)
- **Selected Model:** **Decision Tree / Random Forest Classifier** (Balanced class weights)
- **Test Performance:**
  - **Accuracy:** `1.0000` (100%)
  - **Precision:** `1.0000` (100%)
  - **Recall:** `1.0000` (100%)
  - **F1-Score:** `1.0000`
  - **ROC-AUC:** `1.0000`

---

## 💡 Recommendation Engine Logic

Reorder quantities are dynamically calculated for store operations:

$$\text{Average Daily Demand} = \frac{\text{Forecast 7-Day Demand}}{7}$$

$$\text{Safety Stock} = z \times \sigma_{\text{demand}} \times \sqrt{\text{Lead Days}} \quad (z=1.65 \text{ for 95\% service level})$$

$$\text{Reorder Point} = (\text{Average Daily Demand} \times \text{Lead Days}) + \text{Safety Stock}$$

$$\text{Recommended Reorder Qty} = \max\left(0, \lceil\text{Reorder Point} - \text{Closing Inventory}\rceil\right)$$

---

## 🖥️ Streamlit Dashboard Pages

1. **📊 Executive Summary:** Top KPI cards, daily demand trend, revenue by category, risk donut chart, and dynamic "What needs attention?" insights.
2. **📈 Demand Intelligence:** Detailed sales trends, promotional lift analysis, category/store breakdowns, actual vs forecast alignment, and demand CSV download.
3. **⚠️ Inventory Risk:** Store $\times$ Category risk heatmap, overall stock-out rate %, days of inventory coverage, and sortable risk tables.
4. **🎯 Manager Action Centre:** Urgent action alerts, HIGH/MEDIUM/LOW priority tables, reorder quantity calculations, and item-specific action explainers.
5. **🔍 Product Detail View:** Deep-dive dropdown selector for any Store $\times$ Product combination with full product profile, demand trends, and compact manager summary notes.
6. **🤖 Model Performance:** Held-out test set metrics tables, actual vs predicted scatter plots, confusion matrices, and ROC curves.
7. **💡 Model Explainability:** Horizontal feature importance charts for demand and stock-out models with manager-friendly non-causal explanations.
8. **📑 Executive Reports & Export:** In-app markdown viewer for all project documentation and download buttons for master datasets, quality logs, and action recommendations.

---

## 🛡️ Limitations & Future Improvements

- **Sparse History Handling:** Products with fewer than 7 days of sales history require hierarchical shrinkage or category-level transfer learning.
- **Supply Chain Lead Time Volatility:** Incorporating stochastic supplier delivery delays into safety stock calculations.

---

## 👥 Team Members

- **Jayatheerthan** (Lead Data Scientist & ML Engineer)
- **Kavin Prakash** (Data Engineer & Dashboard Specialist)
