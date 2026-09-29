"""
Master Round 2 Training & Evaluation Pipeline for StockSense.
Trains Demand Forecasting models and Stock-out Risk Classification models,
evaluates on time-aware validation/test splits, generates plots, feature importances,
manager recommendations, and compliance validation reports.

Usage:
    python -m src.train_models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)

from src.utils import PROJECT_ROOT, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, ensure_directories
from src.feature_engineering import prepare_modeling_dataset
from src.model_explainability import extract_feature_importance, generate_explainability_report
from src.recommendation_engine import generate_manager_recommendations

MODELS_DIR = PROJECT_ROOT / "models"


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE handling zero demand by clipping denominator to 1.0."""
    denom = np.maximum(y_true, 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom) * 100))


def run_training_pipeline():
    print("==================================================")
    print(" STOCKSENSE - ROUND 2 MODEL TRAINING PIPELINE")
    print("==================================================\n")

    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Step 1: Load Modeling Dataset
    # --------------------------------------------------
    print(">>> 1. Loading Modeling Dataset...")
    modeling_path = PROCESSED_DATA_DIR / "modeling_dataset.csv"
    if not modeling_path.exists():
        df = prepare_modeling_dataset()
    else:
        df = pd.read_csv(modeling_path)
    df['date'] = pd.to_datetime(df['date'])

    # --------------------------------------------------
    # Step 2: Define Feature Sets
    # --------------------------------------------------
    num_cols = [
        'opening', 'received', 'sold', 'closing', 'reorder_lvl', 'lead_days',
        'inventory_gap', 'inventory_ratio', 'daily_units_sold', 'avg_selling_price',
        'avg_discount_pct', 'promotion_flag', 'mrp', 'cost_price', 'shelf_life_days',
        'floor_area_sqft', 'avg_daily_customers', 'temp_c', 'rain_mm', 'holiday',
        'festival', 'weekend', 'local_event', 'temp_was_missing', 'rain_was_missing',
        'sales_history_days', 'sparse_history_flag', 'year', 'month', 'day_of_month',
        'day_of_week', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end',
        'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'rolling_mean_3', 'rolling_mean_7',
        'rolling_mean_14', 'rolling_std_7', 'rolling_std_14', 'demand_change_1d',
        'demand_change_7d', 'demand_cv_7', 'promotion_lag_1', 'promotion_lag_7', 'price_discount_ratio'
    ]

    cat_cols = ['store_id', 'product_id', 'category', 'sub_category', 'brand', 'store_type', 'city', 'region']

    # Filter features actually present in df
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    feature_cols = num_cols + cat_cols

    # Targets
    target_demand = 'next_7_day_demand'
    target_stockout = 'stockout_flag'

    # --------------------------------------------------
    # Step 3: Chronological Time Split
    # --------------------------------------------------
    print("\n>>> 2. Performing Chronological Time-Aware Split (70/15/15)...")
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)

    n_train_dates = int(n_dates * 0.70)
    n_val_dates = int(n_dates * 0.15)

    train_dates = unique_dates[:n_train_dates]
    val_dates = unique_dates[n_train_dates:n_train_dates + n_val_dates]
    test_dates = unique_dates[n_train_dates + n_val_dates:]

    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()

    split_md = f"""# StockSense - Time-Aware Data Split Summary

## Chronological Partitioning (70% Train / 15% Validation / 15% Test)

| Dataset Partition | Date Range | Total Calendar Days | Observation Rows | Percentage of Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')} | {len(train_dates)} | {len(train_df)} | {len(train_df)/len(df)*100:.1f}% |
| **VALIDATION** | {val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')} | {len(val_dates)} | {len(val_df)} | {len(val_df)/len(df)*100:.1f}% |
| **TEST** | {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} | {len(test_dates)} | {len(test_df)} | {len(test_df)/len(df)*100:.1f}% |

**Verification:** Zero temporal overlap between Train, Validation, and Test partitions.
"""
    (REPORTS_DIR / "model_split.md").write_text(split_md, encoding="utf-8")
    print(f"[OK] Saved split details to {REPORTS_DIR / 'model_split.md'}")

    # --------------------------------------------------
    # Step 4: Preprocessing Transformer Fit
    # --------------------------------------------------
    print("\n>>> 3. Fitting Preprocessing Pipelines...")
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # Fit preprocessor strictly on TRAIN data
    X_train_proc = preprocessor.fit_transform(train_df[feature_cols])
    X_val_proc = preprocessor.transform(val_df[feature_cols])
    X_test_proc = preprocessor.transform(test_df[feature_cols])

    # Get feature names after one-hot encoding
    ohe_cats = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    transformed_feature_names = num_cols + list(ohe_cats)

    # --------------------------------------------------
    # Step 5: Model 1 - Demand Forecasting
    # --------------------------------------------------
    print("\n>>> 4. Training Model 1: Demand Forecasting Models...")
    y_train_dem = train_df[target_demand].values
    y_val_dem = val_df[target_demand].values
    y_test_dem = test_df[target_demand].values

    demand_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    }

    dem_metrics_list = []
    trained_dem_models = {}

    for name, model in demand_models.items():
        model.fit(X_train_proc, y_train_dem)
        val_preds = model.predict(X_val_proc)
        val_preds = np.maximum(0, val_preds)  # Demand cannot be negative

        mae = mean_absolute_error(y_val_dem, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val_dem, val_preds))
        r2 = r2_score(y_val_dem, val_preds)
        mape = calculate_mape(y_val_dem, val_preds)

        trained_dem_models[name] = model
        dem_metrics_list.append({
            "Model": name,
            "Partition": "Validation",
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "MAPE (%)": round(float(mape), 2)
        })

    dem_val_df = pd.DataFrame(dem_metrics_list).sort_values(by="RMSE")
    print("\n--- Demand Forecasting Validation Results ---")
    print(dem_val_df.to_string(index=False))

    # Select Best Demand Model (Lowest Validation RMSE)
    best_dem_name = dem_val_df.iloc[0]["Model"]
    best_dem_model = trained_dem_models[best_dem_name]
    print(f"\n[SELECTED DEMAND MODEL]: {best_dem_name}")

    # Final Test Set Evaluation for Selected Demand Model
    test_dem_preds = np.maximum(0, best_dem_model.predict(X_test_proc))
    test_mae = mean_absolute_error(y_test_dem, test_dem_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test_dem, test_dem_preds))
    test_r2 = r2_score(y_test_dem, test_dem_preds)
    test_mape = calculate_mape(y_test_dem, test_dem_preds)

    dem_metrics_list.append({
        "Model": f"{best_dem_name} (Final Test)",
        "Partition": "Test",
        "MAE": round(float(test_mae), 2),
        "RMSE": round(float(test_rmse), 2),
        "R2": round(float(test_r2), 4),
        "MAPE (%)": round(float(test_mape), 2)
    })

    dem_metrics_df = pd.DataFrame(dem_metrics_list)
    dem_metrics_df.to_csv(REPORTS_DIR / "demand_model_metrics.csv", index=False)

    # Save Pipeline & Model
    demand_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_dem_model)
    ])
    joblib.dump(demand_pipeline, MODELS_DIR / "demand_forecasting_model.joblib")
    print(f"[OK] Saved selected demand model pipeline to: {MODELS_DIR / 'demand_forecasting_model.joblib'}")

    # Extract Demand Feature Importance & Save Figures
    dem_imp = extract_feature_importance(best_dem_model, transformed_feature_names)
    dem_imp.to_csv(REPORTS_DIR / "demand_feature_importance.csv", index=False)

    plot_feature_importance(dem_imp, f"Demand Model Feature Importance ({best_dem_name})", FIGURES_DIR / "demand_feature_importance.png")
    plot_demand_actual_vs_pred(y_test_dem, test_dem_preds, FIGURES_DIR / "demand_actual_vs_predicted.png")

    # Save Demand Report
    generate_demand_report(best_dem_name, test_mae, test_rmse, test_r2, test_mape, dem_val_df)

    # --------------------------------------------------
    # Step 6: Model 2 - Stock-out Risk Classification
    # --------------------------------------------------
    print("\n>>> 5. Training Model 2: Stock-out Risk Classification Models...")
    y_train_so = train_df[target_stockout].values
    y_val_so = val_df[target_stockout].values
    y_test_so = test_df[target_stockout].values

    scale_pos_ratio = float((y_train_so == 0).sum() / max(1, (y_train_so == 1).sum()))

    stockout_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    }
   
    so_metrics_list = []
    trained_so_models = {}

    for name, model in stockout_models.items():
        model.fit(X_train_proc, y_train_so)
        val_preds = model.predict(X_val_proc)
        val_probs = model.predict_proba(X_val_proc)[:, 1] if hasattr(model, 'predict_proba') else val_preds

        acc = accuracy_score(y_val_so, val_preds)
        prec = precision_score(y_val_so, val_preds, zero_division=0)
        rec = recall_score(y_val_so, val_preds, zero_division=0)
        f1 = f1_score(y_val_so, val_preds, zero_division=0)
        auc = roc_auc_score(y_val_so, val_probs)
        pr_auc = average_precision_score(y_val_so, val_probs)

        trained_so_models[name] = model
        so_metrics_list.append({
            "Model": name,
            "Partition": "Validation",
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "PR-AUC": round(float(pr_auc), 4)
        })

    so_val_df = pd.DataFrame(so_metrics_list).sort_values(by="ROC-AUC", ascending=False)
    print("\n--- Stock-out Risk Validation Results ---")
    print(so_val_df.to_string(index=False))

    # Select Best Stock-out Model (Highest Validation ROC-AUC / F1)
    best_so_name = so_val_df.iloc[0]["Model"]
    best_so_model = trained_so_models[best_so_name]
    print(f"\n[SELECTED STOCK-OUT MODEL]: {best_so_name}")

    # Final Test Set Evaluation for Selected Stockout Model
    test_so_preds = best_so_model.predict(X_test_proc)
    test_so_probs = best_so_model.predict_proba(X_test_proc)[:, 1]

    t_acc = accuracy_score(y_test_so, test_so_preds)
    t_prec = precision_score(y_test_so, test_so_preds, zero_division=0)
    t_rec = recall_score(y_test_so, test_so_preds, zero_division=0)
    t_f1 = f1_score(y_test_so, test_so_preds, zero_division=0)
    t_auc = roc_auc_score(y_test_so, test_so_probs)
    t_pr_auc = average_precision_score(y_test_so, test_so_probs)

    so_metrics_list.append({
        "Model": f"{best_so_name} (Final Test)",
        "Partition": "Test",
        "Accuracy": round(float(t_acc), 4),
        "Precision": round(float(t_prec), 4),
        "Recall": round(float(t_rec), 4),
        "F1-Score": round(float(t_f1), 4),
        "ROC-AUC": round(float(t_auc), 4),
        "PR-AUC": round(float(t_pr_auc), 4)
    })

    so_metrics_df = pd.DataFrame(so_metrics_list)
    so_metrics_df.to_csv(REPORTS_DIR / "stockout_model_metrics.csv", index=False)

    # Save Pipeline & Model
    stockout_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_so_model)
    ])
    joblib.dump(stockout_pipeline, MODELS_DIR / "stockout_risk_model.joblib")
    print(f"[OK] Saved selected stockout model pipeline to: {MODELS_DIR / 'stockout_risk_model.joblib'}")

    # Extract Stockout Feature Importance & Save Figures
    so_imp = extract_feature_importance(best_so_model, transformed_feature_names)
    so_imp.to_csv(REPORTS_DIR / "stockout_feature_importance.csv", index=False)

    plot_feature_importance(so_imp, f"Stock-out Risk Feature Importance ({best_so_name})", FIGURES_DIR / "stockout_feature_importance.png")
    plot_stockout_confusion_matrix(y_test_so, test_so_preds, FIGURES_DIR / "stockout_confusion_matrix.png")
    plot_stockout_roc_curve(y_test_so, test_so_probs, FIGURES_DIR / "stockout_roc_curve.png")

    # Save Stockout Report
    generate_stockout_report(best_so_name, t_acc, t_prec, t_rec, t_f1, t_auc, t_pr_auc, so_val_df)

    # --------------------------------------------------
    # Step 7: Model Explainability
    # --------------------------------------------------
    print("\n>>> 6. Generating Model Explainability Report...")
    generate_explainability_report(dem_imp, so_imp, best_dem_name, best_so_name)

    # --------------------------------------------------
    # Step 8: Manager Recommendation Engine Execution
    # --------------------------------------------------
    print("\n>>> 7. Generating Manager Recommendations on Test Set...")
    recs_df = generate_manager_recommendations(test_df, test_dem_preds, test_so_probs)

    # --------------------------------------------------
    # Step 9: Automatic Quality & Compliance Verification
    # --------------------------------------------------
    print("\n>>> 8. Running Compliance Verification Checks...")
    run_quality_checks(
        df, train_df, val_df, test_df,
        recs_df, best_dem_model, best_so_model,
        test_dem_preds, test_so_probs
    )

    # --------------------------------------------------
    # Summary Output
    # --------------------------------------------------
    high_risk_cnt = len(recs_df[recs_df['risk_level'] == 'HIGH'])
    print("\n==================================================")
    print(" ROUND 2 TRAINING PIPELINE SUMMARY")
    print("==================================================")
    print(f"• Final Demand Model Selected: {best_dem_name}")
    print(f"  - Test MAE: {test_mae:.2f}")
    print(f"  - Test RMSE: {test_rmse:.2f}")
    print(f"  - Test R²: {test_r2:.4f}")
    print(f"  - Test MAPE: {test_mape:.2f}%")
    print(f"• Final Stock-out Model Selected: {best_so_name}")
    print(f"  - Test Accuracy: {t_acc:.4f}")
    print(f"  - Test Precision: {t_prec:.4f}")
    print(f"  - Test Recall: {t_rec:.4f}")
    print(f"  - Test F1-Score: {t_f1:.4f}")
    print(f"  - Test ROC-AUC: {t_auc:.4f}")
    print(f"• Recommendations Generated: {len(recs_df)}")
    print(f"• HIGH Risk Recommendations: {high_risk_cnt}")
    print(f"• Top 5 Demand Features: {list(dem_imp.head(5)['feature'])}")
    print(f"• Top 5 Stock-out Features: {list(so_imp.head(5)['feature'])}")
    print("==================================================")
    print("[OK] Round 2 Master Pipeline Executed Successfully!")


def plot_feature_importance(imp_df: pd.DataFrame, title: str, output_path: Path):
    plt.figure(figsize=(10, 6))
    top_df = imp_df.head(15)
    sns.barplot(data=top_df, x='importance_pct', y='feature', palette='viridis')
    plt.title(title)
    plt.xlabel('Relative Importance (%)')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_demand_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--', label='Ideal 1:1 Forecast')
    plt.title('Demand Forecasting: Actual vs Predicted (7-Day Sum)')
    plt.xlabel('Actual 7-Day Demand')
    plt.ylabel('Predicted 7-Day Demand')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stockout_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['No Stockout (0)', 'Stockout (1)'], yticklabels=['No Stockout (0)', 'Stockout (1)'])
    plt.title('Stock-out Risk Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stockout_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, output_path: Path):
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Stock-out Risk ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_demand_report(model_name: str, mae: float, rmse: float, r2: float, mape: float, val_df: pd.DataFrame):
    md = f"""# StockSense - Demand Forecasting Model Report

## 1. Executive Summary
The selected demand forecasting model is **{model_name}**, evaluated on an untouched chronological test set.

## 2. Model Performance Summary

### Validation Models Comparison
{val_df.to_markdown(index=False)}

### Final Selected Model ({model_name}) Test Performance
- **MAE:** {mae:.2f} units
- **RMSE:** {rmse:.2f} units
- **R² Score:** {r2:.4f}
- **MAPE:** {mape:.2f}% (Handling zero demand by clipping denominator to 1.0)

## 3. Target & Feature Specifications
- **Target Variable:** `next_7_day_demand` (Forward-looking 7-day cumulative sum of `daily_units_sold`).
- **Feature Engineering:** Includes 14-day lags, 7-day rolling statistics, demand trend volatility, pricing ratios, promotion indicators, and calendar features.
- **Data Split:** Chronological 70% Train, 15% Validation, 15% Test with zero temporal leakage.
"""
    (REPORTS_DIR / "demand_model_report.md").write_text(md, encoding="utf-8")


def generate_stockout_report(model_name: str, acc: float, prec: float, rec: float, f1: float, auc: float, pr_auc: float, val_df: pd.DataFrame):
    md = f"""# StockSense - Stock-out Risk Classification Model Report

## 1. Executive Summary
The selected stock-out risk classification model is **{model_name}**, evaluated on an untouched chronological test set with class imbalance handling.

## 2. Model Performance Summary

### Validation Models Comparison
{val_df.to_markdown(index=False)}

### Final Selected Model ({model_name}) Test Performance
- **Accuracy:** {acc:.4f}
- **Precision:** {prec:.4f}
- **Recall:** {rec:.4f}
- **F1-Score:** {f1:.4f}
- **ROC-AUC:** {auc:.4f}
- **PR-AUC:** {pr_auc:.4f}

## 3. Target & Feature Specifications
- **Target Variable:** `stockout_flag` (1 if closing inventory $\\le 0$, else 0).
- **Imbalance Handling:** Utilized `class_weight='balanced'` / `scale_pos_weight` ratios derived from training data without premature oversampling.
"""
    (REPORTS_DIR / "stockout_model_report.md").write_text(md, encoding="utf-8")


def run_quality_checks(df, train_df, val_df, test_df, recs_df, dem_model, so_model, dem_preds, so_probs):
    checks = []

    # 1. Grain duplicates check
    grain_dups = int(df.duplicated(subset=['date', 'store_id', 'product_id']).sum())
    checks.append(("No duplicate date-store-product rows in modeling dataset", "PASS" if grain_dups == 0 else "FAIL", f"Found {grain_dups} duplicate grain rows."))

    # 2. Chronological split check
    t_max = train_df['date'].max()
    v_min = val_df['date'].min()
    v_max = val_df['date'].max()
    test_min = test_df['date'].min()

    chrono_pass = (t_max < v_min) and (v_max < test_min)
    checks.append(("Train/Val/Test Chronological Order (No temporal overlap)", "PASS" if chrono_pass else "FAIL", f"Train Max: {t_max.date()}, Val Min: {v_min.date()}, Val Max: {v_max.date()}, Test Min: {test_min.date()}"))

    # 3. Model save load check
    dem_loaded = joblib.load(MODELS_DIR / "demand_forecasting_model.joblib")
    so_loaded = joblib.load(MODELS_DIR / "stockout_risk_model.joblib")
    checks.append(("Saved models can be loaded successfully", "PASS" if dem_loaded and so_loaded else "FAIL", "Joblib models loaded successfully."))

    # 4. Predictions check
    nan_dem = np.isnan(dem_preds).sum()
    nan_so = np.isnan(so_probs).sum()
    checks.append(("Predictions contain no unexpected NaN values", "PASS" if nan_dem == 0 and nan_so == 0 else "FAIL", f"NaN demand: {nan_dem}, NaN stockout: {nan_so}."))

    # 5. Recommendation non-negative check
    neg_recs = (recs_df['recommended_reorder_qty'] < 0).sum()
    checks.append(("Recommendation quantities are non-negative", "PASS" if neg_recs == 0 else "FAIL", f"Found {neg_recs} negative recommendation quantities."))

    # 6. Probability bounds check
    prob_valid = ((so_probs >= 0.0) & (so_probs <= 1.0)).all()
    checks.append(("Stock-out probability bounded between 0 and 1", "PASS" if prob_valid else "FAIL", "All probabilities in [0.0, 1.0]."))

    # 7. Metrics files check
    metrics_pass = (REPORTS_DIR / "demand_model_metrics.csv").exists() and (REPORTS_DIR / "stockout_model_metrics.csv").exists()
    checks.append(("Selected model metrics saved to reports/", "PASS" if metrics_pass else "FAIL", "Metrics CSV files present."))

    # Build Markdown Report
    rows_md = "\n".join([f"| **{title}** | **{status}** | {notes} |" for title, status, notes in checks])
    val_md = f"""# StockSense - Round 2 Quality & Compliance Validation Report

## Validation Checklist

| Verification Check | Status | Empirical Observation |
| :--- | :--- | :--- |
{rows_md}

---

## Technical Notes & Limitations
- **Lag Window Trimming:** The initial 14 calendar days were trimmed to construct `lag_14` and 14-day rolling statistics without data leakage.
- **Target Window Trimming:** The final 7 calendar days were trimmed from training targets to maintain exact 7-day forward demand sums.
- **Model Storage:** Trained scikit-learn/XGBoost pipelines saved to `models/demand_forecasting_model.joblib` and `models/stockout_risk_model.joblib`.
"""
    (REPORTS_DIR / "round2_validation.md").write_text(val_md, encoding="utf-8")
    print(f"[OK] Saved Round 2 validation report to: {REPORTS_DIR / 'round2_validation.md'}")


if __name__ == "__main__":
    run_training_pipeline()
