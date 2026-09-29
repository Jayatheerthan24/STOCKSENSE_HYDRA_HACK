# StockSense - Stock-out Risk Classification Model Report

## 1. Executive Summary
The selected stock-out risk classification model is **Decision Tree**, evaluated on an untouched chronological test set with class imbalance handling.

## 2. Model Performance Summary

### Validation Models Comparison
| Model                    | Partition   |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC |
|:-------------------------|:------------|-----------:|------------:|---------:|-----------:|----------:|---------:|
| Decision Tree            | Validation  |     1      |       1     |        1 |     1      |    1      |   1      |
| Random Forest Classifier | Validation  |     1      |       1     |        1 |     1      |    1      |   1      |
| Logistic Regression      | Validation  |     0.9688 |       0.879 |        1 |     0.9356 |    0.9946 |   0.9802 |

### Final Selected Model (Decision Tree) Test Performance
- **Accuracy:** 1.0000
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **ROC-AUC:** 1.0000
- **PR-AUC:** 1.0000

## 3. Target & Feature Specifications
- **Target Variable:** `stockout_flag` (1 if closing inventory $\le 0$, else 0).
- **Imbalance Handling:** Utilized `class_weight='balanced'` / `scale_pos_weight` ratios derived from training data without premature oversampling.
