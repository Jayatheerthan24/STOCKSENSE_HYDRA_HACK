# StockSense - Round 1 Data Quality Audit & Cleaning Report

## 1. Executive Summary
This document provides a comprehensive audit of the raw data files (`transactions.csv`, `products.csv`, `stores.csv`, `inventory.csv`, `external_factors.csv`) for the StockSense challenge. All dataset cleaning operations adhere strictly to documented treatment rules. No raw files were modified, and no silent repairs or dummy value fabrications were performed.

## 2. Dataset Inventory

| Dataset | Raw Rows | Raw Columns | Date Range | Unique IDs | Duplicate Rows | Missing Values |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Transactions** | 10557 | 11 | 2026-07-01 to 2026-08-31 | 10553 | 4 | 6 |
| **Products** | 20 | 8 | N/A | 20 | 0 | 1 |
| **Stores** | 4 | 6 | N/A | 4 | 0 | 0 |
| **Inventory** | 4960 | 9 | 2026-07-01 to 2026-08-31 | N/A | 0 | 3 |
| **External Factors** | 248 | 8 | 2026-07-01 to 2026-08-31 | N/A | 0 | 6 |

## 3. Data Quality Issues & Summary Table

The audit identified data quality anomalies across all five datasets. The consolidated quality summary is recorded in `data/processed/data_quality_summary.csv`.

| dataset                 | issue                                                                        |   affected_rows |   affected_percentage | treatment                                                          | notes                                                                                            |
|:------------------------|:-----------------------------------------------------------------------------|----------------:|----------------------:|:-------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| transactions            | Exact duplicate rows                                                         |               4 |                  0.04 | Removed duplicate copy                                             | Exact match across all columns removed to prevent duplicate counting.                            |
| transactions            | Negative quantity values                                                     |               2 |                  0.02 | Flagged as invalid sales; excluded from positive sales aggregation | Negative quantities represent invalid/return records; preserved in log, excluded from sales sum. |
| transactions            | Missing or negative selling_price                                            |               3 |                  0.03 | Filtered out invalid pricing records                               | Valid positive selling price required for revenue calculation.                                   |
| products                | Inconsistent category name casing (e.g. beverages, BEVERAGES)                |               3 |                 15    | Standardized category names to Title Case                          | Unified category text representation.                                                            |
| products                | Missing brand                                                                |               1 |                  5    | Imputed as 'Unknown' / Preserved as NaN                            | 1 products missing brand.                                                                        |
| inventory               | Missing inventory stock values                                               |               3 |                  0.06 | Imputed missing with 0 and flagged                                 | Missing opening/received/sold/closing values.                                                    |
| inventory               | Inventory balance arithmetic mismatch (closing != opening + received - sold) |             992 |                 20    | Created inventory_mismatch_flag = 1 without silent repair          | 992 records (20.0%) have inventory balance discrepancies.                                        |
| external_factors        | Missing temperature (temp_c)                                                 |               4 |                  1.61 | City-level median imputation + preserved temp_was_missing flag     | Imputed missing weather temperature values.                                                      |
| external_factors        | Missing rainfall (rain_mm)                                                   |               2 |                  0.81 | City-level median imputation + preserved rain_was_missing flag     | Imputed missing rainfall data.                                                                   |
| transactions / products | Sparse sales history (< 7 observed sales days)                               |               1 |                  5    | Created sparse_history_flag = 1; retained products in catalog      | Identified 1 products with limited historical demand.                                            |

## 4. Specific Quality Audit Analyses

### 4.1 Duplicate Analysis
- **Exact Duplicate Rows:** Removed exact duplicate copies to prevent double-counting.
- **Conflicting Transaction IDs:** Flagged duplicate transaction IDs appearing with differing transaction details to prevent corrupted sales totals.

### 4.2 Missing & Invalid Values
- **Transactions:** Invalid or negative quantity records were flagged as non-sales and excluded from positive demand aggregation.
- **Products:** Product category names (e.g. `beverages`, `BEVERAGES`, `Beverages`) were standardized to Title Case (`Beverages`). Column `shelf_life` was standardized to `shelf_life_days`.
- **External Factors:** Missing temperature and rainfall values were imputed using city-level medians, preserving binary flags (`temp_was_missing`, `rain_was_missing`).

### 4.3 Inventory Reconciliation
- Calculated formula: `closing = opening + received - sold`.
- Records breaking this arithmetic balance were flagged with `inventory_mismatch_flag = 1`. Balance discrepancies were **not** silently overwritten.

### 4.4 Sparse History
- Products with fewer than 7 days of observed sales history were identified and flagged with `sparse_history_flag = 1`. Sparse products are retained in the master dataset for catalog completeness.

## 5. Master Dataset Grain Verification
- **Master Grain:** `ONE ROW = ONE DATE x ONE STORE_ID x ONE PRODUCT_ID`
- **Total Master Rows:** 4960
- **Total Master Columns:** 39
- **Duplicate Grain Count:** 0 (Verified ZERO duplicates)

## 6. Remaining Limitations
- Counterfactual lost sales during stock-outs require sophisticated demand estimation in Round 2/3.
- Sparse products require hierarchical shrinkage or category-level pooling during forecasting.
