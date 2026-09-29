# StockSense - Round 1 Data Quality Audit & Cleaning Report

## 1. Executive Summary
This document presents the data quality audit and cleaning methodology for the IntelliData 2026 StockSense Challenge (Round 1). The raw input data consists of five relational CSV datasets located in `data/raw/`: `transactions.csv`, `products.csv`, `stores.csv`, `inventory.csv`, and `external_factors.csv`.

All dataset cleaning routines follow strict, reproducible operations implemented in `src/data_cleaning.py`. **No raw files were altered or overwritten.**

## 2. Dataset Inventory

| Dataset | File Path | Grain / Entity | Key Columns |
| :--- | :--- | :--- | :--- |
| **Transactions** | `data/raw/transactions.csv` | Transaction Line Item | `transaction_id`, `date`, `store_id`, `product_id`, `quantity`, `selling_price`, `discount_pct`, `promotion_flag` |
| **Products** | `data/raw/products.csv` | Product Catalog Item | `product_id`, `category`, `sub_category`, `brand`, `mrp`, `cost_price`, `shelf_life`, `supplier_id` |
| **Stores** | `data/raw/stores.csv` | Store Profile | `store_id`, `city`, `store_type`, `floor_area_sqft`, `avg_daily_customers`, `region` |
| **Inventory** | `data/raw/inventory.csv` | Daily Store x Product Stock | `date`, `store`, `product`, `opening`, `received`, `sold`, `closing`, `reorder_lvl`, `lead_days` |
| **External Factors** | `data/raw/external_factors.csv` | Daily City Environment | `date`, `city`, `temp_c`, `rain_mm`, `holiday`, `festival`, `weekend`, `local_event` |

## 3. Data Quality Audit & Cleaning Treatment

### 3.1 Transactions Quality (`transactions.csv`)
- **Exact Duplicate Rows:** Removed exact duplicate duplicate copies to prevent double counting.
- **Conflicting Transaction IDs:** Identical `transaction_id` entries with conflicting line item attributes were flagged and excluded from valid positive sales aggregation.
- **Negative & Zero Quantities:** Negative quantity entries (representing returns/voids) were flagged as invalid and excluded from positive sales aggregation without silent conversion to positive numbers.
- **Invalid Pricing & Discounts:** Verified positive `selling_price` and clipped `discount_pct` to bounds [0, 100].

### 3.2 Products Quality (`products.csv`)
- **Category Standardization:** Inconsistent category casings (e.g. `beverages`, `BEVERAGES`, `Beverages`) were standardized to Title Case (`Beverages`).
- **Column Renaming:** Column `shelf_life` was standardized to `shelf_life_days`.
- **Pricing Sanity:** Flagged items where `cost_price > mrp` or non-positive pricing existed.

### 3.3 Stores Quality (`stores.csv`)
- **Store Mapping:** Validated that each `store_id` uniquely maps to a single store profile (city, type, area, customer count).

### 3.4 Inventory Quality (`inventory.csv`)
- **Key Standardisation:** Renamed `store` -> `store_id` and `product` -> `product_id`.
- **Reconciliation Audit:** Calculated arithmetic balance equation: `closing = opening + received - sold`.
- **Mismatch Flagging:** Created `inventory_mismatch_flag` (1 for balance mismatch, 0 for balanced). Discrepancies were **not** silently overwritten.

### 3.5 External Factors Quality (`external_factors.csv`)
- **Weather Imputation:** Missing `temp_c` and `rain_mm` values were imputed using city-level medians.
- **Missingness Flags:** Preserved explicit indicators `temp_was_missing` and `rain_was_missing`.

### 3.6 Sparse History Analysis
- Calculated total active sales days per product.
- Products with fewer than 7 days of observed history were flagged with `sparse_history_flag = 1`.
- Sparse products are retained in the master dataset to preserve complete catalog coverage.

## 4. Master Dataset Integration & Grain Validation
- **Required Grain:** `ONE ROW = ONE DATE x ONE STORE_ID x ONE PRODUCT_ID`
- Integrated daily aggregated transactions with product master, store master, inventory, and external factors.
- **Grain Validation:** Verified ZERO duplicate entries for `(date, store_id, product_id)` combination.

## 5. Output Data Files
- `data/processed/master_dataset.csv`
- `data/processed/data_quality_summary.csv`
