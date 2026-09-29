# StockSense - Round 1 Business EDA Insights Report

## Executive Summary & Business Context
This report synthesizes exploratory data analysis insights from the Master Analytics Dataset (`data/processed/master_dataset.csv`). Analysis covers sales performance, category dynamics, store format variations, promotional lift, and stock-out frequency.

All figures referenced below are stored under `reports/figures/`.

---

## Key Business Findings

### Finding 1: Category Demand & Margin Profile
- **Business Question:** Which product categories drive sales volume vs revenue margin?
- **Finding:** Fast-Moving Consumer Goods (Beverages, Snacks) generate the highest overall transaction volume, while Household and Personal Care categories deliver higher average revenue per unit.
- **Evidence:** Refer to [`01_revenue_by_category.png`](figures/01_revenue_by_category.png) and [`02_units_sold_by_category.png`](figures/02_units_sold_by_category.png).
- **Business Implication:** Inventory stocking priorities should balance high-turnover volume drivers against high-margin revenue generators.

### Finding 2: Store Format Revenue Contribution
- **Business Question:** How does daily revenue vary across store formats and locations?
- **Finding:** Hypermarket locations (e.g. S02 Chennai) achieve significantly higher daily total revenue and units sold compared to Express stores (e.g. S03 Madurai).
- **Evidence:** Refer to [`03_revenue_by_store.png`](figures/03_revenue_by_store.png) and [`10_store_category_heatmap.png`](figures/10_store_category_heatmap.png).
- **Business Implication:** Reorder point formulas (`reorder_lvl`) and lead times must be tuned specifically to store format capacity.

### Finding 3: Promotion Demand Lift
- **Business Question:** What is the quantitative sales lift associated with promotional events?
- **Finding:** Promotional active periods demonstrate a clear increase in average daily units sold compared to non-promotional days.
- **Evidence:** Refer to [`05_promotion_vs_non_promotion.png`](figures/05_promotion_vs_non_promotion.png).
- **Business Implication:** Promotions effectively stimulate demand volume, but require proactive inventory replenishment to avoid stock-outs.

### Finding 4: Stock-out Vulnerability
- **Business Question:** Where are stock-out events concentrated?
- **Finding:** Stock-out occurrences (`closing == 0`) are concentrated in fast-moving beverage items and during promotional surges.
- **Evidence:** Refer to [`08_stockout_frequency_by_store.png`](figures/08_stockout_frequency_by_store.png) and [`09_stockout_frequency_by_category.png`](figures/09_stockout_frequency_by_category.png).
- **Business Implication:** Safety stock buffers should be adjusted dynamically ahead of scheduled promotional campaigns.

### Finding 5: Product Volatility & Demand Uncertainty
- **Business Question:** Which products exhibit high demand volatility?
- **Finding:** Products with high Coefficient of Variation (CV) require higher safety stock allocations.
- **Evidence:** Refer to [`07_product_demand_volatility.png`](figures/07_product_demand_volatility.png).
- **Business Implication:** Dynamic inventory policies should classify items into predictable vs high-volatility replenishment tiers.
