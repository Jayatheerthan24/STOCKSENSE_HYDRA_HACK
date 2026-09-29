# StockSense - Round 1 Business EDA Insights Report

## Business KPI Overview
- **Total Revenue:** ₹4,886,509.00
- **Total Units Sold:** 72,126
- **Stock-out Rate:** 20.16% (1000 observations)
- **Average Days of Inventory:** 83.26 days
- **Inventory Turnover Ratio:** 177.77x
- **Promotion Lift:** 23.38% increase in daily demand

---

## Major Business Findings

### 1. Revenue & Units Sold by Product Category
- **Business Question:** Which product categories generate the highest revenue and volume demand?
- **Finding:** Categories exhibit distinct sales velocity and margin characteristics. High-volume categories like Beverages and Snacks drive top-line volume, while Household and Personal Care contribute high margin per unit.
- **Evidence:** Refer to figures [`01_revenue_by_category.png`](figures/01_revenue_by_category.png) and [`02_units_sold_by_category.png`](figures/02_units_sold_by_category.png).
- **Business Implication:** Inventory allocation must prioritize high-velocity fast-moving categories to avoid lost volume, while preserving buffer stock for high-margin categories.

### 2. Store Level Performance Disparities
- **Business Question:** Do store formats (Hypermarket vs Supermarket vs Express) show significantly different demand patterns?
- **Finding:** Larger store formats (Hypermarkets) demonstrate higher absolute daily revenue and sales volume, driven by larger floor space and higher average daily footfall.
- **Evidence:** Refer to figure [`03_revenue_by_store.png`](figures/03_revenue_by_store.png).
- **Business Implication:** Replenishment frequencies should be tailored by store format rather than applying uniform reorder levels.

### 3. Promotion Lift Analysis
- **Business Question:** What is the quantified lift in demand during promotional periods?
- **Finding:** Promotional flags correlate with an average demand increase of 23.38%.
- **Evidence:** Refer to figure [`05_promotion_vs_non_promotion.png`](figures/05_promotion_vs_non_promotion.png).
- **Business Implication:** Marketing promotions generate substantial sales lift, but demand forecasting models must account for promo schedules to prevent severe stock-outs during promotional bursts.

### 4. Stock-out Risk Analysis
- **Business Question:** How frequent are stock-outs and which stores/categories are most affected?
- **Finding:** The overall stock-out rate is 20.16%. Stock-out frequency increases significantly during promotional campaigns due to inadequate safety stock buffers.
- **Evidence:** Refer to figures [`08_stockout_frequency_by_store.png`](figures/08_stockout_frequency_by_store.png) and [`09_stockout_frequency_by_category.png`](figures/09_stockout_frequency_by_category.png).
- **Business Implication:** Safety stock levels and reorder triggers (`reorder_lvl`) must be dynamically adjusted upwards ahead of promotional launches.

### 5. Product Demand Volatility
- **Business Question:** Which products exhibit high demand variance requiring higher safety stock?
- **Finding:** Perishable items and promotional items exhibit high Coefficient of Variation (CV > 1.0), whereas steady staples demonstrate stable demand patterns.
- **Evidence:** Refer to figure [`07_product_demand_volatility.png`](figures/07_product_demand_volatility.png).
- **Business Implication:** High CV products require agile, short lead-time replenishment strategies.
