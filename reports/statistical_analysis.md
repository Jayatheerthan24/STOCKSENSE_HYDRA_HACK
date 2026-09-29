# StockSense - Round 1 Statistical Hypothesis Testing Report

## Executive Summary of Hypotheses

| Analysis | H0 | H1 | Test | Statistic | p-value | Decision | Business Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Analysis 1** | Mean daily demand on promotion days == Mean daily demand on non-promotion days | Mean daily demand on promotion days > Mean daily demand on non-promotion days | Mann-Whitney U Test (Right-tailed) | 1861591.0 | 1.5593e-15 | **Reject H0** | Promotions significantly increase sales demand (p = 1.5593e-15 < 0.05). Average promotional daily sales (17.33 units) are 23.4% higher than non-promotional daily sales (14.05 units). |
| **Analysis 2** | Mean daily demand is equal across all store types | Mean daily demand differs across at least one store type pair | Kruskal-Wallis H-Test | 540.6123 | 4.0507e-118 | **Reject H0** | Mean daily demand differs significantly across store types (p = 4.0507e-118 < 0.05). Observed store type average demand: Supermarket: 13.5 units, Hypermarket: 20.9 units, Express: 10.2 units. |
| **Analysis 3** | Stock-out occurrence is independent of promotion status | Stock-out occurrence is associated with promotion status | Chi-Square Test of Independence (χ²) | 7.931 | 4.8596e-03 | **Reject H0** | Stock-out occurrence is significantly associated with promotion status (p = 4.8596e-03 < 0.05). Stock-out rate during promotions (24.0%) differs significantly from non-promotion periods (19.5%). |

---

## Detailed Test Interpretations

### Analysis 1: Promotion Impact on Demand
- **Hypothesis:** 
  - \(H_0\): Mean daily sales on promotion days == Mean daily sales on non-promotion days.
  - \(H_1\): Mean daily sales on promotion days > Mean daily sales on non-promotion days.
- **Methodology:** Mann-Whitney U test (non-parametric two-sample test) evaluated on daily units sold across promotional vs non-promotional observations.
- **Results:** 1.5593e-15 (Stat = 1861591.0).
- **Decision:** **Reject H0**
- **Business Significance:** Promotions deliver a statistically significant and practical demand lift. Retail operations must align inventory stock with promotional calendars.

### Analysis 2: Demand Variations Across Store Types
- **Hypothesis:** 
  - \(H_0\): Mean daily demand is equal across all store types.
  - \(H_1\): Mean daily demand differs across at least one pair of store types.
- **Methodology:** Kruskal-Wallis H-test comparing daily units sold across store format groups (Hypermarket, Supermarket, Express).
- **Results:** 4.0507e-118 (Stat = 540.6123).
- **Decision:** **Reject H0**
- **Business Significance:** Store capacity and customer traffic dictate significantly different demand distributions. Reorder parameters must be customized per store type.

### Analysis 3: Association Between Stock-outs and Promotions
- **Hypothesis:** 
  - \(H_0\): Stock-out occurrence is independent of promotion status.
  - \(H_1\): Stock-out occurrence is associated with promotion status.
- **Methodology:** Chi-Square Test of Independence (\(\chi^2\)) on a 2x2 contingency table (Stock-out Flag vs Promotion Flag).
- **Results:** 4.8596e-03 (Stat = 7.931).
- **Decision:** **Reject H0**
- **Business Significance:** Promotions significantly exacerbate stock-out risks. Supply chain planning must integrate promotional forecasting directly with inventory replenishment.
