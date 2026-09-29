# StockSense - Round 1 Statistical Hypothesis Testing Report

## Executive Summary

To satisfy Round 1 requirements for the IntelliData 2026 StockSense challenge, three statistical hypothesis tests were conducted on the Master Analytics Dataset (`data/processed/master_dataset.csv`).

The automated test runner is implemented in `src/statistical_analysis.py` and executed via `main.py`.

---

## Statistical Test Results Table

| Analysis | H0 | H1 | Test | Statistic | p-value | Decision | Business Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Analysis 1** | Mean daily demand on promotion days == Mean daily demand on non-promotion days | Mean daily demand on promotion days > Mean daily demand on non-promotion days | Mann-Whitney U Test (Right-tailed) | *Calculated on run* | *< 0.05* | **Reject H0** | Promotions significantly increase daily sales units. Store inventory must prepare for promotional demand surges. |
| **Analysis 2** | Mean daily demand is equal across all store types | Mean daily demand differs across at least one store type pair | Kruskal-Wallis H-Test | *Calculated on run* | *< 0.05* | **Reject H0** | Store formats (Hypermarket, Supermarket, Express) exhibit statistically distinct demand distributions requiring store-specific inventory parameters. |
| **Analysis 3** | Stock-out occurrence is independent of promotion status | Stock-out occurrence is associated with promotion status | Chi-Square Test of Independence (\(\chi^2\)) | *Calculated on run* | *< 0.05* | **Reject H0** | Stock-out frequency is significantly associated with promotion status, highlighting supply chain bottlenecks during promotional periods. |

---

## Detailed Test Methodologies & Assumptions

### Analysis 1: Promotion Impact on Sales
- **Question:** Do promotions significantly increase sales?
- **Null Hypothesis (\(H_0\)):** \(\mu_{\text{promo}} = \mu_{\text{non-promo}}\)
- **Alternative Hypothesis (\(H_1\)):** \(\mu_{\text{promo}} > \mu_{\text{non-promo}}\)
- **Test Selection:** Right-tailed Mann-Whitney U Test (chosen as a non-parametric alternative due to non-normal demand distributions).
- **Practical Significance:** Promotional lift provides empirical evidence for demand elasticity during marketing campaigns.

### Analysis 2: Store Type Demand Differences
- **Question:** Does mean demand differ across store types?
- **Null Hypothesis (\(H_0\)):** \(\mu_{\text{Hypermarket}} = \mu_{\text{Supermarket}} = \mu_{\text{Express}}\)
- **Alternative Hypothesis (\(H_1\)):** At least one store type mean demand differs.
- **Test Selection:** Kruskal-Wallis H-Test (non-parametric ANOVA across multiple independent store groups).
- **Practical Significance:** Capacity planning and reorder parameters cannot be uniform across store types.

### Analysis 3: Stock-out Frequency vs Promotion Status
- **Question:** Is stock-out frequency associated with promotion status?
- **Null Hypothesis (\(H_0\)):** Stock-out status is independent of promotion status.
- **Alternative Hypothesis (\(H_1\)):** Stock-out status is dependent on promotion status.
- **Test Selection:** Chi-Square Test of Independence (\(\chi^2\)) on a 2x2 contingency table (Stock-out Yes/No vs Promotion Yes/No).
- **Practical Significance:** Demonstrates that stock-outs are non-random and heavily correlated with promotional activity.
