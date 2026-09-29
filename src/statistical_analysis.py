"""
Statistical hypothesis testing module for StockSense Round 1 Implementation.
Executes 3 required statistical tests and outputs structured results table and interpretations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any


def run_statistical_tests(master_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Execute 3 required statistical hypothesis tests:
    1. Promotion Impact on Sales (T-Test / Mann-Whitney U Test)
    2. Store Type Demand Differences (ANOVA / Kruskal-Wallis Test)
    3. Stock-out Frequency vs Promotion Status (Chi-Square Test of Independence)
    
    Returns:
        List[Dict[str, Any]]: List of structured statistical test result dictionaries.
    """
    df = master_df.copy()
    results = []

    # =========================================================================
    # Analysis 1: Do promotions significantly increase sales?
    # =========================================================================
    if 'promotion_flag' in df.columns and 'daily_units_sold' in df.columns:
        promo_sales = df[df['promotion_flag'] == 1]['daily_units_sold'].dropna()
        non_promo_sales = df[df['promotion_flag'] == 0]['daily_units_sold'].dropna()

        # Check normality assumption
        # Use Mann-Whitney U test as non-parametric robust alternative if skewed
        stat_val, p_val = stats.mannwhitneyu(promo_sales, non_promo_sales, alternative='greater')
        
        mean_promo = promo_sales.mean()
        mean_non_promo = non_promo_sales.mean()
        lift_pct = ((mean_promo - mean_non_promo) / mean_non_promo * 100) if mean_non_promo > 0 else 0

        decision = "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
        interp = (
            f"Promotions significantly increase sales demand (p = {p_val:.4e} < 0.05). "
            f"Average promotional daily sales ({mean_promo:.2f} units) are {lift_pct:.1f}% higher "
            f"than non-promotional daily sales ({mean_non_promo:.2f} units)."
            if p_val < 0.05 else
            f"No statistically significant difference in daily sales between promotion and non-promotion days (p = {p_val:.4f} >= 0.05)."
        )

        results.append({
            "analysis_id": "Analysis 1",
            "question": "Do promotions significantly increase sales?",
            "h0": "Mean daily demand on promotion days == Mean daily demand on non-promotion days",
            "h1": "Mean daily demand on promotion days > Mean daily demand on non-promotion days",
            "test_name": "Mann-Whitney U Test (Right-tailed)",
            "sample_def": f"N_promo={len(promo_sales)}, N_non_promo={len(non_promo_sales)}",
            "statistic": round(float(stat_val), 4),
            "p_value": round(float(p_val), 6),
            "p_value_fmt": f"{p_val:.4e}",
            "decision": decision,
            "business_interpretation": interp
        })

    # =========================================================================
    # Analysis 2: Does mean demand differ across store types?
    # =========================================================================
    if 'store_type' in df.columns and 'daily_units_sold' in df.columns:
        store_types = df['store_type'].dropna().unique()
        groups = [df[df['store_type'] == st]['daily_units_sold'].dropna() for st in store_types]
        
        # Execute Kruskal-Wallis H-test for non-parametric ANOVA comparison
        stat_val, p_val = stats.kruskal(*groups)

        group_means = {st: df[df['store_type'] == st]['daily_units_sold'].mean() for st in store_types}
        means_str = ", ".join([f"{st}: {m:.1f} units" for st, m in group_means.items()])

        decision = "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
        interp = (
            f"Mean daily demand differs significantly across store types (p = {p_val:.4e} < 0.05). "
            f"Observed store type average demand: {means_str}."
            if p_val < 0.05 else
            f"No statistically significant difference in mean daily demand across store types (p = {p_val:.4f} >= 0.05)."
        )

        results.append({
            "analysis_id": "Analysis 2",
            "question": "Does mean demand differ across store types?",
            "h0": "Mean daily demand is equal across all store types",
            "h1": "Mean daily demand differs across at least one store type pair",
            "test_name": "Kruskal-Wallis H-Test",
            "sample_def": f"{len(store_types)} store types across N={len(df)} total observations",
            "statistic": round(float(stat_val), 4),
            "p_value": round(float(p_val), 6),
            "p_value_fmt": f"{p_val:.4e}",
            "decision": decision,
            "business_interpretation": interp
        })

    # =========================================================================
    # Analysis 3: Is stock-out frequency associated with promotion status?
    # =========================================================================
    if 'closing' in df.columns and 'promotion_flag' in df.columns:
        df['is_stockout'] = (df['closing'] == 0).astype(int)
        contingency_table = pd.crosstab(df['promotion_flag'], df['is_stockout'])
        
        chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)

        so_promo_pct = (contingency_table.loc[1, 1] / contingency_table.loc[1].sum() * 100) if 1 in contingency_table.index else 0
        so_non_promo_pct = (contingency_table.loc[0, 1] / contingency_table.loc[0].sum() * 100) if 0 in contingency_table.index else 0

        decision = "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
        interp = (
            f"Stock-out occurrence is significantly associated with promotion status (p = {p_val:.4e} < 0.05). "
            f"Stock-out rate during promotions ({so_promo_pct:.1f}%) differs significantly from non-promotion periods ({so_non_promo_pct:.1f}%)."
            if p_val < 0.05 else
            f"No statistically significant association between stock-out frequency and promotion status (p = {p_val:.4f} >= 0.05)."
        )

        results.append({
            "analysis_id": "Analysis 3",
            "question": "Is stock-out frequency associated with promotion status?",
            "h0": "Stock-out occurrence is independent of promotion status",
            "h1": "Stock-out occurrence is associated with promotion status",
            "test_name": "Chi-Square Test of Independence (χ²)",
            "sample_def": f"2x2 Contingency Table (N={len(df)} observations)",
            "statistic": round(float(chi2_stat), 4),
            "p_value": round(float(p_val), 6),
            "p_value_fmt": f"{p_val:.4e}",
            "decision": decision,
            "business_interpretation": interp
        })

    return results


def format_statistical_report_table(results: List[Dict[str, Any]]) -> str:
    """
    Format statistical test results into Markdown summary table required for reports/statistical_analysis.md.
    """
    lines = [
        "| Analysis | H0 | H1 | Test | Statistic | p-value | Decision | Business Interpretation |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in results:
        line = (
            f"| **{r['analysis_id']}** | {r['h0']} | {r['h1']} | {r['test_name']} | "
            f"{r['statistic']} | {r['p_value_fmt']} | **{r['decision']}** | {r['business_interpretation']} |"
        )
        lines.append(line)
    return "\n".join(lines)
