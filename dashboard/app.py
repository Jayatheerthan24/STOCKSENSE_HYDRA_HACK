"""
StockSense - AI-Powered Retail Inventory Intelligence Dashboard (NovaMart Prototype)
====================================================================================
Streamlit Round 3 Application integrating Round 1 & Round 2 Machine Learning Outputs.

Run command:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Set Streamlit page config as first command
st.set_page_config(
    page_title="StockSense - Retail Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dashboard.data_loader import (
    load_master_dataset, load_modeling_dataset, load_manager_recommendations,
    load_data_quality_summary, load_model_metrics, load_feature_importances, load_trained_models
)
from dashboard.visualizations import (
    plot_revenue_by_category, plot_demand_trend, plot_actual_vs_predicted_demand,
    plot_stockout_risk_heatmap, plot_risk_distribution, plot_feature_importance_bar, plot_promo_vs_nonpromo
)
from dashboard.dashboard_utils import render_sidebar_filters, style_risk_level


def main():
    # -------------------------------------------------------------
    # App Header & Banner
    # -------------------------------------------------------------
    st.title("📦 StockSense")
    st.subheader("AI-Powered Retail Inventory Intelligence (NovaMart Operational Prototype)")
    st.caption("Demand forecasting • Stock-out risk • Explainable replenishment")
    st.markdown("---")

    # Load Data & Models
    master_df = load_master_dataset()
    modeling_df = load_modeling_dataset()
    recs_df = load_manager_recommendations()
    quality_summary = load_data_quality_summary()
    dem_metrics, so_metrics = load_model_metrics()
    dem_imp, so_imp = load_feature_importances()
    dem_model, so_model = load_trained_models()

    if master_df.empty:
        st.error("Error: Could not load data/processed/master_dataset.csv. Please ensure Round 1 & Round 2 pipelines have executed.")
        return

    # Render Sidebar Filters
    filtered_master, filtered_recs = render_sidebar_filters(master_df, recs_df)

    # Sidebar Navigation
    st.sidebar.markdown("## 🧭 Dashboard Navigation")
    pages = [
        "📊 Executive Summary",
        "📈 Demand Intelligence",
        "⚠️ Inventory Risk",
        "🎯 Manager Action Centre",
        "🔍 Product Detail View",
        "🤖 Model Performance",
        "💡 Model Explainability",
        "📑 Executive Reports & Export"
    ]
    selected_page = st.sidebar.radio("Go to Page", pages)

    # -------------------------------------------------------------
    # Page 1: Executive Summary
    # -------------------------------------------------------------
    if selected_page == "📊 Executive Summary":
        st.header("📊 Executive Summary")
        date_str = f"{filtered_master['date'].min().strftime('%Y-%m-%d')} to {filtered_master['date'].max().strftime('%Y-%m-%d')}"
        st.info(f"**Operational Reporting Period:** {date_str} | Active Store Observations: {len(filtered_master):,}")

        # Top Executive KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)

        tot_rev = filtered_master['daily_revenue'].sum() if 'daily_revenue' in filtered_master.columns else 0
        tot_units = filtered_master['daily_units_sold'].sum() if 'daily_units_sold' in filtered_master.columns else 0

        if 'closing' in filtered_master.columns:
            so_rate = (filtered_master['closing'] <= 0).mean() * 100
        else:
            so_rate = 0.0

        high_risk_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'HIGH']) if not filtered_recs.empty and 'risk_level' in filtered_recs.columns else 0
        tot_reorder_qty = filtered_recs['recommended_reorder_qty'].sum() if not filtered_recs.empty and 'recommended_reorder_qty' in filtered_recs.columns else 0

        col1.metric("Total Revenue", f"INR {tot_rev:,.0f}")
        col2.metric("Units Sold", f"{tot_units:,.0f}")
        col3.metric("Stock-out Rate", f"{so_rate:.2f}%")
        col4.metric("HIGH Risk Items", f"{high_risk_cnt}")
        col5.metric("Total Reorder Qty", f"{tot_reorder_qty:,} units")

        st.markdown("---")

        # Visual Grid
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.plotly_chart(plot_demand_trend(filtered_master), use_container_width=True)
            st.plotly_chart(plot_revenue_by_category(filtered_master), use_container_width=True)

        with vcol2:
            if not filtered_recs.empty:
                st.plotly_chart(plot_risk_distribution(filtered_recs), use_container_width=True)
            st.plotly_chart(plot_stockout_risk_heatmap(filtered_master), use_container_width=True)

        # Dynamic Data-Driven Attention Insights
        st.markdown("### 🚨 What Needs Immediate Attention?")
        att_col1, att_col2 = st.columns(2)

        with att_col1:
            st.subheader("Top HIGH Risk Store x Product Items")
            if not filtered_recs.empty and 'risk_level' in filtered_recs.columns:
                high_df = filtered_recs[filtered_recs['risk_level'] == 'HIGH'].sort_values(by='stockout_probability', ascending=False).head(5)
                show_cols = [c for c in ['store_id', 'product_id', 'category', 'closing', 'forecast_demand', 'recommended_reorder_qty', 'risk_level'] if c in high_df.columns]
                st.dataframe(high_df[show_cols], hide_index=True, use_container_width=True)
            else:
                st.write("No high-risk items detected in current filter.")

        with att_col2:
            st.subheader("Categories with Highest Stock-out Rates")
            if 'category' in filtered_master.columns and 'closing' in filtered_master.columns:
                cat_so = filtered_master.groupby('category')['closing'].apply(lambda s: (s <= 0).mean() * 100).reset_index()
                cat_so.columns = ['Category', 'Stock-out Rate (%)']
                cat_so = cat_so.sort_values(by='Stock-out Rate (%)', ascending=False)
                st.dataframe(cat_so, hide_index=True, use_container_width=True)

    # -------------------------------------------------------------
    # Page 2: Demand Intelligence
    # -------------------------------------------------------------
    elif selected_page == "📈 Demand Intelligence":
        st.header("📈 Demand Intelligence & Forecasting")

        dcol1, dcol2, dcol3 = st.columns(3)
        tot_units = filtered_master['daily_units_sold'].sum() if 'daily_units_sold' in filtered_master.columns else 0
        avg_units = filtered_master['daily_units_sold'].mean() if 'daily_units_sold' in filtered_master.columns else 0

        # Promo lift calculation
        if 'promotion_flag' in filtered_master.columns and 'daily_units_sold' in filtered_master.columns:
            p_mean = filtered_master[filtered_master['promotion_flag'] == 1]['daily_units_sold'].mean()
            np_mean = filtered_master[filtered_master['promotion_flag'] == 0]['daily_units_sold'].mean()
            promo_lift = ((p_mean - np_mean) / np_mean * 100) if np_mean and np_mean > 0 else 0.0
        else:
            promo_lift = 0.0

        dcol1.metric("Total Volume Demand", f"{tot_units:,.0f} units")
        dcol2.metric("Mean Daily Demand", f"{avg_units:.2f} units/day")
        dcol3.metric("Promotional Sales Lift", f"+{promo_lift:.2f}%")

        st.markdown("---")

        # Forecast Line Chart
        st.plotly_chart(plot_actual_vs_predicted_demand(modeling_df), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_promo_vs_nonpromo(filtered_master), use_container_width=True)
        with col_right:
            st.plotly_chart(plot_revenue_by_category(filtered_master), use_container_width=True)

        st.markdown("### 📋 Historical Demand & Forecast Table")
        if not modeling_df.empty:
            show_m_cols = [c for c in ['date', 'store_id', 'product_id', 'category', 'daily_units_sold', 'lag_1', 'rolling_mean_7', 'next_7_day_demand'] if c in modeling_df.columns]
            st.dataframe(modeling_df[show_m_cols].head(50), hide_index=True, use_container_width=True)
            st.download_button(
                label="📥 Download Demand Data CSV",
                data=modeling_df[show_m_cols].to_csv(index=False).encode('utf-8'),
                file_name="demand_intelligence_data.csv",
                mime="text/csv"
            )

    # -------------------------------------------------------------
    # Page 3: Inventory Risk
    # -------------------------------------------------------------
    elif selected_page == "⚠️ Inventory Risk":
        st.header("⚠️ Inventory Risk & Stock-out Analysis")

        icol1, icol2, icol3, icol4 = st.columns(4)
        if 'closing' in filtered_master.columns:
            so_events = (filtered_master['closing'] <= 0).sum()
            so_pct = (filtered_master['closing'] <= 0).mean() * 100
        else:
            so_events, so_pct = 0, 0.0

        icol1.metric("Overall Stock-out Rate", f"{so_pct:.2f}%")
        icol2.metric("Stock-out Event Observations", f"{so_events:,}")
        icol3.metric("Stores Covered", f"{filtered_master['store_id'].nunique() if 'store_id' in filtered_master.columns else 0}")
        icol4.metric("Products Tracked", f"{filtered_master['product_id'].nunique() if 'product_id' in filtered_master.columns else 0}")

        st.markdown("---")

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.plotly_chart(plot_stockout_risk_heatmap(filtered_master), use_container_width=True)
        with rcol2:
            if not filtered_recs.empty:
                st.plotly_chart(plot_risk_distribution(filtered_recs), use_container_width=True)

        st.markdown("### 📋 Detailed Inventory Risk Table")
        if not filtered_recs.empty:
            sort_by = st.selectbox("Sort Table By", options=['stockout_probability', 'recommended_reorder_qty', 'forecast_demand', 'closing'], index=0)
            sorted_recs = filtered_recs.sort_values(by=sort_by, ascending=False)
            st.dataframe(sorted_recs, hide_index=True, use_container_width=True)
            st.download_button(
                label="📥 Download Inventory Risk CSV",
                data=sorted_recs.to_csv(index=False).encode('utf-8'),
                file_name="inventory_risk_data.csv",
                mime="text/csv"
            )

    # -------------------------------------------------------------
    # Page 4: Manager Action Centre
    # -------------------------------------------------------------
    elif selected_page == "🎯 Manager Action Centre":
        st.header("🎯 Manager Action Centre (NovaMart Operational Reorder Layer)")
        st.write("Actionable inventory replenishment recommendations generated by machine learning models.")

        if filtered_recs.empty:
            st.warning("No recommendations available for current filter selection.")
        else:
            # Cards
            c1, c2, c3, c4 = st.columns(4)
            h_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'HIGH'])
            m_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'MEDIUM'])
            l_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'LOW'])
            tot_qty = filtered_recs['recommended_reorder_qty'].sum()

            c1.metric("HIGH Risk Items", f"{h_cnt}", delta="Urgent Action Required" if h_cnt > 0 else None, delta_color="inverse")
            c2.metric("MEDIUM Risk Items", f"{m_cnt}")
            c3.metric("LOW Risk Items", f"{l_cnt}")
            c4.metric("Total Units to Reorder", f"{tot_qty:,} units")

            if h_cnt > 0:
                st.error(f"🚨 **IMMEDIATE ATTENTION REQUIRED:** {h_cnt} store x product items are at HIGH risk of stock-out or currently out of stock!")

            st.markdown("---")
            st.markdown("### 📋 Reorder Action Table")

            # Sort and style
            act_df = filtered_recs.sort_values(by=['risk_level', 'stockout_probability'], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(act_df, hide_index=True, use_container_width=True)

            st.download_button(
                label="📥 Download Manager Action List CSV",
                data=act_df.to_csv(index=False).encode('utf-8'),
                file_name="manager_action_centre_recommendations.csv",
                mime="text/csv"
            )

            # Drill-down explainer
            st.markdown("### 💡 Item Specific Manager Action Explainer")
            item_options = [f"{row['store_id']} | {row['product_id']} | {row['category']} (Risk: {row['risk_level']})" for _, row in act_df.iterrows()]
            selected_item = st.selectbox("Select Item to Inspect Action Explanation", options=item_options, index=0)

            if selected_item:
                idx = item_options.index(selected_item)
                sel_row = act_df.iloc[idx]

                ex_col1, ex_col2 = st.columns(2)
                with ex_col1:
                    st.info(f"**Why is this risky?**\n- Primary Driver: {sel_row.get('top_reason_1', 'N/A')}\n- Secondary Driver: {sel_row.get('top_reason_2', 'N/A')}")
                    st.warning(f"**How urgent is it?**\n- Stock-out Probability: **{sel_row.get('stockout_probability', 0.0):.1%}**\n- Risk Tier: **{sel_row.get('risk_level', 'N/A')}**")

                with ex_col2:
                    st.success(f"**What should the manager do?**\n- Action: {sel_row.get('manager_recommendation', 'N/A')}\n- **Recommended Reorder Qty:** **{sel_row.get('recommended_reorder_qty', 0)} units**")
                    st.write(f"- Forecast 7-Day Demand: {sel_row.get('forecast_demand', 0):.0f} units\n- Closing Inventory: {sel_row.get('closing', 0):.0f} units\n- Calculated Safety Stock: {sel_row.get('safety_stock', 0):.0f} units")

    # -------------------------------------------------------------
    # Page 5: Product Detail View
    # -------------------------------------------------------------
    elif selected_page == "🔍 Product Detail View":
        st.header("🔍 Product & Store Deep-Dive Drill-Down")

        stores = sorted(master_df['store_id'].unique()) if 'store_id' in master_df.columns else []
        prods = sorted(master_df['product_id'].unique()) if 'product_id' in master_df.columns else []

        pcol1, pcol2 = st.columns(2)
        sel_store = pcol1.selectbox("Select Store ID", options=stores, index=0 if stores else None)
        sel_prod = pcol2.selectbox("Select Product ID", options=prods, index=0 if prods else None)

        if sel_store and sel_prod:
            prod_master = master_df[(master_df['store_id'] == sel_store) & (master_df['product_id'] == sel_prod)].sort_values(by='date')
            prod_rec = recs_df[(recs_df['store_id'] == sel_store) & (recs_df['product_id'] == sel_prod)] if not recs_df.empty else pd.DataFrame()

            if prod_master.empty:
                st.warning("No historical observations found for selected Store and Product.")
            else:
                latest_master = prod_master.iloc[-1]
                latest_rec = prod_rec.iloc[-1] if not prod_rec.empty else pd.Series()

                st.markdown(f"### Product Profile: `{sel_prod}` ({latest_master.get('category', 'N/A')} - {latest_master.get('sub_category', 'N/A')})")
                st.write(f"**Brand:** {latest_master.get('brand', 'N/A')} | **MRP:** INR {latest_master.get('mrp', 0)} | **Cost:** INR {latest_master.get('cost_price', 0)} | **Store:** {sel_store} ({latest_master.get('city', 'N/A')} - {latest_master.get('store_type', 'N/A')})")

                # Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Closing Stock", f"{latest_master.get('closing', 0):.0f} units")
                m2.metric("Reorder Level", f"{latest_master.get('reorder_lvl', 0):.0f} units")
                m3.metric("Lead Days", f"{latest_master.get('lead_days', 0):.0f} days")
                m4.metric("Forecast 7-Day Demand", f"{latest_rec.get('forecast_demand', 0):.0f} units")
                m5.metric("Reorder Qty", f"{latest_rec.get('recommended_reorder_qty', 0):.0f} units")

                st.markdown("---")
                # Plot product historical demand
                st.plotly_chart(plot_demand_trend(prod_master), use_container_width=True)

                st.markdown("### 📝 Manager Summary Note")
                closing_val = latest_master.get('closing', 0)
                fc_val = latest_rec.get('forecast_demand', 0)
                prob_val = latest_rec.get('stockout_probability', 0.0)
                reorder_val = latest_rec.get('recommended_reorder_qty', 0)

                st.info(f"**Current stock is {closing_val:.0f} units versus expected 7-day demand of {fc_val:.0f} units. Stock-out probability is {prob_val:.1%}. Recommended reorder quantity is {reorder_val:.0f} units.**")

    # -------------------------------------------------------------
    # Page 6: Model Performance
    # -------------------------------------------------------------
    elif selected_page == "🤖 Model Performance":
        st.header("🤖 Machine Learning Model Evaluation & Benchmark")
        st.caption("All evaluation metrics are calculated on the held-out test set with zero data leakage.")

        st.markdown("### 1. Demand Forecasting Model Performance (`target = next_7_day_demand`)")
        if not dem_metrics.empty:
            st.dataframe(dem_metrics, hide_index=True, use_container_width=True)

        st.markdown("### 2. Stock-out Risk Classification Performance (`target = stockout_flag`)")
        if not so_metrics.empty:
            st.dataframe(so_metrics, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Model Diagnostic Plots")
        img_col1, img_col2, img_col3 = st.columns(3)

        fig_dir = PROJECT_ROOT / "reports" / "figures"
        with img_col1:
            p1 = fig_dir / "demand_actual_vs_predicted.png"
            if p1.exists():
                st.image(str(p1), caption="Demand: Actual vs Predicted Scatter")
        with img_col2:
            p2 = fig_dir / "stockout_confusion_matrix.png"
            if p2.exists():
                st.image(str(p2), caption="Stockout Risk: Confusion Matrix")
        with img_col3:
            p3 = fig_dir / "stockout_roc_curve.png"
            if p3.exists():
                st.image(str(p3), caption="Stockout Risk: ROC Curve (AUC)")

    # -------------------------------------------------------------
    # Page 7: Model Explainability
    # -------------------------------------------------------------
    elif selected_page == "💡 Model Explainability":
        st.header("💡 Model Explainability & Feature Importance")
        st.write("Feature importance indicates global model influence, not direct causality.")

        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.plotly_chart(plot_feature_importance_bar(dem_imp, "Top Drivers: Demand Forecasting Model"), use_container_width=True)
            st.info("**Demand Drivers Interpretation:** Recent 7-day sales velocity (`rolling_mean_7`), reorder thresholds, and promotional status exert the strongest influence on 7-day demand predictions.")

        with ecol2:
            st.plotly_chart(plot_feature_importance_bar(so_imp, "Top Drivers: Stock-out Risk Model"), use_container_width=True)
            st.warning("**Stock-out Drivers Interpretation:** Closing inventory levels (`closing`) relative to reorder points (`reorder_lvl`) and lead time demand determine stock-out risk probabilities.")

    # -------------------------------------------------------------
    # Page 8: Executive Reports & Export
    # -------------------------------------------------------------
    elif selected_page == "📑 Executive Reports & Export":
        st.header("📑 Executive Markdown Reports & CSV Export Centre")

        rep_options = [
            "Data Quality Audit Report",
            "Business EDA Insights Report",
            "Statistical Hypothesis Testing Report",
            "Model Explainability Report",
            "Recommendation Engine Logic Report",
            "Round 2 Compliance Validation Report"
        ]
        sel_rep = st.selectbox("Select Report to View", options=rep_options)

        rep_map = {
            "Data Quality Audit Report": REPORTS_DIR / "data_quality_report.md",
            "Business EDA Insights Report": REPORTS_DIR / "eda_insights.md",
            "Statistical Hypothesis Testing Report": REPORTS_DIR / "statistical_analysis.md",
            "Model Explainability Report": REPORTS_DIR / "model_explainability.md",
            "Recommendation Engine Logic Report": REPORTS_DIR / "recommendation_logic.md",
            "Round 2 Compliance Validation Report": REPORTS_DIR / "round2_validation.md"
        }

        target_file = rep_map.get(sel_rep)
        if target_file and target_file.exists():
            st.markdown(target_file.read_text(encoding="utf-8"))
        else:
            st.warning("Report file not found.")

        st.markdown("---")
        st.markdown("### 📥 Download Processed CSV Datasets")

        d1, d2, d3 = st.columns(3)
        with d1:
            if not recs_df.empty:
                st.download_button(
                    label="📥 Manager Recommendations CSV",
                    data=recs_df.to_csv(index=False).encode('utf-8'),
                    file_name="manager_recommendations.csv",
                    mime="text/csv"
                )
        with d2:
            if not master_df.empty:
                st.download_button(
                    label="📥 Master Dataset CSV",
                    data=master_df.to_csv(index=False).encode('utf-8'),
                    file_name="master_dataset.csv",
                    mime="text/csv"
                )
        with d3:
            if not quality_summary.empty:
                st.download_button(
                    label="📥 Quality Audit Summary CSV",
                    data=quality_summary.to_csv(index=False).encode('utf-8'),
                    file_name="data_quality_summary.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
