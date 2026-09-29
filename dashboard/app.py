"""
StockSense - Retail Inventory Intelligence Dashboard (NovaMart Operations Prototype)
===================================================================================
A clean, professional, layman-friendly decision support system for store inventory managers.

Run command:
    python -m streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Streamlit Page Config
st.set_page_config(
    page_title="StockSense - Store Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"

from dashboard.data_loader import (
    load_master_dataset, load_modeling_dataset, load_manager_recommendations,
    load_data_quality_summary
)
from dashboard.visualizations import (
    plot_revenue_by_category, plot_demand_trend, plot_actual_vs_predicted_demand,
    plot_stockout_risk_heatmap, plot_risk_distribution, plot_promo_vs_nonpromo
)
from dashboard.dashboard_utils import render_sidebar_filters


def main():
    # -------------------------------------------------------------
    # App Header Banner
    # -------------------------------------------------------------
    st.title("📦 StockSense")
    st.subheader("Smart Retail Inventory Intelligence System (NovaMart Prototype)")
    st.caption("Automated Demand Forecasting • Stock-out Prevention • Manager Action Centre")
    st.markdown("---")

    # Load Data
    master_df = load_master_dataset()
    modeling_df = load_modeling_dataset()
    recs_df = load_manager_recommendations()
    quality_summary = load_data_quality_summary()

    if master_df.empty:
        st.error("Error: Could not load data/processed/master_dataset.csv. Please ensure data processing pipelines have run.")
        return

    # Render Sidebar Filters
    filtered_master, filtered_recs = render_sidebar_filters(master_df, recs_df)

    # Sidebar Navigation - Layman / Store Manager Navigation Only
    st.sidebar.markdown("## 🧭 Manager Navigation")
    pages = [
        "📊 Executive Summary",
        "📈 Sales & Demand Intelligence",
        "⚠️ Inventory Risk & Stock-outs",
        "🎯 Reorder Action Centre",
        "🔍 Store & Product Inspector",
        "📑 Reports & Data Export"
    ]
    selected_page = st.sidebar.radio("Go to Section", pages)

    # Safe Date Range String
    if not filtered_master.empty and 'date' in filtered_master.columns and filtered_master['date'].notna().any():
        min_d = filtered_master['date'].min()
        max_d = filtered_master['date'].max()
        date_str = f"{min_d.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}"
    else:
        date_str = "No active records for current filter selection"

    # -------------------------------------------------------------
    # Page 1: Executive Summary
    # -------------------------------------------------------------
    if selected_page == "📊 Executive Summary":
        st.header("📊 Executive Summary")
        st.info(f"**Operational Period:** {date_str} | Active Store Observations: {len(filtered_master):,}")

        if filtered_master.empty:
            st.warning("No records match the selected filter. Please click 'Reset All Filters' in the sidebar.")
            return

        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)

        tot_rev = filtered_master['daily_revenue'].sum() if 'daily_revenue' in filtered_master.columns else 0
        tot_units = filtered_master['daily_units_sold'].sum() if 'daily_units_sold' in filtered_master.columns else 0

        if 'closing' in filtered_master.columns and len(filtered_master) > 0:
            so_rate = (filtered_master['closing'] <= 0).mean() * 100
        else:
            so_rate = 0.0

        high_risk_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'HIGH']) if not filtered_recs.empty and 'risk_level' in filtered_recs.columns else 0
        tot_reorder_qty = filtered_recs['recommended_reorder_qty'].sum() if not filtered_recs.empty and 'recommended_reorder_qty' in filtered_recs.columns else 0

        col1.metric("Total Revenue", f"INR {tot_rev:,.0f}")
        col2.metric("Total Units Sold", f"{tot_units:,.0f}")
        col3.metric("Stock-out Rate", f"{so_rate:.2f}%")
        col4.metric("HIGH Risk Items", f"{high_risk_cnt}")
        col5.metric("Reorder Qty Needed", f"{tot_reorder_qty:,} units")

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
            st.subheader("Urgent HIGH Risk Items")
            if not filtered_recs.empty and 'risk_level' in filtered_recs.columns:
                high_df = filtered_recs[filtered_recs['risk_level'] == 'HIGH'].sort_values(by='recommended_reorder_qty', ascending=False).head(5)
                show_cols = [c for c in ['store_id', 'product_id', 'category', 'closing', 'forecast_demand', 'recommended_reorder_qty', 'manager_recommendation'] if c in high_df.columns]
                st.dataframe(high_df[show_cols], hide_index=True, use_container_width=True)
            else:
                st.write("No high-risk items detected in current selection.")

        with att_col2:
            st.subheader("Highest Stock-out Categories")
            if 'category' in filtered_master.columns and 'closing' in filtered_master.columns and not filtered_master.empty:
                cat_so = filtered_master.groupby('category')['closing'].apply(lambda s: (s <= 0).mean() * 100).reset_index()
                cat_so.columns = ['Category', 'Stock-out Rate (%)']
                cat_so = cat_so.sort_values(by='Stock-out Rate (%)', ascending=False)
                st.dataframe(cat_so, hide_index=True, use_container_width=True)

    # -------------------------------------------------------------
    # Page 2: Sales & Demand Intelligence
    # -------------------------------------------------------------
    elif selected_page == "📈 Sales & Demand Intelligence":
        st.header("📈 Sales & Demand Intelligence")

        if filtered_master.empty:
            st.warning("No records match the selected filter.")
            return

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
        dcol2.metric("Average Daily Demand", f"{avg_units:.1f} units/day")
        dcol3.metric("Promotional Sales Lift", f"+{promo_lift:.1f}%")

        st.markdown("---")

        # Demand Forecast Chart
        st.plotly_chart(plot_actual_vs_predicted_demand(modeling_df), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_promo_vs_nonpromo(filtered_master), use_container_width=True)
        with col_right:
            st.plotly_chart(plot_revenue_by_category(filtered_master), use_container_width=True)

        st.markdown("### 📋 Historical Demand Summary Data")
        if not modeling_df.empty:
            show_m_cols = [c for c in ['date', 'store_id', 'product_id', 'category', 'daily_units_sold', 'lag_1', 'rolling_mean_7', 'next_7_day_demand'] if c in modeling_df.columns]
            st.dataframe(modeling_df[show_m_cols].head(50), hide_index=True, use_container_width=True)

    # -------------------------------------------------------------
    # Page 3: Inventory Risk & Stock-outs
    # -------------------------------------------------------------
    elif selected_page == "⚠️ Inventory Risk & Stock-outs":
        st.header("⚠️ Inventory Risk & Stock-out Analysis")

        if filtered_master.empty:
            st.warning("No records match the selected filter.")
            return

        icol1, icol2, icol3, icol4 = st.columns(4)
        if 'closing' in filtered_master.columns and len(filtered_master) > 0:
            so_events = (filtered_master['closing'] <= 0).sum()
            so_pct = (filtered_master['closing'] <= 0).mean() * 100
        else:
            so_events, so_pct = 0, 0.0

        icol1.metric("Overall Stock-out Rate", f"{so_pct:.2f}%")
        icol2.metric("Stock-out Event Count", f"{so_events:,}")
        icol3.metric("Active Stores", f"{filtered_master['store_id'].nunique() if 'store_id' in filtered_master.columns else 0}")
        icol4.metric("Active Products", f"{filtered_master['product_id'].nunique() if 'product_id' in filtered_master.columns else 0}")

        st.markdown("---")

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.plotly_chart(plot_stockout_risk_heatmap(filtered_master), use_container_width=True)
        with rcol2:
            if not filtered_recs.empty:
                st.plotly_chart(plot_risk_distribution(filtered_recs), use_container_width=True)

        st.markdown("### 📋 Store Inventory Positions & Risk Tier List")
        if not filtered_recs.empty:
            sort_by = st.selectbox("Sort Table By", options=['recommended_reorder_qty', 'forecast_demand', 'closing'], index=0)
            sorted_recs = filtered_recs.sort_values(by=sort_by, ascending=False)
            st.dataframe(sorted_recs, hide_index=True, use_container_width=True)

    # -------------------------------------------------------------
    # Page 4: Reorder Action Centre (Core Operational Tool)
    # -------------------------------------------------------------
    elif selected_page == "🎯 Reorder Action Centre":
        st.header("🎯 Manager Reorder Action Centre")
        st.write("Actionable inventory replenishment recommendations for store operations.")

        if filtered_recs.empty:
            st.warning("No recommendations available for current filter selection.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            h_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'HIGH'])
            m_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'MEDIUM'])
            l_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'LOW'])
            tot_qty = filtered_recs['recommended_reorder_qty'].sum()

            c1.metric("HIGH Risk Items", f"{h_cnt}", delta="Urgent Order Required" if h_cnt > 0 else None, delta_color="inverse")
            c2.metric("MEDIUM Risk Items", f"{m_cnt}")
            c3.metric("LOW Risk Items", f"{l_cnt}")
            c4.metric("Total Recommended Units", f"{tot_qty:,} units")

            if h_cnt > 0:
                st.error(f"🚨 **IMMEDIATE ATTENTION REQUIRED:** {h_cnt} store items are at HIGH risk of stock-out or currently out of stock!")

            st.markdown("---")
            st.markdown("### 📋 Recommended Store Reorder List")

            act_df = filtered_recs.sort_values(by=['risk_level', 'recommended_reorder_qty'], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(act_df, hide_index=True, use_container_width=True)

            st.download_button(
                label="📥 Download Manager Reorder List CSV",
                data=act_df.to_csv(index=False).encode('utf-8'),
                file_name="manager_reorder_action_list.csv",
                mime="text/csv"
            )

            # Item Action Explainer
            st.markdown("### 💡 Selected Item Action Explainer")
            item_options = [f"{row['store_id']} | {row['product_id']} | {row['category']} (Risk: {row['risk_level']})" for _, row in act_df.iterrows()]
            selected_item = st.selectbox("Select Item to View Action Details", options=item_options, index=0)

            if selected_item:
                idx = item_options.index(selected_item)
                sel_row = act_df.iloc[idx]

                ex_col1, ex_col2 = st.columns(2)
                with ex_col1:
                    st.info(f"**Why is this risky?**\n- Primary Factor: {sel_row.get('top_reason_1', 'N/A')}\n- Secondary Factor: {sel_row.get('top_reason_2', 'N/A')}")
                    st.warning(f"**Risk Level:** **{sel_row.get('risk_level', 'N/A')}**")

                with ex_col2:
                    st.success(f"**Manager Action:**\n- {sel_row.get('manager_recommendation', 'N/A')}\n- **Recommended Reorder Qty:** **{sel_row.get('recommended_reorder_qty', 0)} units**")
                    st.write(f"- Expected 7-Day Demand: {sel_row.get('forecast_demand', 0):.0f} units\n- Current Stock: {sel_row.get('closing', 0):.0f} units\n- Calculated Safety Stock: {sel_row.get('safety_stock', 0):.0f} units")

    # -------------------------------------------------------------
    # Page 5: Store & Product Inspector
    # -------------------------------------------------------------
    elif selected_page == "🔍 Store & Product Inspector":
        st.header("🔍 Store & Product Detail Inspector")

        stores = sorted(master_df['store_id'].unique()) if 'store_id' in master_df.columns else []
        prods = sorted(master_df['product_id'].unique()) if 'product_id' in master_df.columns else []

        pcol1, pcol2 = st.columns(2)
        sel_store = pcol1.selectbox("Select Store ID", options=stores, index=0 if stores else None)
        sel_prod = pcol2.selectbox("Select Product ID", options=prods, index=0 if prods else None)

        if sel_store and sel_prod:
            prod_master = master_df[(master_df['store_id'] == sel_store) & (master_df['product_id'] == sel_prod)].sort_values(by='date')
            prod_rec = recs_df[(recs_df['store_id'] == sel_store) & (recs_df['product_id'] == sel_prod)] if not recs_df.empty else pd.DataFrame()

            if prod_master.empty:
                st.warning("No historical records found for selected Store and Product.")
            else:
                latest_master = prod_master.iloc[-1]
                latest_rec = prod_rec.iloc[-1] if not prod_rec.empty else pd.Series()

                st.markdown(f"### Product: `{sel_prod}` ({latest_master.get('category', 'N/A')} - {latest_master.get('sub_category', 'N/A')})")
                st.write(f"**Brand:** {latest_master.get('brand', 'N/A')} | **MRP:** INR {latest_master.get('mrp', 0)} | **Cost:** INR {latest_master.get('cost_price', 0)} | **Store:** {sel_store} ({latest_master.get('city', 'N/A')} - {latest_master.get('store_type', 'N/A')})")

                # Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Current Stock", f"{latest_master.get('closing', 0):.0f} units")
                m2.metric("Reorder Level", f"{latest_master.get('reorder_lvl', 0):.0f} units")
                m3.metric("Supplier Lead Days", f"{latest_master.get('lead_days', 0):.0f} days")
                m4.metric("Expected 7-Day Demand", f"{latest_rec.get('forecast_demand', 0):.0f} units")
                m5.metric("Recommended Order Qty", f"{latest_rec.get('recommended_reorder_qty', 0):.0f} units")

                st.markdown("---")
                st.plotly_chart(plot_demand_trend(prod_master), use_container_width=True)

                st.markdown("### 📝 Manager Action Note")
                closing_val = latest_master.get('closing', 0)
                fc_val = latest_rec.get('forecast_demand', 0)
                reorder_val = latest_rec.get('recommended_reorder_qty', 0)

                st.info(f"**Current stock is {closing_val:.0f} units versus expected 7-day demand of {fc_val:.0f} units. Recommended reorder quantity is {reorder_val:.0f} units.**")

    # -------------------------------------------------------------
    # Page 6: Reports & Data Export
    # -------------------------------------------------------------
    elif selected_page == "📑 Reports & Data Export":
        st.header("📑 Reports & Data Export Centre")

        rep_options = [
            "Data Quality Audit Report",
            "Business Sales & EDA Report",
            "Recommendation Engine Logic Report"
        ]
        sel_rep = st.selectbox("Select Report to Read", options=rep_options)

        rep_map = {
            "Data Quality Audit Report": REPORTS_DIR / "data_quality_report.md",
            "Business Sales & EDA Report": REPORTS_DIR / "eda_insights.md",
            "Recommendation Engine Logic Report": REPORTS_DIR / "recommendation_logic.md"
        }

        target_file = rep_map.get(sel_rep)
        if target_file and target_file.exists():
            st.markdown(target_file.read_text(encoding="utf-8"))
        else:
            st.warning("Report file not found.")

        st.markdown("---")
        st.markdown("### 📥 CSV Downloads for Managers")

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
                    label="📥 Master Sales Dataset CSV",
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
