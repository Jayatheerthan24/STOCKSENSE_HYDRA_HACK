"""
Streamlit Sidebar Filtering and UI Utilities module for StockSense Dashboard.
"""

import pandas as pd
import numpy as np
import streamlit as st


def render_sidebar_filters(df: pd.DataFrame, recs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Render professional sidebar filters for Date range, City, Store, Category, Sub-category,
    Brand, Store Type, Region, Risk Level, and Promotion Status.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (filtered_master_df, filtered_recs_df)
    """
    st.sidebar.markdown("## 🔍 Global Dashboard Filters")

    if st.sidebar.button("🔄 Reset All Filters"):
        st.session_state.clear()
        st.rerun()

    filtered_df = df.copy()
    filtered_recs = recs_df.copy()

    # 1. Date Range Filter
    if 'date' in filtered_df.columns:
        min_date = filtered_df['date'].min().date()
        max_date = filtered_df['date'].max().date()
        date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) & (filtered_df['date'].dt.date <= end_date)]
            if not filtered_recs.empty and 'date' in filtered_recs.columns:
                filtered_recs = filtered_recs[(filtered_recs['date'].dt.date >= start_date) & (filtered_recs['date'].dt.date <= end_date)]

    # 2. City Filter
    if 'city' in df.columns:
        cities = sorted(df['city'].dropna().unique())
        selected_cities = st.sidebar.multiselect("City", options=cities, default=[])
        if selected_cities:
            filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]
            if 'city' in filtered_recs.columns:
                filtered_recs = filtered_recs[filtered_recs['city'].isin(selected_cities)]

    # 3. Store ID Filter
    if 'store_id' in df.columns:
        stores = sorted(df['store_id'].dropna().unique())
        selected_stores = st.sidebar.multiselect("Store ID", options=stores, default=[])
        if selected_stores:
            filtered_df = filtered_df[filtered_df['store_id'].isin(selected_stores)]
            if 'store_id' in filtered_recs.columns:
                filtered_recs = filtered_recs[filtered_recs['store_id'].isin(selected_stores)]

    # 4. Store Type Filter
    if 'store_type' in df.columns:
        store_types = sorted(df['store_type'].dropna().unique())
        selected_stypes = st.sidebar.multiselect("Store Type", options=store_types, default=[])
        if selected_stypes:
            filtered_df = filtered_df[filtered_df['store_type'].isin(selected_stypes)]
            if 'store_type' in filtered_recs.columns:
                filtered_recs = filtered_recs[filtered_recs['store_type'].isin(selected_stypes)]

    # 5. Category Filter
    if 'category' in df.columns:
        categories = sorted(df['category'].dropna().unique())
        selected_cats = st.sidebar.multiselect("Category", options=categories, default=[])
        if selected_cats:
            filtered_df = filtered_df[filtered_df['category'].isin(selected_cats)]
            if 'category' in filtered_recs.columns:
                filtered_recs = filtered_recs[filtered_recs['category'].isin(selected_cats)]

    # 6. Sub-Category Filter
    if 'sub_category' in df.columns:
        sub_cats = sorted(df['sub_category'].dropna().unique())
        selected_subcats = st.sidebar.multiselect("Sub-Category", options=sub_cats, default=[])
        if selected_subcats:
            filtered_df = filtered_df[filtered_df['sub_category'].isin(selected_subcats)]

    # 7. Product ID Filter
    if 'product_id' in df.columns:
        products = sorted(df['product_id'].dropna().unique())
        selected_prods = st.sidebar.multiselect("Product ID", options=products, default=[])
        if selected_prods:
            filtered_df = filtered_df[filtered_df['product_id'].isin(selected_prods)]
            if 'product_id' in filtered_recs.columns:
                filtered_recs = filtered_recs[filtered_recs['product_id'].isin(selected_prods)]

    # 8. Risk Level Filter (Manager Recommendations)
    if not filtered_recs.empty and 'risk_level' in filtered_recs.columns:
        risk_levels = ['HIGH', 'MEDIUM', 'LOW']
        selected_risks = st.sidebar.multiselect("Risk Level Tier", options=risk_levels, default=[])
        if selected_risks:
            filtered_recs = filtered_recs[filtered_recs['risk_level'].isin(selected_risks)]

    # 9. Promotion Status Filter
    if 'promotion_flag' in df.columns:
        promo_options = ['All', 'Promotional Only', 'Non-Promotional Only']
        promo_choice = st.sidebar.selectbox("Promotion Status", options=promo_options, index=0)
        if promo_choice == 'Promotional Only':
            filtered_df = filtered_df[filtered_df['promotion_flag'] == 1]
        elif promo_choice == 'Non-Promotional Only':
            filtered_df = filtered_df[filtered_df['promotion_flag'] == 0]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Active Master Observations: {len(filtered_df):,}")
    st.sidebar.caption(f"Active Recommendations: {len(filtered_recs):,}")

    return filtered_df, filtered_recs


def style_risk_level(val: str) -> str:
    """CSS background styling for Risk Level dataframe cells."""
    if val == 'HIGH':
        return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
    elif val == 'MEDIUM':
        return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
    elif val == 'LOW':
        return 'background-color: #d4edda; color: #155724;'
    return ''
