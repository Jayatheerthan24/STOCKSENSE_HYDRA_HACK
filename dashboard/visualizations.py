"""
Plotly Visualizations module for StockSense Streamlit Dashboard.
Constructs interactive, layman-friendly charts for store inventory and sales management.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_revenue_by_category(df: pd.DataFrame) -> go.Figure:
    """Bar chart of Total Revenue by Category."""
    if df.empty or 'category' not in df.columns or 'daily_revenue' not in df.columns:
        return go.Figure()

    cat_df = df.groupby('category')['daily_revenue'].sum().reset_index().sort_values(by='daily_revenue', ascending=False)
    fig = px.bar(
        cat_df,
        x='category',
        y='daily_revenue',
        color='category',
        color_discrete_sequence=px.colors.sequential.Blues_r,
        text_auto=',.0f',
        title='Total Revenue by Product Category (INR)',
        labels={'daily_revenue': 'Total Revenue (INR)', 'category': 'Category'}
    )
    fig.update_layout(showlegend=False, template='plotly_white', height=360)
    return fig


def plot_demand_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of Daily Aggregate Demand Trend."""
    if df.empty or 'date' not in df.columns or 'daily_units_sold' not in df.columns:
        return go.Figure()

    daily_df = df.groupby('date')['daily_units_sold'].sum().reset_index()
    fig = px.line(
        daily_df,
        x='date',
        y='daily_units_sold',
        markers=True,
        line_shape='spline',
        title='Daily Store Sales Velocity (Units Sold)',
        labels={'daily_units_sold': 'Units Sold', 'date': 'Date'}
    )
    fig.update_traces(line_color='#008080', line_width=2.5)
    fig.update_layout(template='plotly_white', height=360)
    return fig


def plot_actual_vs_predicted_demand(modeling_df: pd.DataFrame) -> go.Figure:
    """Line chart comparing Actual vs Expected Demand over time."""
    if modeling_df.empty or 'date' not in modeling_df.columns or 'next_7_day_demand' not in modeling_df.columns:
        return go.Figure()

    df_agg = modeling_df.groupby('date')[['next_7_day_demand']].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_agg['date'],
        y=df_agg['next_7_day_demand'],
        mode='lines+markers',
        name='Expected 7-Day Demand',
        line=dict(color='#2b5c8f', width=2.5)
    ))

    fig.update_layout(
        title='Expected 7-Day Customer Demand Projection',
        xaxis_title='Date',
        yaxis_title='Expected 7-Day Units Demand',
        template='plotly_white',
        height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def plot_stockout_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of Stock-out Frequency by Store x Category."""
    if df.empty or 'store_id' not in df.columns or 'category' not in df.columns or 'closing' not in df.columns:
        return go.Figure()

    df_temp = df.copy()
    df_temp['is_stockout'] = (df_temp['closing'] <= 0).astype(int)
    pivot = df_temp.pivot_table(index='store_id', columns='category', values='is_stockout', aggfunc='mean') * 100

    fig = px.imshow(
        pivot.round(1),
        labels=dict(x="Category", y="Store ID", color="Stock-out Rate (%)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale='YlOrRd',
        text_auto='.1f',
        title='Stock-out Risk Heatmap (Store x Category Stock-out %)'
    )
    fig.update_layout(template='plotly_white', height=360)
    return fig


def plot_risk_distribution(recs_df: pd.DataFrame) -> go.Figure:
    """Donut chart of Risk Level Distribution (HIGH, MEDIUM, LOW)."""
    if recs_df.empty or 'risk_level' not in recs_df.columns:
        return go.Figure()

    risk_counts = recs_df['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['risk_level', 'count']

    color_map = {'HIGH': '#d9534f', 'MEDIUM': '#f0ad4e', 'LOW': '#5cb85c'}

    fig = px.pie(
        risk_counts,
        names='risk_level',
        values='count',
        color='risk_level',
        color_discrete_map=color_map,
        hole=0.45,
        title='Inventory Risk Stratification'
    )
    fig.update_traces(textinfo='percent+label+value')
    fig.update_layout(template='plotly_white', height=360)
    return fig


def plot_promo_vs_nonpromo(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing Promotional vs Non-Promotional daily sales."""
    if df.empty or 'promotion_flag' not in df.columns or 'daily_units_sold' not in df.columns:
        return go.Figure()

    df_temp = df.copy()
    df_temp['promo_label'] = df_temp['promotion_flag'].map({0: 'Regular Days', 1: 'Promotional Days'})
    promo_agg = df_temp.groupby('promo_label')['daily_units_sold'].mean().reset_index()

    fig = px.bar(
        promo_agg,
        x='promo_label',
        y='daily_units_sold',
        color='promo_label',
        color_discrete_sequence=['#6c757d', '#0275d8'],
        text_auto='.1f',
        title='Average Sales: Regular vs Promotional Days',
        labels={'daily_units_sold': 'Average Daily Units Sold', 'promo_label': 'Day Type'}
    )
    fig.update_layout(showlegend=False, template='plotly_white', height=360)
    return fig
