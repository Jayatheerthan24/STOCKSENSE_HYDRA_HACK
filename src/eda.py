"""
Exploratory Data Analysis (EDA) and Visualization module for StockSense Round 1.
Generates 10 business-focused charts saved to reports/figures/.
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend for saving figures without display server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from src.utils import FIGURES_DIR

# Set global visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})


def generate_all_eda_figures(master_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> List[str]:
    """
    Generate and save all 10 required Round 1 business EDA charts.
    
    Parameters:
        master_df (pd.DataFrame): Master Analytics Dataset
        output_dir (Path): Output directory for saved figure PNGs
        
    Returns:
        List[str]: List of paths to generated chart files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = master_df.copy()

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    generated_files = []

    # -------------------------------------------------------------
    # Chart 1: Revenue by Category
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    if 'category' in df.columns and 'daily_revenue' in df.columns:
        cat_rev = df.groupby('category')['daily_revenue'].sum().reset_index().sort_values(by='daily_revenue', ascending=False)
        ax = sns.barplot(data=cat_rev, x='category', y='daily_revenue', hue='category', palette='Blues_r', legend=False)
        plt.title('1. Total Revenue by Product Category (INR)')
        plt.xlabel('Category')
        plt.ylabel('Total Revenue (INR)')
        plt.xticks(rotation=30, ha='right')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'Rs.{height:,.0f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart1_path = output_dir / "01_revenue_by_category.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    generated_files.append(str(chart1_path))

    # -------------------------------------------------------------
    # Chart 2: Units Sold by Category
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    if 'category' in df.columns and 'daily_units_sold' in df.columns:
        cat_units = df.groupby('category')['daily_units_sold'].sum().reset_index().sort_values(by='daily_units_sold', ascending=False)
        ax = sns.barplot(data=cat_units, x='category', y='daily_units_sold', hue='category', palette='Greens_r', legend=False)
        plt.title('2. Total Units Sold by Product Category')
        plt.xlabel('Category')
        plt.ylabel('Total Units Sold (Quantity)')
        plt.xticks(rotation=30, ha='right')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:,.0f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart2_path = output_dir / "02_units_sold_by_category.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    generated_files.append(str(chart2_path))

    # -------------------------------------------------------------
    # Chart 3: Revenue by Store
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    if 'store_id' in df.columns and 'daily_revenue' in df.columns:
        store_rev = df.groupby(['store_id', 'city'])['daily_revenue'].sum().reset_index().sort_values(by='daily_revenue', ascending=False)
        store_rev['store_label'] = store_rev['store_id'] + " (" + store_rev['city'] + ")"
        ax = sns.barplot(data=store_rev, x='store_label', y='daily_revenue', hue='store_label', palette='Purples_r', legend=False)
        plt.title('3. Total Revenue by Store Location (INR)')
        plt.xlabel('Store (City)')
        plt.ylabel('Total Revenue (INR)')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'Rs.{height:,.0f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart3_path = output_dir / "03_revenue_by_store.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    generated_files.append(str(chart3_path))

    # -------------------------------------------------------------
    # Chart 4: Daily / Weekly Demand Trend
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    if 'date' in df.columns and 'daily_units_sold' in df.columns:
        daily_trend = df.groupby('date')['daily_units_sold'].sum().reset_index()
        sns.lineplot(data=daily_trend, x='date', y='daily_units_sold', marker='o', color='teal', linewidth=2)
        plt.title('4. Daily Total Demand Trend (Units Sold)')
        plt.xlabel('Date')
        plt.ylabel('Daily Aggregate Units Sold')
        plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    chart4_path = output_dir / "04_daily_demand_trend.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    generated_files.append(str(chart4_path))

    # -------------------------------------------------------------
    # Chart 5: Promotion vs Non-Promotion Sales
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    if 'promotion_flag' in df.columns and 'daily_units_sold' in df.columns:
        df['promo_label'] = df['promotion_flag'].map({0: 'Non-Promotional', 1: 'Promotional'})
        ax = sns.barplot(data=df, x='promo_label', y='daily_units_sold', hue='promo_label', errorbar=None, palette='Set2', legend=False)
        plt.title('5. Average Daily Units Sold: Promotion vs Non-Promotion')
        plt.xlabel('Promotion Status')
        plt.ylabel('Average Daily Units Sold per Observation')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart5_path = output_dir / "05_promotion_vs_non_promotion.png"
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    generated_files.append(str(chart5_path))

    # -------------------------------------------------------------
    # Chart 6: Weekday vs Weekend Demand
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    if 'weekend' in df.columns and 'daily_units_sold' in df.columns:
        df['weekend_label'] = df['weekend'].map({0: 'Weekday', 1: 'Weekend'})
        ax = sns.barplot(data=df, x='weekend_label', y='daily_units_sold', hue='weekend_label', errorbar=None, palette='Oranges', legend=False)
        plt.title('6. Average Demand: Weekday vs Weekend')
        plt.xlabel('Day Type')
        plt.ylabel('Average Daily Units Sold')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart6_path = output_dir / "06_weekday_vs_weekend_demand.png"
    plt.savefig(chart6_path, dpi=300)
    plt.close()
    generated_files.append(str(chart6_path))

    # -------------------------------------------------------------
    # Chart 7: Product Demand Volatility (Coefficient of Variation)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    if 'product_id' in df.columns and 'daily_units_sold' in df.columns:
        volatility = df.groupby('product_id')['daily_units_sold'].agg(
            mean_demand='mean',
            std_demand='std'
        ).reset_index()
        volatility['cv'] = (volatility['std_demand'] / volatility['mean_demand']).fillna(0)
        
        # Merge product category for coloring
        if 'category' in df.columns:
            prod_cats = df[['product_id', 'category']].drop_duplicates()
            volatility = pd.merge(volatility, prod_cats, on='product_id', how='left')
            sns.barplot(data=volatility.sort_values('cv', ascending=False), x='product_id', y='cv', hue='category', dodge=False)
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.barplot(data=volatility.sort_values('cv', ascending=False), x='product_id', y='cv', color='crimson')
            
        plt.title('7. Product Demand Volatility (Coefficient of Variation: Std/Mean)')
        plt.xlabel('Product ID')
        plt.ylabel('Coefficient of Variation (CV)')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    chart7_path = output_dir / "07_product_demand_volatility.png"
    plt.savefig(chart7_path, dpi=300)
    plt.close()
    generated_files.append(str(chart7_path))

    # -------------------------------------------------------------
    # Chart 8: Stock-out Frequency by Store
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    if 'closing' in df.columns and 'store_id' in df.columns:
        df['is_stockout'] = (df['closing'] == 0).astype(int)
        store_so = df.groupby('store_id')['is_stockout'].mean().reset_index()
        store_so['stockout_pct'] = store_so['is_stockout'] * 100
        ax = sns.barplot(data=store_so, x='store_id', y='stockout_pct', hue='store_id', palette='Reds_r', legend=False)
        plt.title('8. Stock-out Frequency (%) by Store')
        plt.xlabel('Store ID')
        plt.ylabel('Stock-out Rate (% of observations)')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart8_path = output_dir / "08_stockout_frequency_by_store.png"
    plt.savefig(chart8_path, dpi=300)
    plt.close()
    generated_files.append(str(chart8_path))

    # -------------------------------------------------------------
    # Chart 9: Stock-out Frequency by Category
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    if 'closing' in df.columns and 'category' in df.columns:
        if 'is_stockout' not in df.columns:
            df['is_stockout'] = (df['closing'] == 0).astype(int)
        cat_so = df.groupby('category')['is_stockout'].mean().reset_index()
        cat_so['stockout_pct'] = cat_so['is_stockout'] * 100
        cat_so = cat_so.sort_values('stockout_pct', ascending=False)
        ax = sns.barplot(data=cat_so, x='category', y='stockout_pct', hue='category', palette='YlOrRd_r', legend=False)
        plt.title('9. Stock-out Frequency (%) by Category')
        plt.xlabel('Category')
        plt.ylabel('Stock-out Rate (% of observations)')
        plt.xticks(rotation=30, ha='right')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart9_path = output_dir / "09_stockout_frequency_by_category.png"
    plt.savefig(chart9_path, dpi=300)
    plt.close()
    generated_files.append(str(chart9_path))

    # -------------------------------------------------------------
    # Chart 10: Store x Category Demand Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    if 'store_id' in df.columns and 'category' in df.columns and 'daily_units_sold' in df.columns:
        pivot_df = df.pivot_table(index='store_id', columns='category', values='daily_units_sold', aggfunc='sum', fill_value=0)
        sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Total Units Sold'})
        plt.title('10. Store x Product Category Total Demand Heatmap')
        plt.xlabel('Product Category')
        plt.ylabel('Store ID')
    plt.tight_layout()
    chart10_path = output_dir / "10_store_category_heatmap.png"
    plt.savefig(chart10_path, dpi=300)
    plt.close()
    generated_files.append(str(chart10_path))

    print(f"[OK] Generated {len(generated_files)} EDA figures in {output_dir}")
    return generated_files
