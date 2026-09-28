"""
eda.py
-------
Generates the core exploratory data analysis charts (Matplotlib/Seaborn)
and saves them to visualizations/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


def plot_revenue_by_category(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    sns.barplot(x=rev.values, y=rev.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Total Revenue")
    ax.set_ylabel("Category")
    ax.set_title("Total Revenue by Category")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "revenue_by_category.png"), dpi=120)
    plt.close(fig)


def plot_monthly_revenue_trend(df, out_dir):
    monthly = df.set_index("date").resample("ME")["revenue"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(ax=ax, marker="o", color="#DD8452")
    ax.set_ylabel("Revenue")
    ax.set_title("Monthly Revenue Trend")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "monthly_revenue_trend.png"), dpi=120)
    plt.close(fig)


def plot_price_vs_quantity(df, out_dir):
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=sample, x="unit_price", y="quantity_sold", hue="category",
        alpha=0.4, ax=ax, s=18
    )
    ax.set_title("Unit Price vs. Quantity Sold (sampled)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "price_vs_quantity.png"), dpi=120)
    plt.close(fig)


def plot_discount_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["discount_percentage"], bins=40, ax=ax, color="#55A868")
    ax.set_title("Distribution of Discount Percentage")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "discount_distribution.png"), dpi=120)
    plt.close(fig)


def plot_revenue_by_region(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    sns.barplot(x=rev.index, y=rev.values, ax=ax, color="#C44E52")
    ax.set_ylabel("Total Revenue")
    ax.set_title("Total Revenue by Region")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "revenue_by_region.png"), dpi=120)
    plt.close(fig)


def plot_segment_performance(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    seg = df.groupby("customer_segment")["revenue"].mean().sort_values(ascending=False)
    sns.barplot(x=seg.index, y=seg.values, ax=ax, color="#8172B2")
    ax.set_ylabel("Average Revenue per Transaction")
    ax.set_title("Average Revenue per Transaction by Customer Segment")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "segment_avg_revenue.png"), dpi=120)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plot_revenue_by_category(df, out_dir)
    plot_monthly_revenue_trend(df, out_dir)
    plot_price_vs_quantity(df, out_dir)
    plot_discount_distribution(df, out_dir)
    plot_revenue_by_region(df, out_dir)
    plot_segment_performance(df, out_dir)
    print(f"Saved 6 EDA charts -> {out_dir}")


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, VIZ_DIR

    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
    run_eda(df, VIZ_DIR)
