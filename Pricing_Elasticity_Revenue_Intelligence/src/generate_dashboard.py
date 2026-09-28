"""
generate_dashboard.py
----------------------
Builds four dashboards (Matplotlib/Seaborn) as PNG files plus one
self-contained HTML report that embeds them.

    1. Executive Revenue Overview
    2. Pricing Analysis
    3. Product Intelligence
    4. Regional Analysis

Run:
    python src/generate_dashboard.py
    python src/generate_dashboard.py --region North --year 2023        # filtered view
    python src/generate_dashboard.py --category Electronics --segment VIP

Filters (all optional): --year, --region, --category, --segment
Output: dashboard/  (or dashboard_filtered/ when any filter is used).
Open pricing_dashboard.html in any web browser.
"""

import argparse
import base64
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


def apply_filters(df, year=None, region=None, category=None, segment=None):
    if year is not None:
        df = df[df["date"].dt.year == year]
    if region:
        df = df[df["region"] == region]
    if category:
        df = df[df["category"] == category]
    if segment:
        df = df[df["customer_segment"] == segment]
    if df.empty:
        raise ValueError("No rows left after applying filters.")
    return df


def filter_label(year, region, category, segment):
    parts = []
    if year is not None:
        parts.append(f"year={year}")
    if region:
        parts.append(f"region={region}")
    if category:
        parts.append(f"category={category}")
    if segment:
        parts.append(f"segment={segment}")
    return " | ".join(parts) if parts else "All data"


def _kpi_panel(ax, df):
    ax.axis("off")
    kpis = [
        ("Total Revenue", f"${df['revenue'].sum():,.0f}"),
        ("Units Sold", f"{df['quantity_sold'].sum():,.0f}"),
        ("Avg Discount", f"{df['discount_percentage'].mean():.1%}"),
        ("Avg Revenue / Txn", f"${df['revenue'].mean():,.2f}"),
    ]
    for i, (name, value) in enumerate(kpis):
        x = 0.12 + i * 0.25
        ax.text(x, 0.62, value, ha="center", fontsize=22, fontweight="bold", color="#1f4e79")
        ax.text(x, 0.25, name, ha="center", fontsize=11, color="#555555")


def dashboard_executive(df, label, out_path):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 2, 2])
    _kpi_panel(fig.add_subplot(gs[0, :]), df)

    ax = fig.add_subplot(gs[1, :])
    monthly = df.set_index("date").resample("ME")["revenue"].sum()
    ax.plot(monthly.index, monthly.values, marker="o", color="#DD8452")
    ax.set_title("Monthly Revenue")
    ax.set_ylabel("Revenue")

    ax = fig.add_subplot(gs[2, 0])
    rev_cat = df.groupby("category")["revenue"].sum().sort_values()
    ax.barh(rev_cat.index, rev_cat.values, color="#4C72B0")
    ax.set_title("Revenue by Category")

    ax = fig.add_subplot(gs[2, 1])
    rev_q = df.groupby(df["date"].dt.to_period("Q").astype(str))["revenue"].sum()
    ax.bar(rev_q.index, rev_q.values, color="#55A868")
    ax.set_title("Revenue by Quarter")
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Dashboard 1 - Executive Revenue Overview   [{label}]", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def dashboard_pricing(df, label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sample = df.sample(min(6000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="unit_price", y="quantity_sold", hue="category",
                    alpha=0.4, s=16, ax=axes[0, 0])
    axes[0, 0].set_title("Unit Price vs Quantity Sold (sample)")
    axes[0, 0].set_xscale("log")

    disc = df.set_index("date").resample("ME")["discount_percentage"].mean()
    axes[0, 1].plot(disc.index, disc.values, marker="o", color="#C44E52")
    axes[0, 1].set_title("Average Discount Over Time")
    axes[0, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    gap = (df.assign(gap=(df["unit_price"] - df["competitor_price"]) / df["competitor_price"])
             .groupby("category")["gap"].mean().sort_values())
    axes[1, 0].barh(gap.index, gap.values * 100, color="#8172B2")
    axes[1, 0].set_title("Avg Price Gap vs Competitor (%)")

    bucket = pd.cut(df["discount_percentage"], [-0.01, 0, 0.10, 0.25, 1.0],
                    labels=["None", "Low (<=10%)", "Medium (<=25%)", "High (>25%)"])
    qty = df.groupby(bucket, observed=True)["quantity_sold"].mean()
    axes[1, 1].bar(qty.index.astype(str), qty.values, color="#55A868")
    axes[1, 1].set_title("Avg Quantity Sold by Discount Bucket")

    fig.suptitle(f"Dashboard 2 - Pricing Analysis   [{label}]", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def dashboard_product(df, label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    top_rev = df.groupby("product_name")["revenue"].sum().nlargest(15).sort_values()
    axes[0, 0].barh(top_rev.index, top_rev.values, color="#4C72B0")
    axes[0, 0].set_title("Top 15 Products by Revenue")

    top_units = df.groupby("product_name")["quantity_sold"].sum().nlargest(15).sort_values()
    axes[0, 1].barh(top_units.index, top_units.values, color="#DD8452")
    axes[0, 1].set_title("Top 15 Products by Units Sold")

    prod = df.groupby(["product_name", "category"]).agg(
        avg_price=("unit_price", "mean"), avg_qty=("quantity_sold", "mean"),
        revenue=("revenue", "sum")).reset_index()
    sns.scatterplot(data=prod, x="avg_price", y="avg_qty", size="revenue", hue="category",
                    sizes=(20, 300), alpha=0.7, ax=axes[1, 0], legend=False)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Products: Avg Price vs Avg Quantity (bubble = revenue)")

    prod_disc = df.groupby("product_name").agg(
        avg_discount=("discount_percentage", "mean"), revenue=("revenue", "sum")).reset_index()
    sns.scatterplot(data=prod_disc, x="avg_discount", y="revenue", ax=axes[1, 1], color="#C44E52")
    axes[1, 1].set_title("Product Revenue vs Average Discount")

    fig.suptitle(f"Dashboard 3 - Product Intelligence   [{label}]", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def dashboard_regional(df, label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    axes[0, 0].bar(rev.index, rev.values, color="#4C72B0")
    axes[0, 0].set_title("Revenue by Region")

    price = df.groupby("region")["unit_price"].mean().reindex(rev.index)
    axes[0, 1].bar(price.index, price.values, color="#55A868")
    axes[0, 1].set_title("Average Unit Price by Region")

    disc = df.groupby("region")["discount_percentage"].mean().reindex(rev.index)
    axes[1, 0].bar(disc.index, disc.values * 100, color="#C44E52")
    axes[1, 0].set_title("Average Discount by Region (%)")

    pivot = df.pivot_table(index="region", columns="category", values="revenue", aggfunc="sum") / 1000
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title("Revenue: Region x Category ($ thousands)")

    fig.suptitle(f"Dashboard 4 - Regional Analysis   [{label}]", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build_html_report(png_paths, titles, out_path, label):
    """Single self-contained HTML page (images embedded as base64)."""
    sections = []
    for path, title in zip(png_paths, titles):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        sections.append(f"<h2>{title}</h2><img src='data:image/png;base64,{encoded}' "
                        f"style='max-width:100%;border:1px solid #ddd;'>")
    html = ("<html><head><meta charset='utf-8'><title>Pricing Dashboards</title>"
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:30px auto;}"
            "h1{color:#1f4e79}</style></head><body>"
            f"<h1>Pricing Elasticity & Revenue Intelligence</h1><p>Filters: {label}</p>"
            + "".join(sections) + "</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    from utils import CLEAN_DATA_PATH, BASE_DIR

    parser = argparse.ArgumentParser(description="Build pricing dashboards.")
    parser.add_argument("--year", type=int)
    parser.add_argument("--region")
    parser.add_argument("--category")
    parser.add_argument("--segment")
    args = parser.parse_args()

    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
    df = apply_filters(df, args.year, args.region, args.category, args.segment)
    label = filter_label(args.year, args.region, args.category, args.segment)

    filtered = label != "All data"
    out_dir = os.path.join(BASE_DIR, "dashboard_filtered" if filtered else "dashboard")
    os.makedirs(out_dir, exist_ok=True)

    builders = [
        ("dashboard_1_executive_revenue.png", "Dashboard 1 - Executive Revenue Overview", dashboard_executive),
        ("dashboard_2_pricing_analysis.png", "Dashboard 2 - Pricing Analysis", dashboard_pricing),
        ("dashboard_3_product_intelligence.png", "Dashboard 3 - Product Intelligence", dashboard_product),
        ("dashboard_4_regional_analysis.png", "Dashboard 4 - Regional Analysis", dashboard_regional),
    ]
    paths, titles = [], []
    for filename, title, fn in builders:
        path = os.path.join(out_dir, filename)
        fn(df, label, path)
        paths.append(path)
        titles.append(title)

    html_path = os.path.join(out_dir, "pricing_dashboard.html")
    build_html_report(paths, titles, html_path, label)
    print(f"Saved 4 dashboards + HTML report ({label}) -> {out_dir}")


if __name__ == "__main__":
    main()
