"""
data_cleaning.py
-----------------
Deterministic cleaning steps applied to the raw pricing dataset.

Design choices (documented deliberately, not hidden):
- Negative unit_price values are data-entry errors -> take absolute value.
- Rows with negative quantity_sold are dropped (cannot sell negative units).
- Exact duplicate rows are dropped.
- Missing discount_percentage is treated as "no discount recorded" -> 0.
- Missing competitor_price / marketing_spend are imputed with the
  category-level median (documented as an assumption, not "ground truth").
- Missing unit_price rows are dropped (price is central to this analysis,
  so imputing it would bias the elasticity estimates).
- Extreme quantity outliers (top 0.1%) are capped (winsorized) rather than
  dropped, since bulk orders are plausible but should not dominate a
  linear regression.
"""

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    n_dupes_removed = before - len(df)

    # Fix impossible negative prices (data entry error -> take abs value)
    df["unit_price"] = df["unit_price"].abs()

    # Drop rows with no recorded price (price is essential for this analysis)
    df = df.dropna(subset=["unit_price"])

    # Drop rows with negative or impossible quantities
    df = df[df["quantity_sold"] >= 0]

    # Missing discount -> assume no discount was applied
    df["discount_percentage"] = df["discount_percentage"].fillna(0.0)
    df["discount_percentage"] = df["discount_percentage"].clip(0, 1)

    # Impute competitor_price / marketing_spend with category-level median
    for col in ["competitor_price", "marketing_spend"]:
        df[col] = df.groupby("category")[col].transform(
            lambda s: s.fillna(s.median())
        )
        # fallback for any category-level all-NaN edge case
        df[col] = df[col].fillna(df[col].median())

    # Parse dates and derive calendar features used later
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.weekday >= 5

    # Winsorize extreme quantity outliers at the 0.1% / 99.9% level
    low, high = df["quantity_sold"].quantile([0.001, 0.999])
    df["quantity_sold_raw"] = df["quantity_sold"]
    df["quantity_sold"] = df["quantity_sold"].clip(lower=low, upper=high)

    # Recompute revenue consistently after cleaning
    df["revenue"] = (
        df["quantity_sold"] * df["unit_price"] * (1 - df["discount_percentage"])
    ).round(2)

    df = df.reset_index(drop=True)

    print(f"Removed {n_dupes_removed} duplicate rows.")
    print(f"Rows after cleaning: {len(df):,} (from {before:,})")

    return df


def build_sql_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with columns renamed/ordered to match sql/01_schema.sql."""
    sql_df = df.rename(columns={"date": "txn_date"})[
        [
            "transaction_id",
            "txn_date",
            "product_id",
            "product_name",
            "category",
            "region",
            "store_id",
            "quantity_sold",
            "unit_price",
            "discount_percentage",
            "revenue",
            "customer_segment",
            "competitor_price",
            "marketing_spend",
        ]
    ]
    return sql_df


if __name__ == "__main__":
    from utils import RAW_DATA_PATH, CLEAN_DATA_PATH, BASE_DIR, ensure_dirs
    import os

    ensure_dirs()
    raw = pd.read_csv(RAW_DATA_PATH)
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_DATA_PATH}")

    sql_export_path = os.path.join(BASE_DIR, "data", "processed", "pricing_transactions_sql.csv")
    build_sql_export(cleaned).to_csv(sql_export_path, index=False)
    print(f"Saved SQL-ready export -> {sql_export_path}")
