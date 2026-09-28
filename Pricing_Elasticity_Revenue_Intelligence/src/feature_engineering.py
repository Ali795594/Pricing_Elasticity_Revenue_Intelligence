"""
feature_engineering.py
-----------------------
Creates model-ready features from the cleaned pricing dataset.

Features created:
- log_price, log_quantity, log_competitor_price : for log-log elasticity models
- price_gap_pct        : (own price - competitor price) / competitor price
- discount_bucket      : categorical bucket of discount depth
- relative_price_index : product price relative to its own category average
- month / is_weekend    : already present from cleaning, kept for modeling
- one-hot encoded category, region, customer_segment for regression
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avoid log(0) issues
    df = df[(df["unit_price"] > 0) & (df["quantity_sold"] >= 0)].copy()

    df["log_price"] = np.log(df["unit_price"])
    df["log_quantity"] = np.log1p(df["quantity_sold"])
    df["log_competitor_price"] = np.log(df["competitor_price"].clip(lower=0.01))

    df["price_gap_pct"] = (
        (df["unit_price"] - df["competitor_price"]) / df["competitor_price"]
    )

    df["discount_bucket"] = pd.cut(
        df["discount_percentage"],
        bins=[-0.01, 0.0, 0.10, 0.25, 1.0],
        labels=["none", "low", "medium", "high"],
    )

    category_avg_price = df.groupby("category")["unit_price"].transform("mean")
    df["relative_price_index"] = df["unit_price"] / category_avg_price

    df["log_marketing_spend"] = np.log1p(df["marketing_spend"])

    return df


def build_model_matrix(df: pd.DataFrame, target: str = "quantity_sold"):
    """Return (X, y) with one-hot encoded categoricals for regression."""
    feature_cols_numeric = [
        "unit_price",
        "discount_percentage",
        "competitor_price",
        "marketing_spend",
        "price_gap_pct",
        "relative_price_index",
        "is_weekend",
    ]
    categorical_cols = ["category", "region", "customer_segment"]

    X = df[feature_cols_numeric + categorical_cols].copy()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    y = df[target].copy()
    return X, y


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, BASE_DIR
    import os

    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
    featured = add_features(df)
    out_path = os.path.join(BASE_DIR, "data", "processed", "pricing_features.csv")
    featured.to_csv(out_path, index=False)
    print(f"Saved feature-engineered dataset -> {out_path}")
    print(featured[["log_price", "log_quantity", "price_gap_pct", "discount_bucket"]].head())
