"""
data_validation.py
-------------------
Lightweight data-quality checks run BEFORE cleaning, so that issues in the
raw data are visible and documented rather than silently fixed.
"""

import pandas as pd


def validate(df: pd.DataFrame) -> dict:
    """Return a dictionary summarizing data-quality issues found in df."""
    report = {}

    report["n_rows"] = len(df)
    report["n_duplicate_rows"] = int(df.duplicated().sum())
    report["n_duplicate_transaction_ids"] = int(
        df["transaction_id"].duplicated().sum()
    )
    report["missing_values_by_column"] = df.isna().sum().to_dict()
    report["negative_unit_price_count"] = int((df["unit_price"] < 0).sum())
    report["negative_quantity_count"] = int((df["quantity_sold"] < 0).sum())
    report["discount_out_of_range_count"] = int(
        ((df["discount_percentage"] < 0) | (df["discount_percentage"] > 1)).sum()
    )

    q_low, q_high = df["quantity_sold"].quantile([0.001, 0.999])
    report["quantity_extreme_outliers_iqr_like"] = int(
        ((df["quantity_sold"] < q_low) | (df["quantity_sold"] > q_high)).sum()
    )

    report["unique_products"] = df["product_id"].nunique()
    report["unique_stores"] = df["store_id"].nunique()
    report["unique_regions"] = df["region"].nunique()
    report["date_min"] = str(df["date"].min())
    report["date_max"] = str(df["date"].max())

    return report


def print_report(report: dict):
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    for key, value in report.items():
        if key == "missing_values_by_column":
            print("\nMissing values by column:")
            for col, n in value.items():
                if n > 0:
                    print(f"  - {col}: {n}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    from utils import RAW_DATA_PATH

    df = pd.read_csv(RAW_DATA_PATH)
    report = validate(df)
    print_report(report)
