"""
elasticity_analysis.py
------------------------
Estimates price elasticity of demand using a log-log OLS specification:

    log(quantity) = beta0 + beta1 * log(price) + controls

beta1 is interpreted as the price elasticity of demand: the approximate
percentage change in quantity for a 1% change in price, holding the
included controls constant.

Elasticity is estimated overall and broken out by product, category,
region and customer segment so the business can see where demand is
most price-sensitive.

ASSUMPTIONS & LIMITATIONS (see README for the full discussion):
- This is observational, cross-sectional variation in price (largely
  driven by store/region price indices and small noise), not a
  randomized price experiment. Estimates should be read as
  "statistical association", not a guaranteed causal price response.
- Groups with too few observations are skipped to avoid unstable
  estimates (MIN_OBS threshold below).
- Extreme values were already winsorized during cleaning.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MIN_OBS = 80


def estimate_elasticity(df: pd.DataFrame) -> float | None:
    """Fit log(quantity) ~ log(price) + discount and return the price coefficient."""
    if len(df) < MIN_OBS:
        return None
    if df["unit_price"].std() < 1e-6:
        return None
    try:
        model = smf.ols("log_quantity ~ log_price + discount_percentage", data=df)
        result = model.fit()
        return float(result.params["log_price"])
    except Exception:
        return None


def elasticity_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_col, observed=True):
        elasticity = estimate_elasticity(group_df)
        rows.append(
            {
                group_col: group_value,
                "n_obs": len(group_df),
                "avg_price": group_df["unit_price"].mean(),
                "avg_quantity": group_df["quantity_sold"].mean(),
                "estimated_elasticity": elasticity,
            }
        )
    result = pd.DataFrame(rows).sort_values("estimated_elasticity")
    return result


def classify_elasticity(e: float | None) -> str:
    if e is None or np.isnan(e):
        return "insufficient data"
    if e < -1:
        return "elastic (demand sensitive to price)"
    if -1 <= e < 0:
        return "inelastic (demand less sensitive to price)"
    return "unexpected positive coefficient - interpret with caution"


def estimate_elasticity_fe(df: pd.DataFrame) -> float | None:
    """
    Same as estimate_elasticity, but adds product fixed effects
    (C(product_id)) as controls.

    Why this matters: a naive elasticity estimated across MULTIPLE products
    (e.g. "by category") mixes within-product price variation with
    between-product differences in base price and base demand. If
    higher-priced products in a category also happen to have unrelated
    demand differences, the simple regression can produce a confounded,
    even wrong-signed, coefficient (a version of Simpson's paradox).

    Adding product fixed effects isolates the within-product relationship
    between price and quantity, which is a much more defensible estimate
    of price sensitivity when analyzing a group with more than one product.
    """
    if len(df) < MIN_OBS or df["product_id"].nunique() < 2:
        return None
    if df["unit_price"].std() < 1e-6:
        return None
    try:
        model = smf.ols(
            "log_quantity ~ log_price + discount_percentage + C(product_id)",
            data=df,
        )
        result = model.fit()
        return float(result.params["log_price"])
    except Exception:
        return None


def elasticity_by_group_fe(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Group-level elasticity controlling for product fixed effects."""
    rows = []
    for group_value, group_df in df.groupby(group_col, observed=True):
        elasticity = estimate_elasticity_fe(group_df)
        rows.append(
            {
                group_col: group_value,
                "n_obs": len(group_df),
                "n_products": group_df["product_id"].nunique(),
                "estimated_elasticity_fe": elasticity,
            }
        )
    return pd.DataFrame(rows).sort_values("estimated_elasticity_fe")


def full_elasticity_report(df: pd.DataFrame) -> dict:
    overall = estimate_elasticity(df)
    report = {
        "overall_elasticity": overall,
        "overall_interpretation": classify_elasticity(overall),
        # Naive (no product fixed effects) - shown to illustrate confounding
        "by_category": elasticity_by_group(df, "category"),
        "by_region": elasticity_by_group(df, "region"),
        "by_segment": elasticity_by_group(df, "customer_segment"),
        "by_product": elasticity_by_group(df, "product_id"),
        # Product-fixed-effects versions - the more defensible estimates
        "by_category_fe": elasticity_by_group_fe(df, "category"),
        "by_region_fe": elasticity_by_group_fe(df, "region"),
        "by_segment_fe": elasticity_by_group_fe(df, "customer_segment"),
    }
    return report


if __name__ == "__main__":
    from utils import BASE_DIR
    import os

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "pricing_features.csv"))
    report = full_elasticity_report(df)

    print(f"Overall elasticity: {report['overall_elasticity']:.3f} "
          f"-> {report['overall_interpretation']}")
    print("\nElasticity by category:")
    print(report["by_category"].round(3).to_string(index=False))
    print("\nElasticity by region:")
    print(report["by_region"].round(3).to_string(index=False))
    print("\nElasticity by customer segment:")
    print(report["by_segment"].round(3).to_string(index=False))

    print("\n--- With product fixed effects (controls for confounding) ---")
    print("\nElasticity by category (FE):")
    print(report["by_category_fe"].round(3).to_string(index=False))
    print("\nElasticity by region (FE):")
    print(report["by_region_fe"].round(3).to_string(index=False))
    print("\nElasticity by customer segment (FE):")
    print(report["by_segment_fe"].round(3).to_string(index=False))

    out_path = os.path.join(BASE_DIR, "data", "processed", "elasticity_by_category.csv")
    report["by_category"].to_csv(out_path, index=False)
    print(f"\nSaved category elasticity table -> {out_path}")
