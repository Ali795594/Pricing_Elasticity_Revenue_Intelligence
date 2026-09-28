"""
statistical_analysis.py
------------------------
Uses statsmodels OLS to quantify RELATIONSHIPS (not causal effects) between
price, discount, marketing spend and demand/revenue.

IMPORTANT: This is observational data. Coefficients describe statistical
association conditional on the included controls, not a proven causal
effect. See README "Limitations" for a full discussion.
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_ols(df: pd.DataFrame, formula: str):
    """Fit an OLS model from a patsy formula string and return the result."""
    model = smf.ols(formula=formula, data=df)
    result = model.fit()
    return result


def summarize_result(result, label: str) -> dict:
    conf_int = result.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]
    summary_df = pd.DataFrame(
        {
            "coefficient": result.params,
            "std_err": result.bse,
            "t_value": result.tvalues,
            "p_value": result.pvalues,
        }
    ).join(conf_int)

    return {
        "label": label,
        "r_squared": result.rsquared,
        "adj_r_squared": result.rsquared_adj,
        "n_obs": int(result.nobs),
        "coefficients": summary_df,
    }


def run_all_analyses(df: pd.DataFrame) -> dict:
    """Run the core statistical relationships requested by the business."""
    results = {}

    # price -> quantity (log-log => coefficient is an elasticity estimate)
    m1 = fit_ols(df, "log_quantity ~ log_price + discount_percentage + C(category) + C(region)")
    results["price_to_quantity"] = summarize_result(m1, "log(price) -> log(quantity)")

    # discount -> quantity
    m2 = fit_ols(df, "quantity_sold ~ discount_percentage + unit_price + C(category)")
    results["discount_to_quantity"] = summarize_result(m2, "discount -> quantity")

    # price -> revenue
    m3 = fit_ols(df, "revenue ~ unit_price + discount_percentage + C(category) + C(region)")
    results["price_to_revenue"] = summarize_result(m3, "price -> revenue")

    # marketing spend -> revenue
    m4 = fit_ols(df, "revenue ~ marketing_spend + unit_price + discount_percentage")
    results["marketing_to_revenue"] = summarize_result(m4, "marketing spend -> revenue")

    return results, {"m1": m1, "m2": m2, "m3": m3, "m4": m4}


def print_results(results: dict):
    for key, res in results.items():
        print("=" * 70)
        print(f"{res['label']}  (n={res['n_obs']}, R2={res['r_squared']:.3f}, "
              f"Adj R2={res['adj_r_squared']:.3f})")
        print("=" * 70)
        print(res["coefficients"].round(4))
        print()


if __name__ == "__main__":
    from utils import BASE_DIR
    import os

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "pricing_features.csv"))
    results, _models = run_all_analyses(df)
    print_results(results)
