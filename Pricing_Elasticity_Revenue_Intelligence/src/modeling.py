"""
modeling.py
------------
Trains a Linear Regression model to predict quantity_sold from pricing,
discount, marketing and categorical features.

Includes:
- train/test split (time-aware: trains on earlier dates, tests on later ones,
  which is more realistic for a pricing/demand use case than a random split)
- a naive baseline (predict the training mean) for comparison
- MAE, RMSE, R2
- residual diagnostics
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import build_model_matrix


def time_based_split(df: pd.DataFrame, date_col="date", test_frac=0.2):
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ensure train/test one-hot columns match (test set may lack a category)."""
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return X_train, X_test


def evaluate(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2}


def train_and_evaluate(df: pd.DataFrame, target="quantity_sold"):
    train_df, test_df = time_based_split(df)

    X_train, y_train = build_model_matrix(train_df, target=target)
    X_test, y_test = build_model_matrix(test_df, target=target)
    X_train, X_test = align_columns(X_train, X_test)

    # Baseline: always predict the training-set mean
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    baseline_metrics = evaluate(y_test, baseline_pred, "Baseline (mean predictor)")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_metrics = evaluate(y_test, y_pred, "Linear Regression")

    residuals = y_test.values - y_pred

    coef_table = (
        pd.Series(model.coef_, index=X_train.columns)
        .sort_values(key=abs, ascending=False)
        .rename("coefficient")
        .to_frame()
    )

    return {
        "model": model,
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "residuals": residuals,
        "coefficients": coef_table,
        "feature_columns": list(X_train.columns),
    }


if __name__ == "__main__":
    from utils import BASE_DIR
    import os

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "pricing_features.csv"), parse_dates=["date"])
    results = train_and_evaluate(df)

    print("\nTop 10 coefficients by magnitude:")
    print(results["coefficients"].head(10))

    improvement = (
        (results["baseline_metrics"]["rmse"] - results["model_metrics"]["rmse"])
        / results["baseline_metrics"]["rmse"]
        * 100
    )
    print(f"\nModel reduces RMSE vs. baseline by {improvement:.1f}%")
