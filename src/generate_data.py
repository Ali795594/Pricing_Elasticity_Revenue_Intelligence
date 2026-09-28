"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC transactional pricing dataset for the
Pricing Elasticity & Revenue Intelligence System.

Run:
    python src/generate_data.py

Output:
    data/raw/pricing_transactions.csv  (~55,000 rows)

The data is generated with a fixed random seed so it is reproducible,
but it deliberately contains noise, missing values, outliers and
seasonality so it behaves like a real-world dataset rather than a
toy, perfectly-clean one.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
N_TARGET_ROWS = 55_000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 12, 31)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pricing_transactions.csv")


def build_catalog(rng):
    """Create a product catalog with categories and base prices/elasticities."""
    categories = {
        "Electronics": {"base_price": (80, 900), "elasticity": (-2.2, -1.4)},
        "Home & Kitchen": {"base_price": (15, 220), "elasticity": (-1.6, -0.9)},
        "Apparel": {"base_price": (10, 150), "elasticity": (-2.0, -1.1)},
        "Beauty & Personal Care": {"base_price": (5, 90), "elasticity": (-1.3, -0.6)},
        "Sports & Outdoors": {"base_price": (12, 300), "elasticity": (-1.8, -1.0)},
        "Grocery": {"base_price": (2, 40), "elasticity": (-0.9, -0.3)},
    }

    products = []
    product_id = 1
    for category, cfg in categories.items():
        n_products = rng.integers(6, 10)
        for _ in range(n_products):
            base_price = round(rng.uniform(*cfg["base_price"]), 2)
            elasticity = round(rng.uniform(*cfg["elasticity"]), 2)
            products.append(
                {
                    "product_id": f"P{product_id:04d}",
                    "product_name": f"{category.split()[0]} Item {product_id}",
                    "category": category,
                    "base_price": base_price,
                    "base_elasticity": elasticity,
                    "base_demand": rng.uniform(15, 120),
                }
            )
            product_id += 1
    return pd.DataFrame(products)


def build_stores(rng):
    regions = ["North", "South", "East", "West", "Central"]
    stores = []
    store_id = 1
    for region in regions:
        n_stores = rng.integers(3, 6)
        for _ in range(n_stores):
            stores.append(
                {
                    "store_id": f"S{store_id:03d}",
                    "region": region,
                    "region_price_index": rng.uniform(0.9, 1.15),
                }
            )
            store_id += 1
    return pd.DataFrame(stores)


def seasonal_multiplier(date):
    """Simple yearly seasonality + a Nov/Dec promo spike."""
    day_of_year = date.timetuple().tm_yday
    season = 1 + 0.18 * np.sin(2 * np.pi * (day_of_year / 365.0))
    holiday_boost = 1.35 if date.month in (11, 12) else 1.0
    weekend_boost = 1.08 if date.weekday() >= 5 else 1.0
    return season * holiday_boost * weekend_boost


def generate(n_rows=N_TARGET_ROWS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    catalog = build_catalog(rng)
    stores = build_stores(rng)
    segments = ["Budget", "Regular", "Premium", "VIP"]
    segment_probs = [0.35, 0.35, 0.20, 0.10]
    segment_price_sensitivity = {"Budget": 1.35, "Regular": 1.0, "Premium": 0.75, "VIP": 0.55}

    date_range_days = (END_DATE - START_DATE).days

    # Sample which product/store/date each transaction belongs to
    product_idx = rng.integers(0, len(catalog), size=n_rows)
    store_idx = rng.integers(0, len(stores), size=n_rows)
    day_offsets = rng.integers(0, date_range_days + 1, size=n_rows)

    rows = []
    for i in range(n_rows):
        prod = catalog.iloc[product_idx[i]]
        store = stores.iloc[store_idx[i]]
        date = START_DATE + timedelta(days=int(day_offsets[i]))
        segment = rng.choice(segments, p=segment_probs)

        season_mult = seasonal_multiplier(date)

        # Discounting behaviour: more common in promo months, some randomness
        promo_month = date.month in (6, 7, 11, 12)
        discount_base = rng.beta(2, 8) * (1.6 if promo_month else 1.0)
        discount_percentage = float(np.clip(discount_base, 0, 0.7))

        # Price varies by store region index and small day-to-day noise
        unit_price = (
            prod["base_price"]
            * store["region_price_index"]
            * rng.normal(1.0, 0.04)
        )
        unit_price = max(unit_price, 0.5)

        # Competitor price correlated with own price plus independent noise
        competitor_price = unit_price * rng.normal(1.0, 0.08)

        # Marketing spend: monthly campaigns, noisy
        marketing_spend = max(0.0, rng.gamma(2.0, 40) * (1.5 if promo_month else 1.0))

        # Demand model: base demand adjusted by price elasticity, discount
        # uplift, seasonality, segment sensitivity and marketing effect.
        elasticity = prod["base_elasticity"] * segment_price_sensitivity[segment]
        price_ratio = unit_price / prod["base_price"]
        price_effect = price_ratio ** elasticity  # elasticity is negative
        discount_effect = 1 + 1.4 * discount_percentage
        marketing_effect = 1 + 0.0015 * marketing_spend

        expected_qty = (
            prod["base_demand"]
            * price_effect
            * discount_effect
            * marketing_effect
            * season_mult
        )
        # Poisson-like noisy demand realization
        quantity_sold = rng.poisson(lam=max(expected_qty, 0.1))

        revenue = round(quantity_sold * unit_price * (1 - discount_percentage), 2)

        rows.append(
            {
                "transaction_id": f"T{i+1:07d}",
                "date": date.strftime("%Y-%m-%d"),
                "product_id": prod["product_id"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "region": store["region"],
                "store_id": store["store_id"],
                "quantity_sold": int(quantity_sold),
                "unit_price": round(unit_price, 2),
                "discount_percentage": round(discount_percentage, 4),
                "revenue": revenue,
                "customer_segment": segment,
                "competitor_price": round(competitor_price, 2),
                "marketing_spend": round(marketing_spend, 2),
            }
        )

    df = pd.DataFrame(rows)

    # ---- Inject realistic data-quality issues ----

    # 1) Missing values (MCAR-ish) in a few columns
    for col, frac in [
        ("discount_percentage", 0.02),
        ("competitor_price", 0.03),
        ("marketing_spend", 0.025),
        ("unit_price", 0.005),
    ]:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan

    # 2) A handful of duplicate transactions (data entry duplication)
    dup_sample = df.sample(frac=0.004, random_state=seed)
    df = pd.concat([df, dup_sample], ignore_index=True)

    # 3) Outliers: a small percentage of rows get an extreme quantity spike
    #    (e.g. bulk/wholesale orders or data entry errors)
    outlier_mask = rng.random(len(df)) < 0.006
    spike_factor = rng.uniform(6, 15, size=outlier_mask.sum())
    df.loc[outlier_mask, "quantity_sold"] = (
        df.loc[outlier_mask, "quantity_sold"] * spike_factor
    ).round().astype(int)
    df.loc[outlier_mask, "revenue"] = (
        df.loc[outlier_mask, "quantity_sold"]
        * df.loc[outlier_mask, "unit_price"].fillna(df["unit_price"].median())
        * (1 - df.loc[outlier_mask, "discount_percentage"].fillna(0))
    ).round(2)

    # 4) A few negative / impossible values to simulate entry errors
    error_mask = rng.random(len(df)) < 0.002
    df.loc[error_mask, "unit_price"] = -df.loc[error_mask, "unit_price"].abs()

    # Shuffle rows so duplicates/outliers aren't clustered at the end
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {OUTPUT_PATH}")
    print(df.head())
    print("\nMissing value counts:\n", df.isna().sum())


if __name__ == "__main__":
    main()
