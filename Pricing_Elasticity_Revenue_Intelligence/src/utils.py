"""Shared path/IO helpers for the pricing elasticity project."""

import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "pricing_transactions.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "pricing_transactions_clean.csv")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")


def ensure_dirs():
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
