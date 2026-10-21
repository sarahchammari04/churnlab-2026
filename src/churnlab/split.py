"""Reproducible stratified 60/20/20 split on customers."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churnlab.data import ID_COLUMN, TARGET

SPLIT_SEED = 42


def make_splits(df: pd.DataFrame, seed: int = SPLIT_SEED):
    """Return (train, valid, test) DataFrames, stratified on the target."""
    train_df, remaining_df = train_test_split(
        df, test_size=0.4, stratify=df[TARGET], random_state=seed
    )
    valid_df, test_df = train_test_split(
        remaining_df, test_size=0.5, stratify=remaining_df[TARGET], random_state=seed
    )
    return train_df, valid_df, test_df


def allocation_table(train_df, valid_df, test_df) -> pd.DataFrame:
    """One row per customer with the split it belongs to."""
    parts = [
        pd.DataFrame({ID_COLUMN: d[ID_COLUMN].values, "split": name})
        for name, d in [("train", train_df), ("valid", valid_df), ("test", test_df)]
    ]
    return pd.concat(parts, ignore_index=True).sort_values(ID_COLUMN, ignore_index=True)


def check_disjoint(train_df, valid_df, test_df) -> dict:
    """Count customer IDs shared between splits (all must be 0)."""
    a, b, c = (set(d[ID_COLUMN]) for d in (train_df, valid_df, test_df))
    return {
        "train_valid": len(a & b),
        "train_test": len(a & c),
        "valid_test": len(b & c),
    }
