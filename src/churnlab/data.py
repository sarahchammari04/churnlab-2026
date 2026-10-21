"""Loading, integrity check and explicit parsing of the raw Telco snapshot.

Boundary rules (applied identically at training and at inference):
- `TotalCharges` that is empty or whitespace-only becomes a missing value (NaN).
- Any other non-numeric text in a numeric column raises an error: it is never
  silently converted to a missing value.
- The original CSV is only read, never written.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path("data/raw/telco.csv")
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

ID_COLUMN = "customerID"
TARGET = "Churn"
POSITIVE_LABEL = "Yes"  # Churn = Yes is the positive class (encoded as 1)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path = RAW_CSV, expected_sha256: str = EXPECTED_SHA256) -> pd.DataFrame:
    """Read the raw CSV after checking its hash. All columns stay as read."""
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise ValueError(f"Source hash mismatch for {path}: {actual}")
    return pd.read_csv(path)


def parse_numeric(series: pd.Series) -> pd.Series:
    """Blank/whitespace-only -> NaN; any other non-numeric text -> error."""
    as_text = series.astype("string")
    blank = as_text.str.strip().eq("").fillna(False)
    cleaned = as_text.mask(blank)  # blanks become <NA>, everything else unchanged
    # errors="raise": malformed non-blank text must fail loudly.
    return pd.to_numeric(cleaned, errors="raise").astype("float64")


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Check the inference schema and return the six features with documented types.

    Input: a DataFrame containing at least the six FEATURES columns.
    Output: numeric columns as float64 (NaN allowed), categorical columns as object.
    A missing column raises a KeyError instead of producing a silent prediction.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    out = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        out[col] = parse_numeric(out[col])
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("object").where(out[col].notna(), np.nan)
    return out


def encode_target(series: pd.Series) -> pd.Series:
    """Map Churn Yes/No to 1/0 and refuse any other label."""
    mapped = series.map({"No": 0, "Yes": 1})
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unexpected target labels: {bad}")
    return mapped.astype("int64")
