"""Data helpers: source location and integrity check for the Telco snapshot."""

import hashlib
from pathlib import Path

RAW_CSV = Path("data/raw/telco.csv")
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
TARGET = "Churn"
FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
