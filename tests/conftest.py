"""Shared test data: small invented customers, no Telco CSV, no network, no MLflow."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churnlab.data import validate_features
from churnlab.pipeline import build_pipeline

_COLUMNS = [
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]

# 16 invented customers in raw text form (as read from a CSV with dtype=str),
# followed by the label (1 = Churn Yes). Both classes are present.
# fmt: off
_ROWS = [
    ("TEST-00", "1",  "70.5",  "70.5",   "Month-to-month", "Fiber optic", "Electronic check",          1),
    ("TEST-01", "2",  "85",    "170",    "Month-to-month", "Fiber optic", "Electronic check",          1),
    ("TEST-02", "5",  "95.2",  "476",    "Month-to-month", "Fiber optic", "Mailed check",              1),
    ("TEST-03", "8",  "60",    "480",    "Month-to-month", "DSL",         "Electronic check",          0),
    ("TEST-04", "12", "99.9",  "1198.8", "Month-to-month", "Fiber optic", "Electronic check",          1),
    ("TEST-05", "3",  "75",    "225",    "Month-to-month", "DSL",         "Mailed check",              1),
    ("TEST-06", "24", "45",    "1080",   "One year",       "DSL",         "Bank transfer (automatic)", 0),
    ("TEST-07", "36", "55.5",  "1998",   "One year",       "No",          "Credit card (automatic)",   0),
    ("TEST-08", "48", "25",    "1200",   "Two year",       "No",          "Bank transfer (automatic)", 0),
    ("TEST-09", "60", "20.1",  "1206",   "Two year",       "No",          "Credit card (automatic)",   0),
    ("TEST-10", "70", "105",   "7350",   "Two year",       "Fiber optic", "Credit card (automatic)",   1),
    ("TEST-11", "15", "80",    "1200",   "One year",       "DSL",         "Mailed check",              0),
    ("TEST-12", "30", "50",    "1500",   "One year",       "DSL",         "Bank transfer (automatic)", 0),
    ("TEST-13", "45", "65",    "2925",   "Two year",       "No",          "Mailed check",              0),
    ("TEST-14", "4",  "90",    "360",    "Month-to-month", "Fiber optic", "Electronic check",          1),
    ("TEST-15", "66", "30",    "1980",   "Two year",       "No",          "Credit card (automatic)",   0),
]
# fmt: on

CLASSIFIERS = {
    "logistic": lambda: LogisticRegression(max_iter=1000),
    "forest": lambda: RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0),
}


@pytest.fixture
def raw_batch() -> pd.DataFrame:
    """Raw inference batch: six features + customerID, all as text (no label)."""
    return pd.DataFrame([row[:-1] for row in _ROWS], columns=_COLUMNS)


@pytest.fixture
def sample_features(raw_batch) -> pd.DataFrame:
    """The six features after the documented parsing rules."""
    return validate_features(raw_batch)


@pytest.fixture
def sample_target() -> pd.Series:
    """Encoded label, kept separate from the features."""
    return pd.Series([row[-1] for row in _ROWS], name="Churn")


@pytest.fixture(params=sorted(CLASSIFIERS))
def fitted_pipeline(request, sample_features, sample_target):
    """Real project pipeline (build_pipeline), fitted on the synthetic rows only.

    Parametrized: every test using it runs for both model families of TP02.
    """
    pipe = build_pipeline(CLASSIFIERS[request.param]())
    pipe.fit(sample_features, sample_target)
    return pipe
