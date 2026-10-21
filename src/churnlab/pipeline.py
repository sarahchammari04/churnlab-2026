"""Construction of UNFITTED pipelines. Nothing here reads data or trains."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlab.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Documented preprocessing policy (also written into the artifact metadata).
PREPROCESSING = {
    "numeric": {
        "columns": NUMERIC_FEATURES,
        "imputer": "median (learned on training data only)",
        "scaler": "StandardScaler (mean/std learned on training data only)",
    },
    "categorical": {
        "columns": CATEGORICAL_FEATURES,
        "imputer": "most_frequent (learned on training data only)",
        "encoder": "OneHotEncoder(handle_unknown='ignore'): an unseen category "
        "is encoded as all zeros for that column, it does not raise",
    },
}


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(classifier) -> Pipeline:
    """Fresh preprocessing + the given (unfitted) classifier."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
