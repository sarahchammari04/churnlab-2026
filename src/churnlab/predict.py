"""Reusable inference boundary: raw rows in, one score/prediction row per customer out.

Importing this module never trains, loads a model or contacts MLflow.
"""

import pandas as pd

from churnlab.data import ID_COLUMN, validate_features
from churnlab.evaluate import THRESHOLD, positive_scores


def score_batch(pipeline, raw: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Validate raw inputs, then return customerID (if present), churn score and prediction.

    Extra columns such as `Churn` or `customerID` are ignored by the model:
    only the six validated features reach the pipeline.
    """
    X = validate_features(raw)  # KeyError if a required column is missing
    scores = positive_scores(pipeline, X)  # class-1 column found via classes_
    ids = raw[ID_COLUMN].to_numpy() if ID_COLUMN in raw.columns else range(len(raw))
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            "churn_score": scores,
            "churn_prediction": (scores >= threshold).astype(int),
        }
    )
