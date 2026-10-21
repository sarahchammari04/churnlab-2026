import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churnlab.data import FEATURES, TARGET, encode_target, parse_numeric, validate_features
from churnlab.pipeline import build_pipeline


def _toy():
    return pd.DataFrame({
        "tenure": ["1", "20", "40", "5"],
        "MonthlyCharges": ["30.5", "80", "60", "99.9"],
        "TotalCharges": [" ", "1600", "2400", "500"],
        "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
        "InternetService": ["DSL", "Fiber optic", "No", "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)", "Electronic check"],
        TARGET: ["Yes", "No", "No", "Yes"],
    })


def test_blank_becomes_nan_but_text_raises():
    assert np.isnan(parse_numeric(pd.Series([" "]))[0])
    with pytest.raises(ValueError):
        parse_numeric(pd.Series(["abc"]))


def test_missing_column_raises():
    with pytest.raises(KeyError):
        validate_features(_toy().drop(columns=["Contract"]))


def test_target_not_in_features():
    assert TARGET not in FEATURES and "customerID" not in FEATURES


def test_builder_returns_unfitted_pipeline_and_fits_end_to_end():
    pipe = build_pipeline(LogisticRegression())
    assert not hasattr(pipe.named_steps["classifier"], "coef_")
    df = _toy()
    pipe.fit(validate_features(df), encode_target(df[TARGET]))
    unseen = validate_features(df).head(1).assign(InternetService="Satellite")
    assert pipe.predict_proba(unseen).shape == (1, 2)
