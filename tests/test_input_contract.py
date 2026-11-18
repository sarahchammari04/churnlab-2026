"""Input contract: required columns, permitted variations, excluded columns."""

import numpy as np
import pandas as pd
import pytest

from churnlab.data import FEATURES, validate_features
from churnlab.predict import score_batch


# Test A: a missing required feature must be rejected with a clear message.
@pytest.mark.parametrize("column", FEATURES)
def test_missing_required_feature_is_rejected(fitted_pipeline, raw_batch, column):
    broken = raw_batch.drop(columns=[column])

    with pytest.raises(KeyError, match=f"Missing required feature columns: .*{column}"):
        score_batch(fitted_pipeline, broken)


# Test B: blank numeric value + category never seen in training are accepted.
def test_missing_numeric_and_unknown_category_are_accepted(fitted_pipeline, raw_batch):
    seen = set(raw_batch["InternetService"])
    assert "Satellite" not in seen  # the category really is absent from training

    varied = raw_batch.copy()
    varied.loc[0, "TotalCharges"] = " "  # documented rule: blank -> NaN -> imputed
    varied.loc[1, "InternetService"] = "Satellite"  # handle_unknown="ignore"

    out = score_batch(fitted_pipeline, varied)

    assert len(out) == len(varied)
    assert np.isfinite(out["churn_score"]).all()
    assert out["churn_score"].between(0, 1).all()
    assert set(out["churn_prediction"]) <= {0, 1}


def test_malformed_numeric_text_is_rejected(raw_batch):
    # Whitespace-only is allowed (above); any other non-numeric text must fail.
    bad = raw_batch.copy()
    bad.loc[0, "MonthlyCharges"] = "abc"

    with pytest.raises(ValueError):
        validate_features(bad)


def test_target_and_id_columns_do_not_change_predictions(fitted_pipeline, raw_batch):
    # Churn and customerID may be present in a file but must never be model inputs.
    with_extra = raw_batch.assign(Churn=["Yes"] * len(raw_batch))
    features_only = raw_batch.drop(columns=["customerID"])

    a = score_batch(fitted_pipeline, with_extra)["churn_score"].to_numpy()
    b = score_batch(fitted_pipeline, features_only)["churn_score"].to_numpy()

    np.testing.assert_allclose(a, b)
    assert list(validate_features(with_extra).columns) == FEATURES


def test_one_output_row_per_customer_in_input_order(fitted_pipeline, raw_batch):
    shuffled = raw_batch.sample(frac=1, random_state=1).reset_index(drop=True)

    out = score_batch(fitted_pipeline, shuffled)

    pd.testing.assert_series_equal(out["customerID"], shuffled["customerID"], check_names=False)
