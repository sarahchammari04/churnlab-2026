"""Probability contract. Class order: classes_ == [0, 1], 1 = Churn Yes."""

import numpy as np

from churnlab.evaluate import positive_scores
from churnlab.predict import score_batch


# Test C: valid probabilities, one row per customer, positive class via classes_.
def test_probability_contract(fitted_pipeline, sample_features):
    probabilities = fitted_pipeline.predict_proba(sample_features)

    assert list(fitted_pipeline.classes_) == [0, 1]
    assert probabilities.shape == (len(sample_features), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)

    positive_col = list(fitted_pipeline.classes_).index(1)
    np.testing.assert_allclose(
        positive_scores(fitted_pipeline, sample_features), probabilities[:, positive_col]
    )


class _ReversedClassesModel:
    """Stand-in whose classes_ are [1, 0]: column 0 is the positive class."""

    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        return np.tile([0.9, 0.1], (len(X), 1))


def test_positive_score_uses_classes_not_column_position(sample_features):
    # A hardcoded proba[:, 1] would return 0.1 here instead of 0.9.
    scores = positive_scores(_ReversedClassesModel(), sample_features)

    np.testing.assert_allclose(scores, 0.9)


def test_prediction_follows_documented_threshold(fitted_pipeline, raw_batch):
    out = score_batch(fitted_pipeline, raw_batch, threshold=0.5)

    expected = (out["churn_score"] >= 0.5).astype(int)
    assert (out["churn_prediction"] == expected).all()
