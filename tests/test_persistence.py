"""Save/reload agreement for the complete fitted pipeline (written in tmp_path only)."""

import joblib
import numpy as np


# Test D: serialization preserves behavior, without refitting.
def test_reload_preserves_predictions(tmp_path, fitted_pipeline, sample_features):
    artifact = tmp_path / "pipeline.joblib"
    before_proba = fitted_pipeline.predict_proba(sample_features)
    before_class = fitted_pipeline.predict(sample_features)

    joblib.dump(fitted_pipeline, artifact)
    reloaded = joblib.load(artifact)  # own artifact created by this test only

    np.testing.assert_array_equal(reloaded.predict(sample_features), before_class)
    np.testing.assert_allclose(reloaded.predict_proba(sample_features), before_proba, atol=1e-12)


def test_saved_artifact_contains_fitted_preprocessing(tmp_path, fitted_pipeline, sample_features):
    # Raw categories (text) can only be scored if the encoder was saved with the model.
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_pipeline, artifact)
    reloaded = joblib.load(artifact)

    assert list(reloaded.named_steps) == ["preprocessor", "classifier"]
    encoder = reloaded.named_steps["preprocessor"].named_transformers_["cat"]
    assert hasattr(encoder.named_steps["encoder"], "categories_")
    assert reloaded.predict(sample_features).shape == (len(sample_features),)
