"""Focused reliability checks on the saved artifact (TP02 section 5).

uv run python scripts/checks.py   -> evidence/tp02/checks.json
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlab.data import (
    CATEGORICAL_FEATURES, FEATURES, ID_COLUMN, NUMERIC_FEATURES, load_raw, validate_features,
)

EVIDENCE = Path("evidence/tp02")
ARTIFACT = Path("artifacts/churn_pipeline.joblib")
TOL = 1e-9
results = {}

pipe = joblib.load(ARTIFACT)
raw = load_raw()
alloc = pd.read_csv(EVIDENCE / "split_allocation.csv")

# 1. Disjoint customer IDs
groups = {s: set(alloc.loc[alloc.split == s, ID_COLUMN]) for s in ["train", "valid", "test"]}
results["1_disjoint_ids"] = {
    "sizes": {k: len(v) for k, v in groups.items()},
    "train_valid": len(groups["train"] & groups["valid"]),
    "train_test": len(groups["train"] & groups["test"]),
    "valid_test": len(groups["valid"] & groups["test"]),
    "all_customers_allocated_once": int(alloc[ID_COLUMN].is_unique and len(alloc) == len(raw)),
}

# 2. Learned statistics come from training rows only
num = pipe.named_steps["preprocessor"].named_transformers_["num"]
imputer_stats = num.named_steps["imputer"].statistics_.copy()
scaler_mean = num.named_steps["scaler"].mean_.copy()
train_X = validate_features(raw[raw[ID_COLUMN].isin(groups["train"])])
full_X = validate_features(raw)
train_median = train_X[NUMERIC_FEATURES].median().to_numpy()
full_median = full_X[NUMERIC_FEATURES].median().to_numpy()
imputed_train = train_X[NUMERIC_FEATURES].fillna(dict(zip(NUMERIC_FEATURES, train_median)))
results["2_training_only_statistics"] = {
    "imputer_medians": dict(zip(NUMERIC_FEATURES, imputer_stats.tolist())),
    "training_medians": dict(zip(NUMERIC_FEATURES, train_median.tolist())),
    "full_dataset_medians": dict(zip(NUMERIC_FEATURES, full_median.tolist())),
    "imputer_matches_training": bool(np.allclose(imputer_stats, train_median)),
    "scaler_mean_matches_training": bool(np.allclose(scaler_mean, imputed_train.mean().to_numpy())),
}

# 3. Predicting does not refit
valid_X = validate_features(raw[raw[ID_COLUMN].isin(groups["valid"])])
before = copy.deepcopy((imputer_stats, scaler_mean))
pipe.predict_proba(valid_X)
after = (num.named_steps["imputer"].statistics_, num.named_steps["scaler"].mean_)
results["3_predict_does_not_refit"] = {
    "imputer_unchanged": bool(np.array_equal(before[0], after[0])),
    "scaler_unchanged": bool(np.array_equal(before[1], after[1])),
}

# 4. Documented policies: missing numeric value and unseen category
base = raw[raw[ID_COLUMN].isin(groups["valid"])][FEATURES].head(1).astype(str).reset_index(drop=True)
missing_num = base.copy(); missing_num.loc[0, "TotalCharges"] = " "
unseen_cat = base.copy(); unseen_cat.loc[0, "InternetService"] = "Satellite"
pre = pipe.named_steps["preprocessor"]
enc = pre.named_transformers_["cat"].named_steps["encoder"]
inet_idx = CATEGORICAL_FEATURES.index("InternetService")
start = len(NUMERIC_FEATURES) + sum(len(c) for c in enc.categories_[:inet_idx])
width = len(enc.categories_[inet_idx])
row_missing = pre.transform(validate_features(missing_num))[0]
row_unseen = pre.transform(validate_features(unseen_cat))[0]
results["4_policies"] = {
    "missing_TotalCharges_imputed_to_training_median": bool(np.isclose(
        row_missing[2] * num.named_steps["scaler"].scale_[2] + scaler_mean[2], imputer_stats[2])),
    "missing_TotalCharges_score": float(pipe.predict_proba(validate_features(missing_num))[0, 1]),
    "unseen_category_encoded_as_zeros": bool(np.all(row_unseen[start:start + width] == 0)),
    "unseen_category_score": float(pipe.predict_proba(validate_features(unseen_cat))[0, 1]),
    "note": "Handling an unseen category only avoids a crash; it does not make the prediction reliable.",
}

# 4b. Malformed non-blank numeric text must fail
bad = base.copy(); bad.loc[0, "MonthlyCharges"] = "abc"
try:
    validate_features(bad); results["4b_malformed_numeric"] = "NOT DETECTED"
except (ValueError, TypeError) as e:
    results["4b_malformed_numeric"] = f"raised {type(e).__name__}"

# 5. Missing required feature -> visible failure (boundary and pipeline)
dropped = base.drop(columns=["Contract"])
out5 = {}
try:
    validate_features(dropped); out5["boundary"] = "NOT DETECTED"
except KeyError as e:
    out5["boundary"] = f"raised KeyError: {e}"
try:
    pipe.predict(validate_features(base).drop(columns=["Contract"])); out5["pipeline"] = "NOT DETECTED"
except (KeyError, ValueError) as e:
    out5["pipeline"] = f"raised {type(e).__name__}"
results["5_missing_feature"] = out5

# 6. Reload in a FRESH process (scripts/predict.py) and compare with the original scores
pred_path = Path("reports/reload_check_predictions.csv")
subprocess.run([sys.executable, "scripts/predict.py", "--input", str(EVIDENCE / "inference_batch.csv"),
                "--output", str(pred_path)], check=True, capture_output=True)
ref = pd.read_csv(EVIDENCE / "reference_predictions.csv")
new = pd.read_csv(pred_path)
max_diff = float(np.max(np.abs(ref["score"].to_numpy() - new["churn_score"].to_numpy())))
predict_src = Path("scripts/predict.py").read_text(encoding="utf-8")
results["6_fresh_process_reload"] = {
    "rows": len(ref),
    "max_abs_score_difference": max_diff,
    "tolerance": TOL,
    "scores_agree": max_diff <= TOL,
    "classes_agree": bool((ref["prediction"].to_numpy() == new["churn_prediction"].to_numpy()).all()),
    "inference_script_calls_fit": ".fit(" in predict_src,
    "inference_input_contains_target": "Churn" in pd.read_csv(EVIDENCE / "inference_batch.csv").columns,
}

(EVIDENCE / "checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
