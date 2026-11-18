"""Inference only (TP03): load the registered pipeline through its alias.

No training data, no labels and no fit call. Run it in a NEW terminal while
the tracking server is running:
    uv run python scripts/predict_registered.py \
        --input evidence/tp03/inference_batch.csv \
        --reference evidence/tp03/reference_predictions.csv
"""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from churnlab.data import ID_COLUMN, validate_features

parser = argparse.ArgumentParser()
parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
parser.add_argument("--name", default="telco-churn")
parser.add_argument("--alias", default="candidate")
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--reference", type=Path, help="Scores from the original in-memory pipeline")
parser.add_argument("--output", type=Path, default=Path("reports/tp03_reloaded_predictions.csv"))
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--tol", type=float, default=1e-9, help="Absolute tolerance on scores")
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
model_uri = f"models:/{args.name}@{args.alias}"

# The alias is a mutable pointer: the concrete version is printed for traceability.
mv = mlflow.MlflowClient().get_model_version_by_alias(args.name, args.alias)
print(f"{model_uri} -> version {mv.version} (source run {mv.run_id})")

pipeline = mlflow.sklearn.load_model(model_uri)  # trusted, self-produced artifact only
raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
X = validate_features(raw)  # same stateless boundary rules as TP02
scores = pipeline.predict_proba(X)[:, list(pipeline.classes_).index(1)]
out = pd.DataFrame(
    {
        ID_COLUMN: raw[ID_COLUMN] if ID_COLUMN in raw else range(len(raw)),
        "churn_score": scores,
        "churn_prediction": (scores >= args.threshold).astype(int),
    }
)
args.output.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, index=False)
print(out.to_string(index=False))

if args.reference:
    ref = pd.read_csv(args.reference)
    max_diff = float(np.max(np.abs(ref["score"].to_numpy() - out["churn_score"].to_numpy())))
    classes_agree = bool((ref["prediction"].to_numpy() == out["churn_prediction"].to_numpy()).all())
    ok = classes_agree and max_diff <= args.tol
    print(
        f"\nRows: {len(ref)} | classes agree: {classes_agree} | "
        f"max |score diff|: {max_diff:.3e} (tolerance {args.tol:g}) -> {'AGREEMENT OK' if ok else 'MISMATCH'}"
    )
    sys.exit(0 if ok else 1)
