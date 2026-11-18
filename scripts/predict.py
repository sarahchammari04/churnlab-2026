"""Inference only: load the trusted artifact and score a CSV of customers.

No training data, no labels and no fit call are used here.
uv run python scripts/predict.py --input evidence/tp02/inference_batch.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churnlab.predict import score_batch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=Path("artifacts/churn_pipeline.joblib"))
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()

pipeline = joblib.load(args.model)  # trusted artifact only
raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
out = score_batch(pipeline, raw, args.threshold)  # explicit boundary: schema + types
args.output.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, index=False)
print(out.to_string(index=False))
