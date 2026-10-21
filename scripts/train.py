"""Terminal entry point: uv run python scripts/train.py [--seed 42]"""

import argparse
from pathlib import Path

import pandas as pd

from churnlab.train import run

parser = argparse.ArgumentParser(description="Train and compare churn models.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
parser.add_argument("--evidence-dir", type=Path, default=Path("evidence/tp02"))
args = parser.parse_args()

out = run(args.seed, args.artifacts_dir, args.evidence_dir)

fm = out["first_model"]
print("== Milestone 1: tenure + MonthlyCharges (validation) ==")
for name in ["majority_baseline", "logistic_regression"]:
    m = fm[name]
    print(f"{name:22s} accuracy={m['accuracy']:.3f} recall={m['recall']:.3f} cm={m['confusion_matrix']}")

print("\n== Comparison: six features, validation ==")
cols = ["model", "average_precision", "precision", "recall", "f1", "accuracy",
        "cm_tn", "cm_fp", "cm_fn", "cm_tp", "fit_seconds"]
with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
    print(out["comparison"][cols].to_string(index=False))
print(f"\nSelected: {out['selected']} -> {out['metadata']['artifact']}")
