"""Read-only summary of what the tracking server holds (used before/after restart).

uv run python scripts/tracking_status.py [--export evidence/tp03/search_runs.csv]
"""

import argparse
from pathlib import Path

import mlflow

from churnlab.tracking import ALIAS, EXPERIMENT_NAME, REGISTERED_MODEL, TRACKING_URI

parser = argparse.ArgumentParser()
parser.add_argument("--tracking-uri", default=TRACKING_URI)
parser.add_argument("--export", type=Path, help="Optional CSV export of the candidate runs")
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
client = mlflow.MlflowClient()

runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME], order_by=["metrics.val_average_precision DESC"]
)
tracked = runs[runs["tags.run_role"].isin(["candidate", "baseline"])]
print(f"Experiment '{EXPERIMENT_NAME}': {len(runs)} runs ({len(tracked)} candidate/baseline)")

cols = [
    "run_id",
    "tags.mlflow.runName",
    "metrics.val_average_precision",
    "metrics.val_precision",
    "metrics.val_recall",
    "metrics.val_f1",
]
print(tracked[cols].to_string(index=False))
if args.export:
    args.export.parent.mkdir(parents=True, exist_ok=True)
    tracked[cols].to_csv(args.export, index=False)

versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
print(
    f"\nRegistered model '{REGISTERED_MODEL}': versions {sorted(int(v.version) for v in versions)}"
)
mv = client.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
print(f"Alias '{ALIAS}' -> version {mv.version} (run {mv.run_id}, source {mv.source})")
