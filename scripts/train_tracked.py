"""Tracked training (TP03): one MLflow run per TP02 candidate, then registration.

Start the tracking server first (separate terminal, project root):
    uv run mlflow server --host 127.0.0.1 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
Then:
    uv run python scripts/train_tracked.py
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from churnlab.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    RAW_CSV,
    SOURCE_URL,
    TARGET,
    encode_target,
    load_raw,
    sha256_of,
    validate_features,
)
from churnlab.evaluate import THRESHOLD, positive_scores
from churnlab.pipeline import PREPROCESSING
from churnlab.split import SPLIT_SEED, allocation_table, check_disjoint, make_splits
from churnlab.tracking import (
    ALIAS,
    EXPERIMENT_NAME,
    REGISTERED_MODEL,
    TRACKING_URI,
    connect,
    dataset_records,
    register_selected,
    track_candidate,
    write_json,
)
from churnlab.train import (
    COMPLEXITY_ORDER,
    SELECTION_METRIC,
    SIMPLER_MARGIN,
    candidate_models,
    environment_identity,
    git_identity,
    select,
)

# Descriptive run names for the TP02 configurations (unchanged settings).
RUN_NAMES = {
    "majority_baseline": "baseline-majority",
    "logistic_regression": "logistic-C1",
    "decision_tree": "tree-depth5-leaf20",
    "random_forest": "forest-300trees-depth8",
}
TP02_ALLOCATION = Path("evidence/tp02/split_allocation.csv")

parser = argparse.ArgumentParser(description="Train TP02 candidates with MLflow tracking.")
parser.add_argument("--seed", type=int, default=SPLIT_SEED)
parser.add_argument("--tracking-uri", default=TRACKING_URI)
parser.add_argument("--experiment", default=EXPERIMENT_NAME)
parser.add_argument("--evidence-dir", type=Path, default=Path("evidence/tp03"))
parser.add_argument("--no-register", action="store_true", help="Skip registry + alias")
args = parser.parse_args()

experiment = connect(args.tracking_uri, args.experiment)
print(f"Experiment '{experiment.name}' (id {experiment.experiment_id}) on {args.tracking_uri}")

# --- Data and split: identical to TP02 --------------------------------------
raw = load_raw(RAW_CSV)  # hash checked before reading
data_sha256 = sha256_of(RAW_CSV)
train_df, valid_df, test_df = make_splits(raw, args.seed)
overlap = check_disjoint(train_df, valid_df, test_df)
if any(overlap.values()):
    raise RuntimeError(f"Customer overlap between splits: {overlap}")

# The customer allocation must match the one recorded in TP02 (no silent re-split).
alloc = allocation_table(train_df, valid_df, test_df)
reference_alloc = pd.read_csv(TP02_ALLOCATION)
if not alloc.reset_index(drop=True).equals(reference_alloc):
    raise RuntimeError("Split allocation differs from evidence/tp02/split_allocation.csv")
allocation_sha256 = sha256_of(TP02_ALLOCATION)
# test_df is never evaluated here: the test set stays reserved.

X_train, X_valid = validate_features(train_df), validate_features(valid_df)
y_train, y_valid = encode_target(train_df[TARGET]), encode_target(valid_df[TARGET])
train_dataset, valid_dataset = dataset_records(train_df, valid_df, source=RAW_CSV.as_posix())

# --- Shared provenance ----------------------------------------------------
git = git_identity()
if git["uncommitted_changes"]:
    print(
        "WARNING: uncommitted changes present; the commit ID alone does not describe the code run."
    )

tags = {
    "git_commit": str(git["commit"]),
    "git_uncommitted_changes": str(git["uncommitted_changes"]).lower(),
    "data_sha256": data_sha256,
    "split_allocation_sha256": allocation_sha256,
}
provenance = {
    "source": {"url": SOURCE_URL, "file": RAW_CSV.as_posix(), "sha256": data_sha256},
    "split": {
        "seed": args.seed,
        "ratios": "60/20/20 stratified on Churn",
        "sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
        "allocation_file": TP02_ALLOCATION.as_posix(),
        "allocation_sha256": allocation_sha256,
        "test_set": "reserved, not used for training or selection",
    },
    "features": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "excluded": [ID_COLUMN, TARGET],
    },
    "class_mapping": {"No": 0, "Yes": 1, "positive_class": f"{TARGET} = {POSITIVE_LABEL} (1)"},
    "preprocessing": PREPROCESSING,
    "evaluation": {
        "population": "validation split only",
        "primary_metric": "val_average_precision (continuous class-1 scores, column found via classes_)",
        "threshold_metrics": {
            "threshold": THRESHOLD,
            "metrics": ["precision", "recall", "f1", "accuracy"],
        },
        "confusion_matrix_layout": "rows = true [0, 1], cols = predicted [0, 1]",
    },
    "selection_rule": f"best validation {SELECTION_METRIC}; prefer simpler model if within "
    f"{SIMPLER_MARGIN} (order: {' < '.join(COMPLEXITY_ORDER)})",
    "code": git,
    "environment": environment_identity() | {"mlflow": mlflow.__version__},
}

# --- One run per candidate ------------------------------------------------
data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_valid": X_valid,
    "y_valid": y_valid,
    "train_dataset": train_dataset,
    "valid_dataset": valid_dataset,
}
results = []
for key, classifier in candidate_models(args.seed).items():
    role = "baseline" if key == "majority_baseline" else "candidate"
    res = track_candidate(
        RUN_NAMES[key],
        key,
        classifier,
        data,
        {"seed": args.seed, "tags": tags, "provenance": provenance, "run_role": role},
    )
    results.append(res)
    print(
        f"{res['run_name']:24s} run_id={res['run_id']}  AP={res['val_average_precision']:.4f}  "
        f"recall={res['val_recall']:.4f}  F1={res['val_f1']:.4f}"
    )

# --- Selection with the rule fixed in TP02 --------------------------------
selected_key = select(
    [{"model": r["model"], SELECTION_METRIC: r["val_average_precision"]} for r in results]
)
selected = next(r for r in results if r["model"] == selected_key)

table = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
table["decision"] = ["selected" if m == selected_key else "not selected" for m in table["model"]]
args.evidence_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(args.evidence_dir / "comparison.csv", index=False)

# --- Reference predictions from the in-memory pipeline (for the reload check) ---
batch = valid_df[[ID_COLUMN] + FEATURES].head(20)
batch.to_csv(args.evidence_dir / "inference_batch.csv", index=False)
X_batch = validate_features(batch)
pd.DataFrame(
    {
        ID_COLUMN: batch[ID_COLUMN].values,
        "score": positive_scores(selected["pipeline"], X_batch),
        "prediction": selected["pipeline"].predict(X_batch),
    }
).to_csv(args.evidence_dir / "reference_predictions.csv", index=False)

handover = {
    "selected_model": selected_key,
    "source_run_name": selected["run_name"],
    "source_run_id": selected["run_id"],
    "logged_model_uri": selected["model_uri"],
    "val_average_precision": selected["val_average_precision"],
    "git_commit": git["commit"],
    "git_uncommitted_changes": git["uncommitted_changes"],
    "data_sha256": data_sha256,
    "split_allocation_sha256": allocation_sha256,
    "tracking_uri": args.tracking_uri,
    "experiment": args.experiment,
}

# --- Registry: name + version + alias -------------------------------------
if not args.no_register:
    version = register_selected(
        selected["model_uri"],
        version_tags={
            "source_run_id": selected["run_id"],
            "selection_metric": "val_average_precision",
            "data_sha256": data_sha256,
            "git_commit": git["commit"],
        },
    )
    resolved = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    handover |= {
        "registered_model": REGISTERED_MODEL,
        "registered_version": version.version,
        "alias": ALIAS,
        "alias_resolves_to": resolved.version,
    }

write_json(args.evidence_dir / "handover.json", handover)
print(f"\nSelected: {selected_key} ({selected['run_name']})")
for k, v in handover.items():
    print(f"  {k}: {v}")
