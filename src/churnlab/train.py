"""Training workflow: split, first model, fair comparison, selection, artifact."""

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from churnlab.data import (
    EXPECTED_SHA256,
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    POSITIVE_LABEL,
    RAW_CSV,
    SOURCE_URL,
    TARGET,
    encode_target,
    load_raw,
    sha256_of,
    validate_features,
)
from churnlab.evaluate import THRESHOLD, evaluate
from churnlab.pipeline import PREPROCESSING, build_pipeline
from churnlab.split import SPLIT_SEED, allocation_table, check_disjoint, make_splits

# Selection rule, fixed BEFORE looking at results:
# highest validation average precision; if a simpler model is within
# SIMPLER_MARGIN of the best, keep the simpler one.
SELECTION_METRIC = "average_precision"
SIMPLER_MARGIN = 0.005
COMPLEXITY_ORDER = ["logistic_regression", "decision_tree", "random_forest"]


def candidate_models(seed: int) -> dict:
    """Modest, explicit configurations (no hyperparameter search)."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=seed
        ),
    }


def git_identity() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "uncommitted_changes": dirty}


def environment_identity() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            p: version(p) for p in ["churnlab", "scikit-learn", "pandas", "numpy", "joblib"]
        },
    }


def first_model(train_df, valid_df) -> dict:
    """Milestone 1: two numeric features, one simple model and a baseline."""
    feats = ["tenure", "MonthlyCharges"]
    X_train = train_df[feats].astype("float64")
    X_valid = valid_df[feats].astype("float64")
    y_train, y_valid = encode_target(train_df[TARGET]), encode_target(valid_df[TARGET])
    out = {"features": feats, "positive_class": "Churn = Yes (1)",
           "confusion_matrix_layout": "rows = true [0, 1], cols = predicted [0, 1]"}
    for name, clf in [
        ("majority_baseline", DummyClassifier(strategy="most_frequent")),
        ("logistic_regression", LogisticRegression(max_iter=1000)),
    ]:
        clf.fit(X_train, y_train)  # baseline is fitted on TRAINING data too
        out[name] = evaluate(clf, X_valid, y_valid)
    return out


def select(results: list[dict]) -> str:
    learned = [r for r in results if r["model"] in COMPLEXITY_ORDER]
    best = max(r[SELECTION_METRIC] for r in learned)
    for name in COMPLEXITY_ORDER:  # simplest first
        r = next(r for r in learned if r["model"] == name)
        if r[SELECTION_METRIC] >= best - SIMPLER_MARGIN:
            return name
    raise RuntimeError("no candidate selected")


def run(seed: int = SPLIT_SEED, artifacts_dir: Path = Path("artifacts"),
        evidence_dir: Path = Path("evidence/tp02")) -> dict:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(RAW_CSV)
    train_df, valid_df, test_df = make_splits(raw, seed)
    overlap = check_disjoint(train_df, valid_df, test_df)
    if any(overlap.values()):
        raise RuntimeError(f"Customer overlap between splits: {overlap}")
    alloc = allocation_table(train_df, valid_df, test_df)
    alloc_path = evidence_dir / "split_allocation.csv"
    alloc.to_csv(alloc_path, index=False)

    # Milestone 1
    first = first_model(train_df, valid_df)
    (evidence_dir / "first_model.json").write_text(json.dumps(first, indent=2), encoding="utf-8")

    # Milestones 2-3: six features, same split for every candidate
    X_train, X_valid = validate_features(train_df), validate_features(valid_df)
    y_train, y_valid = encode_target(train_df[TARGET]), encode_target(valid_df[TARGET])

    results, fitted = [], {}
    for name, clf in candidate_models(seed).items():
        pipe = build_pipeline(clf)  # fresh, unfitted preprocessing each time
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics = evaluate(pipe, X_valid, y_valid)
        cm = metrics.pop("confusion_matrix")
        results.append({
            "model": name,
            **metrics,
            **{f"cm_{k}": v for k, v in cm.items()},
            "fit_seconds": round(fit_seconds, 3),
            "config": json.dumps({k: v for k, v in clf.get_params().items()}, default=str),
        })
        fitted[name] = pipe

    table = pd.DataFrame(results)
    table.to_csv(evidence_dir / "comparison.csv", index=False)
    (evidence_dir / "comparison.json").write_text(table.to_json(orient="records", indent=2), encoding="utf-8")

    selected_name = select(results)
    selected = fitted[selected_name]
    artifact_path = artifacts_dir / "churn_pipeline.joblib"
    joblib.dump(selected, artifact_path)

    # Reference batch for the fresh-process reload check (features only, raw text form).
    batch = valid_df[[ID_COLUMN] + FEATURES].head(20)
    batch.to_csv(evidence_dir / "inference_batch.csv", index=False)
    ref = pd.DataFrame({
        ID_COLUMN: batch[ID_COLUMN].values,
        "score": selected.predict_proba(validate_features(batch))[:, list(selected.classes_).index(1)],
        "prediction": selected.predict(validate_features(batch)),
    })
    ref.to_csv(evidence_dir / "reference_predictions.csv", index=False)

    selected_row = next(r for r in results if r["model"] == selected_name)
    metadata = {
        "artifact": artifact_path.as_posix(),
        "artifact_sha256": sha256_of(artifact_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"url": SOURCE_URL, "file": RAW_CSV.as_posix(), "sha256": EXPECTED_SHA256},
        "split": {
            "seed": seed, "ratios": "60/20/20 stratified on Churn",
            "sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
            "allocation_file": alloc_path.as_posix(),
            "allocation_sha256": sha256_of(alloc_path),
            "test_set": "reserved, not used for training or selection",
        },
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
                     "identifier_not_used": ID_COLUMN, "target": TARGET},
        "input_schema": {c: "float (blank allowed -> NaN)" for c in NUMERIC_FEATURES}
                        | {c: "string category" for c in CATEGORICAL_FEATURES},
        "preprocessing": PREPROCESSING,
        "positive_class": f"{TARGET} = {POSITIVE_LABEL} (encoded 1)",
        "threshold": THRESHOLD,
        "selection": {"metric": SELECTION_METRIC, "data": "validation only",
                      "rule": f"best {SELECTION_METRIC}; prefer simpler if within {SIMPLER_MARGIN}",
                      "selected_model": selected_name},
        "model_params": json.loads(selected_row["config"]),
        "validation_metrics": {k: selected_row[k] for k in
                               ["average_precision", "precision", "recall", "f1", "accuracy",
                                "cm_tn", "cm_fp", "cm_fn", "cm_tp"]},
        "code": git_identity(),
        "environment": environment_identity(),
    }
    meta_text = json.dumps(metadata, indent=2)
    (artifacts_dir / "churn_pipeline.metadata.json").write_text(meta_text, encoding="utf-8")
    (evidence_dir / "model_metadata.json").write_text(meta_text, encoding="utf-8")
    return {"first_model": first, "comparison": table, "selected": selected_name,
            "metadata": metadata}
