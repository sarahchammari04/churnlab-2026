"""MLflow tracking helpers (TP03). Importing this module never trains anything.

One candidate = one MLflow run holding:
- params: estimator type, actual hyperparameters, split seed, threshold, selection metric
- metrics: val_average_precision, val_precision, val_recall, val_f1 (+ accuracy, fit time)
- tags: Git commit / dirty flag, SHA-256 of the original CSV and of the split allocation
- artifacts: validation confusion matrix + provenance.json
- dataset inputs: training and validation records (distinct contexts)
- model: the complete fitted preprocessing + classifier pipeline
"""

import json
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: figures are only written to files
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from churnlab.data import FEATURES, TARGET
from churnlab.evaluate import THRESHOLD, evaluate
from churnlab.pipeline import build_pipeline

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "telco-churn-tp03"
REGISTERED_MODEL = "telco-churn"
ALIAS = "candidate"
MODEL_ARTIFACT_NAME = "churn_pipeline"


def connect(tracking_uri: str = TRACKING_URI, experiment: str = EXPERIMENT_NAME):
    """Point the client at the tracking server and select (or create) the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(experiment)


def dataset_records(train_df: pd.DataFrame, valid_df: pd.DataFrame, source: str):
    """Dataset metadata (name, digest, schema) for the existing splits.

    The raw target column is included only to describe the dataset; the
    estimator still receives the six features and the encoded target.
    Logging these records does not store the DataFrames themselves.
    """
    cols = FEATURES + [TARGET]
    train = mlflow.data.from_pandas(train_df[cols], source=source, name="telco_train", targets=TARGET)
    valid = mlflow.data.from_pandas(valid_df[cols], source=source, name="telco_validation", targets=TARGET)
    return train, valid


def _confusion_matrix_png(cm: dict, title: str, out_dir: Path) -> Path:
    # Class order: rows = true [0, 1], columns = predicted [0, 1].
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["0: No churn", "1: Churn"]).plot(ax=ax, colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = out_dir / "val_confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _param_value(value):
    # MLflow params are strings: None and objects are written explicitly.
    return "None" if value is None else value if isinstance(value, (int, float, str, bool)) else str(value)


def track_candidate(run_name: str, model_key: str, classifier, data: dict, context: dict) -> dict:
    """Fit, evaluate and record one candidate inside its own MLflow run.

    `data` holds X_train, y_train, X_valid, y_valid and the two dataset records.
    `context` holds the shared tags and provenance (identical for every candidate).
    """
    pipeline = build_pipeline(classifier)  # fresh, unfitted preprocessing for each run
    actual_params = classifier.get_params()  # settings actually passed, not intended ones

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model_key": model_key,
            "model_type": type(classifier).__name__,
            "split_seed": context["seed"],
            "threshold": THRESHOLD,
            "selection_metric": "val_average_precision",
            **{f"clf.{k}": _param_value(v) for k, v in actual_params.items()},
        })
        mlflow.set_tags({**context["tags"], "run_role": context["run_role"]})

        # Dataset records are attached inside the active run, with distinct contexts.
        mlflow.log_input(data["train_dataset"], context="training")
        mlflow.log_input(data["valid_dataset"], context="validation")

        start = time.perf_counter()
        pipeline.fit(data["X_train"], data["y_train"])  # training rows only
        fit_seconds = time.perf_counter() - start

        # TP02 evaluation function: AP on class-1 scores, other metrics at THRESHOLD.
        metrics = evaluate(pipeline, data["X_valid"], data["y_valid"])
        cm = metrics.pop("confusion_matrix")
        val_metrics = {f"val_{k}": float(v) for k, v in metrics.items()}
        mlflow.log_metrics({**val_metrics, "fit_seconds": fit_seconds})

        with tempfile.TemporaryDirectory() as tmp:
            png = _confusion_matrix_png(cm, f"{run_name} - validation (threshold {THRESHOLD})", Path(tmp))
            mlflow.log_artifact(str(png), artifact_path="evaluation")
        mlflow.log_dict(
            {**context["provenance"],
             "candidate": {"run_name": run_name, "model_key": model_key,
                           "model_type": type(classifier).__name__,
                           "params": {k: _param_value(v) for k, v in actual_params.items()}},
             "validation_confusion_matrix": {**cm, "layout": "rows = true [0, 1], cols = predicted [0, 1]"}},
            "provenance.json",
        )

        # Complete fitted pipeline (preprocessing + classifier) through the MLflow model API.
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=MODEL_ARTIFACT_NAME,
            input_example=data["X_train"].head(3),  # original, unencoded feature columns
            serialization_format="cloudpickle",
        )

    return {
        "model": model_key,
        "run_name": run_name,
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        **val_metrics,
        **{f"cm_{k}": v for k, v in cm.items()},
        "fit_seconds": round(fit_seconds, 3),
        "pipeline": pipeline,
    }


def register_selected(model_uri: str, name: str = REGISTERED_MODEL, alias: str = ALIAS,
                      version_tags: dict | None = None):
    """Register a logged model and point the alias at the returned version."""
    version = mlflow.register_model(model_uri, name)
    client = mlflow.MlflowClient()
    for key, value in (version_tags or {}).items():
        client.set_model_version_tag(name, version.version, key, str(value))
    client.set_registered_model_alias(name, alias, version.version)
    return version


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
