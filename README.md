# churnlab-2026

Reproducible churn prediction project on the IBM Telco Customer Churn dataset (ESILV, Machine Learning Operations).

## Business question

A telecommunications company wants to identify customers who are likely to leave its service, so it can act before they do.

- **Target column:** `Churn` (`Yes` / `No`), a binary classification target.
- **Baseline input fields:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`.
- `customerID` is an identifier and is never used as a feature. `Churn` must never be used as an input.

## Environment setup

Requirements: Python **3.11** (3.12 also accepted), Git and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"   # expected: 0.1.0
```

`--locked` installs exactly what `uv.lock` records and fails if `pyproject.toml` and `uv.lock` disagree. Do not regenerate the lock to make the error go away: investigate the mismatch.

To record the installed environment:

```sh
uv run python scripts/record_environment.py   # writes evidence/tp01/environment.json
```

## Dataset

| Item | Value |
|---|---|
| Source | IBM `telco-customer-churn-on-icp4d`, pinned to commit `1fd6fd7` |
| URL | https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv |
| Local path | `data/raw/telco.csv` (not committed) |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Shape | 7,043 rows, 21 columns |

Get the data and check its fingerprint:

```sh
uv run python scripts/download_data.py   # downloads if absent, verifies hash, writes data/raw/telco.source.json
uv run python scripts/inspect_telco.py    # writes reports/tp01_data_overview.json
```

Expected inspection results: 5,174 `No`, 1,869 `Yes` (about 26.5% churn), 0 duplicate `customerID`, 11 whitespace-only `TotalCharges`.

A file name is not a data version: only the hash identifies the bytes. Never open and re-save the CSV in a spreadsheet tool, it changes the bytes. The hash proves which file was used, not that the labels are correct.

## Repository layout

| Path | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, Python constraint, direct dependencies |
| `uv.lock` | Exact resolved versions of every package |
| `src/churnlab/` | Importable project code (data helpers, later training and serving) |
| `scripts/` | Executable workflow commands |
| `tests/` | Small behavior checks (`uv run pytest`) |
| `data/raw/` | Original input snapshot and its source metadata, never modified |
| `data/processed/` | Generated splits (later TPs) |
| `artifacts/` | Generated model files (later TPs, not committed) |
| `reports/` | Generated reports (not committed, except `tp03_summary.md`) |
| `mlflow.db`, `mlartifacts/` | Local MLflow backend and artifact stores (TP03, not committed) |
| `evidence/` | Small, non-sensitive submission evidence |
| `k8s/`, `pipelines/` | Deployment and orchestration (later TPs) |

## What this dataset is not

This is a **single customer snapshot**: one row per customer with their state at extraction time and whether they churned. It is **not** an event log with timestamps, so we cannot know the order in which things happened, and it is **not** an intervention or A/B dataset, so no retention action was randomly assigned. Models trained on it find associations, not causal effects of an action. No event history or treatment assignment should be invented and presented as observed IBM data.

## TP02: training pipeline

### Run it

```sh
uv sync --locked                      # on macOS add --no-editable (see note below)
uv run python scripts/train.py        # split, first model, comparison, selection, artifact
uv run python scripts/checks.py       # reliability checks -> evidence/tp02/checks.json
uv run python scripts/predict.py --input evidence/tp02/inference_batch.csv   # inference only
uv run pytest
```

`scripts/train.py --seed 42` is the single terminal entry point. The seed fixes the split and the tree-based models.

### What the workflow does

1. **Load and verify**: `churnlab.data.load_raw` checks the SHA-256 before reading the CSV. The file is never modified.
2. **Split**: stratified 60/20/20 on `Churn`, seed 42 (4,225 / 1,409 / 1,409 customers). The allocation of every `customerID` is saved in `evidence/tp02/split_allocation.csv`. The **test set is reserved**: it is not used for training or model selection.
3. **First model**: logistic regression on `tenure` and `MonthlyCharges`, plus a majority-class baseline, both fitted on train and evaluated on validation (`evidence/tp02/first_model.json`).
4. **Pipeline**: `churnlab.pipeline.build_pipeline(classifier)` returns an **unfitted** `Pipeline(ColumnTransformer -> classifier)`. It never reads data.
5. **Comparison**: baseline, logistic regression, decision tree and random forest, same six features, same split, same metrics (`evidence/tp02/comparison.csv`).
6. **Selection and artifact**: the selected fitted pipeline (preprocessing + classifier together) is saved to `artifacts/churn_pipeline.joblib`, with its metadata in `artifacts/churn_pipeline.metadata.json` (copied to `evidence/tp02/model_metadata.json`).

### Prediction task

- Positive class: `Churn = Yes`, encoded as 1. Decision threshold: 0.5 on the class-1 probability.
- `customerID` is kept only for identification and split checks. Neither `customerID` nor `Churn` is ever an input.

### Input schema (inference)

| Column | Type | Missing allowed? |
|---|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | number (or numeric text) | Yes: empty or whitespace-only becomes NaN, then imputed |
| `Contract`, `InternetService`, `PaymentMethod` | text category | Yes, imputed; unseen categories are accepted |

Boundary rules, applied by `churnlab.data.validate_features` before the pipeline, both at training and inference:
- whitespace-only numbers become missing values; **any other non-numeric text raises an error** (`pd.to_numeric(errors="raise")`);
- a missing required column raises a `KeyError`.

### Preprocessing (learned on training data only)

| Branch | Steps |
|---|---|
| Numeric | `SimpleImputer(median)` then `StandardScaler` |
| Categorical | `SimpleImputer(most_frequent)` then `OneHotEncoder(handle_unknown="ignore")` |

An unseen category (for example `InternetService = "Satellite"`) is encoded as all zeros for that column instead of crashing. This keeps the service running, but the prediction for such a customer is **not** reliable and should be monitored.

### Model selection rule (fixed before looking at the results)

Primary metric: **average precision** on validation, computed from class-1 probabilities. It focuses on how well the model ranks churners, it is threshold-independent and it is not inflated by the 73% majority class, unlike accuracy. If a simpler model is within 0.005 of the best, the simpler one is kept (logistic regression < decision tree < random forest). Precision, recall, F1, accuracy and the confusion matrix at threshold 0.5 are reported as supporting evidence.

### Artifact

`artifacts/` is ignored by Git. The fitted artifact is published as a GitHub release asset, and its SHA-256 is recorded in the metadata so a downloaded copy can be verified. Only load artifacts you trust: `joblib.load` can execute code. Load it with the library versions recorded in the metadata (scikit-learn 1.5.2).

### macOS note

On macOS the editable install file (`.pth`) can be marked hidden and ignored by Python, which gives `ModuleNotFoundError: churnlab`. Use `uv sync --locked --no-editable`. After changing code in `src/`, reinstall the package with `uv sync --locked --no-editable --reinstall-package churnlab`.

### Known limitations

- Probabilities are not calibrated: a score of 0.8 is not a verified 80% risk.
- Fit durations are indicative, measured on one machine.
- The test set has not been evaluated yet; it is kept for a single final evaluation.
- Experiment tracking is added in TP03 (see below).

## TP03: experiment tracking and model registry (MLflow)

### Run it

Terminal 1, from the project root (keep it running; UI at http://127.0.0.1:5000):

```sh
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

Terminal 2:

```sh
uv sync --locked --no-editable                 # MLflow 3.x is pinned in uv.lock
uv run python scripts/tracking_smoke.py        # first connection check: one run, one parameter
uv run python scripts/train_tracked.py         # one run per TP02 candidate, selection, registration + alias
uv run python scripts/predict_registered.py --input evidence/tp03/inference_batch.csv --reference evidence/tp03/reference_predictions.csv
uv run python scripts/tracking_status.py       # runs, registered versions and alias (before/after a restart)
```

`predict_registered.py` is inference only: it loads `models:/telco-churn@candidate`, applies `churnlab.data.validate_features`, never calls `fit` and imports no training code. It must be run in a new terminal process.

### What each run records

| Category | Content |
|---|---|
| Parameters | `model_type`, every actual classifier setting (`clf.*`), `split_seed`, `threshold`, `selection_metric` |
| Metrics | `val_average_precision` (primary), `val_precision`, `val_recall`, `val_f1`, `val_accuracy`, `fit_seconds` |
| Tags | `git_commit`, `git_uncommitted_changes`, `data_sha256`, `split_allocation_sha256`, `run_role` |
| Artifacts | `evaluation/val_confusion_matrix.png`, `provenance.json` (features, class mapping, split reference, evaluation rules, environment) |
| Dataset inputs | `telco_train` (context `training`), `telco_validation` (context `validation`) |
| Model | `churn_pipeline`: the complete fitted preprocessing + classifier pipeline (`mlflow.sklearn.log_model`, cloudpickle, input example + signature) |

The split is the TP02 one: `train_tracked.py` stops if the recomputed allocation differs from `evidence/tp02/split_allocation.csv`. The test set is never evaluated.

### Where tracking data live (local, not committed)

| Component | Location | Holds |
|---|---|---|
| Backend store | `mlflow.db` (SQLite) | experiments, runs, params, metrics, tags, dataset records, registered models, versions, aliases |
| Artifact store | `mlartifacts/` | model files (`MLmodel`, `model.pkl`, environment files), confusion matrices, `provenance.json` |
| Tracking server | `mlflow server` process | tracking API + UI; stopping it deletes nothing |

Both are ignored by Git but must be kept: restart the server **from the project root with the same command** to get everything back.

### Recover the selected model

`evidence/tp03/handover.json` records the source run ID, the logged model URI, the registered version, the alias, the Git commit and the data hash. With the server running, `mlflow.sklearn.load_model("models:/telco-churn@candidate")` (or `models:/telco-churn/<version>` for a fixed version) returns the fitted pipeline without retraining.

### Limitations

- The tracking server is local (`127.0.0.1`): a run ID or a model URI alone is useless on another machine. Sharing needs a team server, or exporting both `mlflow.db` and `mlartifacts/`, plus this repository at the recorded commit, the `uv.lock` environment and the original CSV (checked against its SHA-256).
- The alias `candidate` is a mutable pointer, not a deployment and not a copy: always record the version number with it.
- The dataset records describe the data (name, digest, schema); they do not store it and are not dataset version control. MLflow's dataset digest is not the CSV SHA-256.
- The model is serialized with cloudpickle: load only trusted, self-produced artifacts.
