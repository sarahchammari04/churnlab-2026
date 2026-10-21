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
| `reports/` | Generated reports (not committed) |
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
- No experiment tracking yet (next TP).
