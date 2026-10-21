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
