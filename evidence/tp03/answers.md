# TP03 answers

**Student:** Sarah Chammari and Shaima Kraiem
**Entry points:** `uv run python scripts/train_tracked.py` (training) and `uv run python scripts/predict_registered.py` (inference)
**Code commit used for the runs:** d51e0ff
**MLflow:** 3.18.0, local server on port 5000, experiment `telco-churn-tp03`

## 1. Local tracking

```sh
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

First test run: `smoke-test`, run ID `8ab3e683638242a18f9a5cca7bb3b55b` (experiment id 1).

| Component | Where | Content |
|---|---|---|
| Backend store | `mlflow.db` | runs, params, metrics, tags, registry |
| Artifact store | `mlartifacts/` | model files, confusion matrices, provenance.json |
| Server | `mlflow server` | API + UI |
| Client | `scripts/train_tracked.py` | runs the training and sends the logs |

- With only `mlflow.db`, the colleague can see the runs but cannot load the model: the model files are in `mlartifacts/`. They also need the same environment (`uv.lock`).
- An experiment groups several runs on the same problem. A run is one execution.
- The UI does not run the training, it only shows what the script sent to the server.

## 2. What is recorded

Each candidate run has:
- params: model type, hyperparameters (`clf.*`), split seed (42), threshold (0.5)
- metrics: `val_average_precision`, `val_precision`, `val_recall`, `val_f1`
- tags: git commit, uncommitted changes flag, SHA-256 of the CSV and of the split file
- artifacts: validation confusion matrix and `provenance.json`
- datasets: `telco_train` (training) and `telco_validation` (validation)
- the full fitted pipeline (`mlflow.sklearn.log_model`)

Answers:
- Two allocations from the same CSV have the same CSV hash. What changes is the split file hash and the dataset digests.
- A metric is a number we can compare between runs. An artifact is a file.
- The seed does not guarantee the same result on every machine: library versions or hardware can change it.
- No, the hash does not contain the data, it only checks that we have the right file.
- If there are uncommitted changes, the commit does not match the code that ran. We committed before the final runs (`git_uncommitted_changes = false`).

## 3. Comparison

| Run ID | Candidate | Val AP | Val recall | Val F1 | Decision |
|---|---|---:|---:|---:|---|
| `7db43456b782425ea7b2cc3db086b1b0` | forest-300trees-depth8 | **0.664** | 0.484 | 0.556 | Selected |
| `faae1bdf0aba4e7f8141f143fa0c267f` | logistic-C1 | 0.647 | 0.529 | 0.589 | No |
| `bb180cefda2d430783618894572fd04d` | tree-depth5-leaf20 | 0.617 | 0.602 | 0.618 | No |
| `e9b2059c37594370952b4956de435c78` | baseline-majority | 0.265 | 0.000 | 0.000 | Baseline |

We kept the same rule as TP02: best validation AP, unless a simpler model is within 0.005. The random forest is 0.016 above logistic regression, so we keep it. Its recall is lower at 0.5, but AP was our main metric from the start. The results are the same as in TP02.

- With different validation sets, we would not know if the difference comes from the model or from the data.
- If we choose on the test set, its score is no longer a fair estimate.

## 4. Model and registry

| Identifier | Value |
|---|---|
| Run ID | `7db43456b782425ea7b2cc3db086b1b0` |
| Logged model URI | `models:/m-7bac82e966864855815d5c34789d1d28` |
| Registered model | `telco-churn`, version 1 |
| Alias | `candidate` → version 1 |

The input schema has the 6 raw columns (3 `double`, 3 `string`), not the encoded ones.

Inference in a new terminal:

```sh
uv run python scripts/predict_registered.py --input evidence/tp03/inference_batch.csv --reference evidence/tp03/reference_predictions.csv
```

Same predictions on the 20 rows (max difference 8.5e-17), no `fit`.

- In the training notebook the model is still in memory, so it proves nothing. A new process shows the model can really be loaded from MLflow.
- If the alias moves to another version, the model already loaded does not change. We have to load it again.

## 5. Restart

After restarting the server with the same command, we still have the 5 runs, version 1 and the alias. The inference gives the same results.

- Only the database: we see the runs and metrics but cannot load the model.
- Only the model files: we can load the model but we lose the params, metrics and the link to the run.
- A colleague on another machine needs the server (or `mlflow.db` + `mlartifacts/`), the code at the right commit, `uv.lock` and the CSV.
