# TP03 answers

**Student:** Sarah Chammari and Shaima Kraiem
**Entry points:** `uv run python scripts/train_tracked.py` (tracked training) · `uv run python scripts/predict_registered.py` (inference only)
**Code commit used for the runs:** d51e0ff (no uncommitted changes at run time)
**MLflow:** 3.18.0, local server `http://127.0.0.1:5000`, experiment `telco-churn-tp03`
**Summary with screenshots:** `reports/tp03_summary.md`

## 1. Local tracking setup

Server command, from the project root:

```sh
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

First run (`scripts/tracking_smoke.py`): experiment `telco-churn-tp03` (id 1), run `smoke-test` `8ab3e683638242a18f9a5cca7bb3b55b`, one parameter `purpose`.

| Component | Location in this project | What it holds |
|---|---|---|
| Backend store | `mlflow.db` (SQLite) | experiments, runs, parameters, metrics, tags, dataset records, registered models, versions, aliases |
| Artifact store | `mlartifacts/` (`1/<run_id>/artifacts/`, `1/models/m-<id>/artifacts/`) | `MLmodel`, `model.pkl`, `requirements.txt`, `python_env.yaml`, input example, confusion matrices, `provenance.json` |
| Tracking server | `mlflow server` process | receives logging and query requests, serves the UI |
| Training client | `scripts/train_tracked.py` | runs the training code and sends records to the server |

**If a colleague receives only `mlflow.db`, what else would they need to load the pipeline?** The database only contains metadata and references to artifact locations. They also need the model files in `mlartifacts/` (or access to the same server), a compatible environment (`uv.lock`: scikit-learn 1.5.2, MLflow 3.18.0, Python 3.11), and our package code if the pipeline used custom transformers (ours uses only scikit-learn classes).

**Experiment vs run.** An experiment is a named group of comparable runs on the same problem and protocol (`telco-churn-tp03`). A run is one execution, with its own parameters, metrics, tags, artifacts and dataset inputs.

**Is the browser page executing the training code?** No. The training runs in the client Python process (terminal). The client sends records to the server; the UI only displays what is stored.

## 2. What each run records

| Category | Recorded |
|---|---|
| Parameters | `model_key`, `model_type`, all actual classifier settings (`clf.*`, from `get_params()`), `split_seed` = 42, `threshold` = 0.5, `selection_metric` |
| Metrics | `val_average_precision`, `val_precision`, `val_recall`, `val_f1`, `val_accuracy`, `fit_seconds` |
| Tags | `git_commit`, `git_uncommitted_changes`, `data_sha256`, `split_allocation_sha256`, `run_role` |
| Artifacts | `evaluation/val_confusion_matrix.png` (rows = true [0, 1], cols = predicted [0, 1]), `provenance.json` |
| Dataset inputs | `telco_train` (context `training`, 4,225 rows, digest `6834438a`), `telco_validation` (context `validation`, 1,409 rows, digest `4fb2e5e6`) |
| Model | `churn_pipeline`: full fitted preprocessing + classifier, `mlflow.sklearn.log_model`, logged in every candidate run |

The class-1 score column is found from `classes_`. AP uses the continuous class-1 scores; precision, recall and F1 use the fixed threshold 0.5.

**What evidence distinguishes two train/validation allocations made from the same CSV?** The CSV SHA-256 is the same in both cases, so it cannot. What differs: the `customerID → split` allocation file and its SHA-256 (`split_allocation_sha256`), the seed and ratios recorded in parameters and `provenance.json`, and the MLflow digests of the training and validation dataset records (different rows give different digests). The training script also refuses to run if the recomputed allocation differs from `evidence/tp02/split_allocation.csv`.

**Metric vs artifact.** A metric is a numeric value stored in the backend database, which can be sorted, compared across runs and plotted. An artifact is a file (image, JSON, model) stored in the artifact store; MLflow keeps its path, not its content, in the database.

**Does recording a random seed guarantee identical results on every machine?** No. It fixes the random draws of one library version, but results can still change with other versions of scikit-learn or NumPy, BLAS, the OS, the CPU or parallelism. It makes results reproducible in the same environment, which is why the environment is also recorded.

**Can someone recover the training data from its hash alone?** No. A hash identifies bytes; it does not contain them. It only lets someone check that a file they already have is the right one.

**What happens to traceability if a commit is logged while training uncommitted edits?** The commit no longer describes the code that actually ran, so the result cannot be reproduced from it. For this reason the runs record `git_uncommitted_changes`, and the code was committed (d51e0ff) before the final runs: the tag is `false` on every candidate run.

## 3. Candidate comparison

Same six features, same TP02 allocation, same metrics, same threshold.

| Run ID | Candidate / setting | Validation AP | Validation recall | Validation F1 | Decision |
|---|---|---:|---:|---:|---|
| `7db43456b782425ea7b2cc3db086b1b0` | `forest-300trees-depth8` | **0.6639** | 0.4840 | 0.5561 | Selected |
| `faae1bdf0aba4e7f8141f143fa0c267f` | `logistic-C1` | 0.6475 | 0.5294 | 0.5893 | Not selected |
| `bb180cefda2d430783618894572fd04d` | `tree-depth5-leaf20` | 0.6173 | 0.6016 | 0.6181 | Not selected |
| `e9b2059c37594370952b4956de435c78` | `baseline-majority` | 0.2654 | 0.0000 | 0.0000 | Reference only |

### Selection note

The rule was fixed in TP02 before seeing any result: highest validation average precision, and a simpler model is kept if it is within 0.005 of the best. The random forest has the best AP (0.6639); logistic regression is 0.0164 below, more than the margin, so the random forest is selected. Its recall and F1 at threshold 0.5 are lower, but they are supporting metrics, and the threshold was not changed after seeing the results. The tracked metrics are identical to TP02, which confirms that adding tracking did not change the experiment. With one validation split, the gap is not proof of a statistically significant improvement.

**Why are results from different validation populations harder to compare?** The score depends on the model and on the sample. With different validation customers, a gap could come from easier or harder customers rather than from a better model, so the comparison is no longer fair.

**Why avoid selecting repeatedly on the test set?** Each choice based on the test set adapts the model to it. The test score then becomes optimistic and no longer estimates performance on new customers. The test set stays reserved for one final evaluation.

## 4. Logged, registered and recovered model

| Identifier | Value | Question it answers |
|---|---|---|
| Run ID | `7db43456b782425ea7b2cc3db086b1b0` | Which execution produced the result? |
| Logged model URI | `models:/m-7bac82e966864855815d5c34789d1d28` | Which recorded model artifact should be loaded? |
| Registered name + version | `telco-churn` version 1 | Which version belongs to the managed model? |
| Alias | `candidate` → version 1 | Which version is currently selected for this role? |

Registration and alias were done through the API (`mlflow.register_model`, `MlflowClient.set_registered_model_alias`) in `scripts/train_tracked.py`. The returned version (1) was recorded in `handover.json`, not assumed.

**Stored input schema.** Inferred from `X_train.head(3)`: `tenure`, `MonthlyCharges`, `TotalCharges` as `double`; `Contract`, `InternetService`, `PaymentMethod` as `string`; output `long`. It describes the raw columns the pipeline expects, not the internal one-hot encoding. The schema marks every column as required, while our boundary rules allow a blank `TotalCharges` (it becomes NaN and is imputed). This only matters for pyfunc schema enforcement: `mlflow.sklearn.load_model` returns the plain scikit-learn pipeline.

**Fresh-process inference:**

```sh
uv run python scripts/predict_registered.py --input evidence/tp03/inference_batch.csv --reference evidence/tp03/reference_predictions.csv
```

Result: `models:/telco-churn@candidate -> version 1`, 20 rows, classes agree, max score difference 8.5e-17 (tolerance 1e-9). The script does not call `fit` and imports no training code.

**Why is loading in a new process stronger evidence than predicting again in the training notebook?** In the training session the fitted object is still in memory, with all imports and variables available. Success there proves nothing about storage. A new process can only use what was actually recorded, so success proves that the model can be recovered from the registry without retraining.

**If the alias later points to a new version, does an already-loaded Python object change?** No. `load_model` returns an in-memory copy of the version the alias pointed to at load time. To use the new version, the model must be loaded again (for example by restarting the service). This is why the version number is recorded next to the alias.

## 5. Persistence and handover

**Restart observation.** The server was stopped (no answer on `/health`; `mlflow.db` and `mlartifacts/` still on disk) and restarted from the project root with the same command. Before and after: 5 runs in `telco-churn-tp03`, `telco-churn` versions [1], `candidate` → version 1 (`status_before_restart.txt` and `status_after_restart.txt` are identical). The inference command gave the same predictions with `AGREEMENT OK`.

**What would be unavailable if the database survived but the model files were lost?** Runs, parameters, metrics, tags, versions and the alias would still be visible, but loading the model would fail, and the confusion matrices and `provenance.json` would be missing.

**What would be missing if only the model files survived?** A model folder could still be loaded directly from its path, but its run, parameters, metrics, tags, dataset records, the link between run and model, the registered versions and the alias would be lost. Nothing would say which model was selected or why.

**What does a colleague on another machine need?** Our run IDs and `127.0.0.1` URIs are useless on their machine. They need a shared tracking server or a copy of both `mlflow.db` and `mlartifacts/`, this repository at commit d51e0ff, the `uv.lock` environment, and the original CSV, checked against its SHA-256 (`16320c9c…3e055e91`).

## Limitations

- Local server only: sharing requires a team server or exporting both stores.
- The alias is a mutable pointer, not a deployment and not a copy.
- Dataset records describe the data (name, digest, schema); they do not store it and are not dataset version control.
- The model is serialized with cloudpickle: load only trusted, self-produced artifacts.
- Probabilities are not calibrated, and the comparison uses a single validation split.
