# TP02 answers

**Student:** Sarah Chammari
**Entry point:** `uv run python scripts/train.py` (seed 42)
**Code commit used for the artifact:** see `code.commit` in `model_metadata.json`

## 1. First model (validation, positive class = Churn Yes = 1)

Features: `tenure`, `MonthlyCharges`. Confusion matrix layout: rows = true [0, 1], columns = predicted [0, 1].

| Model | Accuracy | Recall (churn) | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.735 | 0.000 | 1035 | 0 | 374 | 0 |
| Logistic regression | 0.783 | 0.409 | 950 | 85 | 221 | 153 |

- A system that always predicts "No churn" reaches 73.5% accuracy and finds zero churners. High accuracy and a useless model can coexist.
- The baseline is fitted on the **training** set, like every model. Only training data influences fitting; only validation data influences model selection; test data influences nothing.

## 2. Pipeline

- `fit` learns parameters from data (medians, means, categories, model weights). `transform` applies the learned parameters without changing them. `predict` transforms then applies the classifier.
- Fitting a scaler on validation data leaks validation information into the model and makes the validation score optimistic.
- A Pipeline does not prevent leakage by itself: if it is fitted on the wrong data (for example the full dataset), it learns from it. It only guarantees that the same steps are applied in the same order.
- `Pipeline` orders operations (preprocessing then classifier). `ColumnTransformer` routes columns (numeric columns to the numeric branch, categorical columns to the categorical branch).
- `pd.to_numeric(errors="coerce")` silently turns any bad text into NaN; `errors="raise"` fails. I first convert only whitespace-only values to NaN, then use `errors="raise"`, so malformed values are detected instead of hidden.

## 3. Comparison (validation, six features, same split)

| Model | Avg precision | Precision | Recall | F1 | Accuracy | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 | ~0.01 |
| Logistic regression | 0.647 | 0.664 | 0.529 | 0.589 | 0.804 | ~0.3 |
| Decision tree (depth 5) | 0.617 | 0.636 | 0.602 | 0.618 | 0.803 | ~0.02 |
| Random forest (300 trees, depth 8) | **0.664** | 0.653 | 0.484 | 0.556 | 0.795 | ~0.9 |

Full table with confusion matrices and configurations: `comparison.csv`.

### Selection note

The criterion was fixed before running the comparison: highest validation average precision, keeping a simpler model if it is within 0.005 of the best. The random forest has the best average precision (0.664), 0.017 above logistic regression, which is more than the margin, so the random forest is selected. The decision was made on validation only; the test set was not used.

Remarks:
- At threshold 0.5 the random forest has a lower recall (0.484) than the decision tree (0.602). Average precision measures ranking quality; the threshold is a separate business decision that could be tuned later on validation.
- The gain over the first model cannot be attributed to the algorithm alone: the feature set also changed (2 to 6 features).
- Logistic regression is close and simpler and more interpretable; with a tighter margin rule it would have been a reasonable choice.
- A score of 0.8 is not automatically a calibrated 80% risk: probabilities were not calibrated.
- Changing the split for each model would mix the effect of the model with the effect of the data, making the comparison impossible to interpret.

## 4. Artifact

- Saving only the classifier is insufficient: it expects already imputed, scaled and one-hot encoded columns. Without the fitted preprocessing, nobody can rebuild exactly the same inputs.
- Fitted information in the preprocessing: imputer medians and most frequent categories, scaler means and standard deviations, the list of categories seen by the encoder.
- Two files called `model.joblib` can contain different models, which is why the metadata records the artifact SHA-256, the data hash, the split, the parameters and the code commit.
- A sklearn Pipeline is a single in-memory object that chains transformations in one process. A workflow DAG such as Kubeflow Pipelines orchestrates separate steps (data loading, training, evaluation, deployment) that can run in different containers and machines.

## 5. Reliability checks (`checks.json`)

| Check | Result |
|---|---|
| Customer IDs disjoint across train/valid/test | 0 overlap, 7,043 customers allocated once |
| Imputer medians = training medians (not full dataset) | TotalCharges: 1373.625 (train) vs 1397.475 (full): matches train |
| Prediction does not refit | imputer and scaler statistics unchanged |
| Blank `TotalCharges` | imputed with the training median, prediction produced |
| Unseen category `Satellite` | encoded as zeros, prediction produced (not reliable) |
| Malformed numeric text `"abc"` | ValueError |
| Missing `Contract` column | KeyError at the boundary, ValueError in the pipeline |
| Fresh-process reload (`scripts/predict.py`) | 20 rows, max score difference 0.0 (tolerance 1e-9), no fit call, no target column |

## Peer review

- Reviewer: <NAME>
- Issue found: <ISSUE>
- Resolution: <RESOLUTION>
