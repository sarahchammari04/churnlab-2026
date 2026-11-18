# TP04 summary

**Students:** Sarah Chammari and Shaima Kraiem
**Tools:** pytest 8.3.5, pytest-cov 7.1.0, Ruff 0.17.5, pre-commit 4.7.0
**Answers to the questions:** `evidence/tp04/answers.md`

## 1. Contract

- The six TP02 features are required. `Churn` and `customerID` are never model inputs.
- Numbers: blank → NaN (imputed), any other non-numeric text → error.
- Missing values are imputed, unknown categories are encoded as zeros (`handle_unknown="ignore"`).
- One output row per customer, in the same order.
- Class order `[0, 1]`, 1 = Churn Yes. The positive score is taken with `classes_`.
- A saved and reloaded pipeline gives the same predictions.

Four defects the tests should catch:
1. A column is missing and the model still predicts (or crashes with an unclear error).
2. Someone changes the encoder and an unknown category makes the prediction crash.
3. The code uses `proba[:, 1]` without checking `classes_`.
4. Only the classifier is saved, without the preprocessing.

## 2. Tests

Fixture: 16 invented customers with both classes (`tests/conftest.py`). The pipeline fixture uses our real `build_pipeline` and runs for a logistic regression and a random forest. No CSV, no network, no MLflow.

| Test | Risk it covers |
|---|---|
| A. `test_missing_required_feature_is_rejected` (6 columns × 2 models) | A missing column is not detected |
| B. `test_missing_numeric_and_unknown_category_are_accepted` | Blank number or new category breaks prediction |
| C. `test_probability_contract` + `test_positive_score_uses_classes_not_column_position` | Invalid probabilities, or the wrong class column is used |
| D. `test_reload_preserves_predictions` + `test_saved_artifact_contains_fitted_preprocessing` | Saving loses the preprocessing or changes the predictions |
| Extra | Malformed number rejected, `Churn`/`customerID` ignored, threshold rule, row order |

Result: `34 passed` (`uv run pytest -q`).

## 3. Regression

We changed the encoder in `src/churnlab/pipeline.py` from `handle_unknown="ignore"` to `handle_unknown="error"` (`evidence/tp04/regression_diff.txt`).

Test B failed for both models (`evidence/tp04/regression_failure.txt`):

```
ValueError: Found unknown categories ['Satellite'] in column 1 during transform
FAILED tests/test_input_contract.py::test_missing_numeric_and_unknown_category_are_accepted[forest]
FAILED tests/test_input_contract.py::test_missing_numeric_and_unknown_category_are_accepted[logistic]
2 failed, 17 passed
```

We restored `"ignore"` and the full suite passed again: `34 passed` (`evidence/tp04/regression_fixed.txt`).

## 4. Ruff

Rules `E4, E7, E9, F, I`, line length 100, on `src scripts tests`.

- Before: 9 errors (3 unsorted imports, 6 lines with `;` in `scripts/checks.py`) and 14 files not formatted (`evidence/tp04/ruff_check_before.txt`).
- We split the `;` lines by hand, used `ruff check --fix` for the imports and `ruff format`, then checked the diff and reran the tests.
- After: `All checks passed!` and `24 files already formatted`.

## 5. Coverage

```
src/churnlab/data.py          41      7    83%   34, 39-42, 76-77
src/churnlab/evaluate.py      12      4    67%   24-27
src/churnlab/pipeline.py      12      0   100%
src/churnlab/predict.py        8      0   100%
src/churnlab/split.py         14     14     0%   3-34
src/churnlab/tracking.py      72     72     0%   12-183
src/churnlab/train.py         92     92     0%   3-236
TOTAL                        252    189    25%
```

(full output: `evidence/tp04/coverage.txt`)

25% is low because `train.py` and `tracking.py` run the full workflow (CSV, MLflow) and are not unit-tested. The inference path (`pipeline`, `predict`) is fully covered.

Uncovered behavior: `load_raw` in `data.py` (lines 39-42) should refuse a CSV whose hash is wrong. Proposed test: write a small fake CSV in `tmp_path`, call `load_raw(path, expected_sha256="0" * 64)` and check `pytest.raises(ValueError, match="Source hash mismatch")`.

## 6. pre-commit

`.pre-commit-config.yaml`: `ruff-check --fix` then `ruff-format` (ruff-pre-commit `v0.17.5`, same version as the CLI), plus trailing whitespace, end of file, YAML and TOML checks (pre-commit-hooks `v6.0.0`). Limited to `src/`, `scripts/`, `tests/` and config files.

`pre-commit run --all-files`: all hooks passed.

Blocked commit: we added `tests/test_persistence.py` with a badly formatted line (`artifact=tmp_path/'pipeline.joblib'` + trailing spaces).

```
ruff format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook
exit code: 1
```

We checked the change with `git diff`, staged it again and retried: all hooks passed and the commit was created (`01c65d0`). Logs: `evidence/tp04/precommit_blocked.txt` and `precommit_retry.txt`.

## 7. Quality Gate

SonarQube was not run (optional). What we keep from it:

- Tests check behavior. Ruff checks the code without running it. Coverage only says which lines were executed. A Quality Gate is a rule (for example "coverage on new code > 80%") applied to these results.
- The analysis can succeed and the gate still fail: they are two different results.
- Passing tests or a passing gate do not prove the model is good, fair or secure. They only check what we wrote tests and rules for.
