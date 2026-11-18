# TP04 answers

**Student:** Sarah Chammari and Shaima Kraiem
**Commands:** `uv run pytest -q`, `uv run ruff check src scripts tests`, `uv run pre-commit run --all-files`
**Summary:** `reports/tp04_summary.md`

## 1. Contract

Four realistic defects:
- a required column is missing and nobody notices;
- an unknown category makes the prediction crash after a change in the encoder;
- the positive score is taken with `proba[:, 1]` without checking `classes_`;
- only the classifier is saved, without the fitted preprocessing.

"The program runs" is not enough: all four can happen while the program still runs.

## 2. Tests

- Test A: we remove each required column in turn and expect a `KeyError` with "Missing required feature columns". We catch only `KeyError`, not `Exception`.
- Test B: blank `TotalCharges` and `InternetService = "Satellite"`. We first check that "Satellite" is not in the training data.
- Test C: shape `(16, 2)`, values between 0 and 1, rows sum to 1, and the positive column comes from `classes_`. A fake model with `classes_ = [1, 0]` checks that the code does not just take column 1.
- Test D: save with joblib in `tmp_path`, reload, same classes and same probabilities.

**Does a prediction on an unknown category prove it is accurate?** No. It only proves it does not crash. The new category is encoded as zeros, so the model predicts as if it had no information on that column.

## 3. Regression

We changed `handle_unknown="ignore"` to `"error"`. Test B failed (`ValueError: Found unknown categories ['Satellite']`). We put `"ignore"` back and the 34 tests passed.

## 4. Ruff and coverage

- E4/E7/E9 are import, statement and syntax errors, F is Pyflakes (unused imports, undefined names), I is import sorting. The formatter is separate: it only changes the layout.
- Coverage is 25%. `train.py` and `tracking.py` are not covered because they need the real CSV or MLflow.

**What does an executed line tell you?** Only that it ran during a test, not that its result was checked.

**Can a test execute a line without asserting the right behavior?** Yes. A test that calls `predict` without any `assert` covers the lines but checks nothing.

**Which third-party code is outside the coverage scope?** scikit-learn, pandas, NumPy, joblib and MLflow. We measure only our package `churnlab`.

## 5. pre-commit

- The config is in Git, but `pre-commit install` is local: a classmate must run it after cloning.
- The hooks can be skipped with `--no-verify`, so they do not replace CI.
- We did not put pytest or training in the hooks, so commits stay fast.

## 6. Quality Gate

**If tests pass, what can still be wrong?** Anything we did not test: the model can be bad on real data, the data can change, the threshold can be wrong for the business.

**If a Quality Gate passes, does it prove the model is useful, fair or secure?** No. It only checks the rules it was given (issues, coverage, duplication). It says nothing about model quality or fairness.
