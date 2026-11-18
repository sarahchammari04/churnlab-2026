"""First connection check (TP03 section 1): one small run with one parameter.

uv run python scripts/tracking_smoke.py
"""

import mlflow

from churnlab.tracking import EXPERIMENT_NAME, TRACKING_URI, connect

experiment = connect(TRACKING_URI, EXPERIMENT_NAME)
with mlflow.start_run(run_name="smoke-test") as run:
    mlflow.log_param("purpose", "connection check")
    mlflow.set_tag("run_role", "smoke-test")  # kept apart from candidate runs
print(f"Experiment: {experiment.name} (id {experiment.experiment_id})")
print(f"Run ID: {run.info.run_id}")
