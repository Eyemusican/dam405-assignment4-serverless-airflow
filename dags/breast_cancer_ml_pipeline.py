"""Breast cancer ML pipeline orchestrated by Airflow.

extract -> validate (Great Expectations gate) -> transform -> train -> deploy

Only small values (file paths, metrics) pass between tasks through XCom.
The data itself stays in files inside data/runs/<run_id>/.
"""
import logging
import re
from datetime import datetime, timedelta

from airflow.sdk import Param, Variable, dag, get_current_context, task

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # older import path
    from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)

# Safe defaults; real values come from the Airflow Variable "ml_pipeline_config"
DEFAULT_CONFIG = {
    "data_dir": "/opt/airflow/data",
    "model_dir": "/opt/airflow/lambda_fn/model",
    "min_rows": 500,
    "min_recall": 0.9,
    "test_size": 0.2,
    "random_state": 42,
}


def get_config():
    config = dict(DEFAULT_CONFIG)
    config.update(Variable.get("ml_pipeline_config", default={}, deserialize_json=True))
    return config


def run_folder(config):
    """One folder per DAG run, so reruns overwrite the same files."""
    run_id = get_current_context()["run_id"]
    safe_run_id = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
    return f"{config['data_dir']}/runs/{safe_run_id}", safe_run_id


@dag(
    dag_id="breast_cancer_ml_pipeline",
    description="Extract, validate with Great Expectations, transform, train and deploy to Lambda",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args={"owner": "tenzin", "retries": 1, "retry_delay": timedelta(seconds=30)},
    params={
        "inject_bad_data": Param(
            False, type="boolean", description="Corrupt the extracted data to test the quality gate"
        )
    },
    tags=["dam405", "mlops", "assignment4"],
)
def breast_cancer_ml_pipeline():

    @task
    def extract() -> str:
        from ml_pipeline import steps

        config = get_config()
        folder, _ = run_folder(config)
        inject = bool(get_current_context()["params"]["inject_bad_data"])
        raw_path = steps.extract(folder, inject_bad_data=inject)
        log.info("Extracted raw data to %s (inject_bad_data=%s)", raw_path, inject)
        return raw_path

    @task(retries=0)
    def validate(raw_path: str) -> str:
        from ml_pipeline import steps

        config = get_config()
        folder, _ = run_folder(config)
        summary = steps.validate(raw_path, folder, config["min_rows"])
        log.info("Quality gate summary: %s", summary)
        if not summary["success"]:
            raise AirflowFailException(
                f"Data quality gate FAILED: {summary['checks_failed']} of "
                f"{summary['checks_evaluated']} checks failed: {summary['failed_checks']}"
            )
        log.info("Data quality gate PASSED: all %s checks passed", summary["checks_evaluated"])
        return raw_path

    @task
    def transform(raw_path: str) -> dict:
        from ml_pipeline import steps

        config = get_config()
        folder, _ = run_folder(config)
        split = steps.transform(raw_path, folder, config["test_size"], config["random_state"])
        log.info("Split data: %s", split)
        return split

    @task
    def train(split: dict) -> dict:
        from ml_pipeline import steps

        config = get_config()
        folder, _ = run_folder(config)
        result = steps.train(split["train_path"], split["test_path"], folder, config["random_state"])
        log.info("Trained model, test metrics: %s", result["metrics"])
        return result

    @task(retries=0)
    def deploy(result: dict) -> dict:
        from ml_pipeline import steps

        config = get_config()
        _, safe_run_id = run_folder(config)
        try:
            deployed = steps.deploy(
                result["model_path"],
                result["metrics"],
                config["model_dir"],
                config["min_recall"],
                model_version=f"v2-airflow-{safe_run_id}",
            )
        except steps.ModelQualityError as err:
            raise AirflowFailException(str(err))
        log.info("Deployed model %s to %s", deployed["model_version"], deployed["model_dir"])
        return deployed

    raw = extract()
    checked = validate(raw)
    split = transform(checked)
    trained = train(split)
    deploy(trained)


breast_cancer_ml_pipeline()