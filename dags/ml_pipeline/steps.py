"""Pipeline step functions used by the breast cancer DAG.

Each function does one job (atomic). Inputs and outputs are files inside a
run folder, so running a step again for the same run replaces the same files
(idempotent). There is no Airflow code here, so the steps can be tested alone.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "target"
LABEL_MEANING = {"0": "benign", "1": "malignant"}


class ModelQualityError(Exception):
    """Raised when a trained model is not good enough to deploy."""


# ---------- helpers for atomic writes ----------

def _tmp_path(path):
    return path.with_name(path.name + ".tmp")


def write_csv_atomic(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_json_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def dump_model_atomic(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    joblib.dump(model, tmp)
    os.replace(tmp, path)


def feature_columns():
    return list(load_breast_cancer().feature_names)


# ---------- pipeline steps ----------

def extract(run_dir, inject_bad_data=False):
    """Load the raw dataset and save it as raw.csv (1 = malignant)."""
    df = load_breast_cancer(as_frame=True).frame.copy()
    df[TARGET] = 1 - df[TARGET]

    if inject_bad_data:
        # Deliberate faults so the quality gate has something to catch
        df.loc[0:9, "mean radius"] = np.nan   # missing values
        df.loc[10:14, "mean texture"] = -5.0  # impossible negative values
        df.loc[15, TARGET] = 2                # label outside {0, 1}

    path = Path(run_dir) / "raw.csv"
    write_csv_atomic(df, path)
    return str(path)


def validate(raw_path, run_dir, min_rows):
    """Great Expectations data quality gate. Returns a summary dict."""
    import great_expectations as gx

    df = pd.read_csv(raw_path)
    features = feature_columns()

    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="pipeline_source")
    asset = source.add_dataframe_asset(name="raw_data")
    batch_definition = asset.add_batch_definition_whole_dataframe("raw_batch")

    suite = context.suites.add(gx.ExpectationSuite(name="raw_data_quality"))
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=min_rows))
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=features + [TARGET], exact_match=True)
    )
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1]))
    for col in features:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0))

    validation = context.validation_definitions.add(
        gx.ValidationDefinition(name="raw_data_gate", data=batch_definition, suite=suite)
    )
    result = validation.run(batch_parameters={"dataframe": df})

    failed_checks = [
        {
            "expectation": r.expectation_config.type,
            "column": r.expectation_config.kwargs.get("column"),
            "unexpected_count": (r.result or {}).get("unexpected_count"),
        }
        for r in result.results
        if not r.success
    ]
    summary = {
        "success": bool(result.success),
        "rows": int(len(df)),
        "checks_evaluated": len(result.results),
        "checks_failed": len(failed_checks),
        "failed_checks": failed_checks,
    }
    write_json_atomic(summary, Path(run_dir) / "validation_report.json")
    return summary


def transform(raw_path, run_dir, test_size, random_state):
    """Stratified train/test split. Scaling happens inside the model pipeline
    so test data never leaks into the scaler."""
    df = pd.read_csv(raw_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    train_path = Path(run_dir) / "train.csv"
    test_path = Path(run_dir) / "test.csv"
    write_csv_atomic(train_df, train_path)
    write_csv_atomic(test_df, test_path)
    return {"train_path": str(train_path), "test_path": str(test_path),
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df))}


def train(train_path, test_path, run_dir, random_state):
    """Train StandardScaler + LogisticRegression and evaluate on the test set."""
    features = feature_columns()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Plain arrays, because Lambda sends plain numbers (no column names)
    X_train, y_train = train_df[features].to_numpy(), train_df[TARGET].to_numpy()
    X_test, y_test = test_df[features].to_numpy(), test_df[TARGET].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
    }
    model_path = Path(run_dir) / "model.joblib"
    dump_model_atomic(model, model_path)
    write_json_atomic(metrics, Path(run_dir) / "metrics.json")
    return {"model_path": str(model_path), "metrics": metrics}


def deploy(model_path, metrics, model_dir, min_recall, model_version):
    """Second gate: only publish the model to the Lambda folder if recall is good enough."""
    if metrics["recall"] < min_recall:
        raise ModelQualityError(
            f"Recall {metrics['recall']} is below the minimum {min_recall}, model not deployed"
        )

    model = joblib.load(model_path)
    features = feature_columns()
    metadata = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_by": "airflow dag breast_cancer_ml_pipeline",
        "sklearn_version": sklearn.__version__,
        "n_features": len(features),
        "feature_names": features,
        "label_meaning": LABEL_MEANING,
        "metrics": metrics,
    }
    dump_model_atomic(model, Path(model_dir) / "model.joblib")
    write_json_atomic(metadata, Path(model_dir) / "metadata.json")
    return {"model_version": model_version, "model_dir": str(model_dir), "metrics": metrics}