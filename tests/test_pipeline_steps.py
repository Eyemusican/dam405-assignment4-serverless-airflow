"""Unit tests for the pipeline step functions (run without Airflow)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dags"))

from ml_pipeline import steps  # noqa: E402


def test_extract_writes_clean_data(tmp_path):
    raw_path = steps.extract(tmp_path)
    df = pd.read_csv(raw_path)
    assert len(df) == 569
    assert set(df[steps.TARGET].unique()) == {0, 1}
    assert df.isna().sum().sum() == 0


def test_extract_is_idempotent(tmp_path):
    first = Path(steps.extract(tmp_path)).read_bytes()
    second = Path(steps.extract(tmp_path)).read_bytes()
    assert first == second
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw.csv"]


def test_inject_bad_data_adds_faults(tmp_path):
    df = pd.read_csv(steps.extract(tmp_path, inject_bad_data=True))
    assert df["mean radius"].isna().sum() == 10
    assert (df["mean texture"] < 0).sum() == 5
    assert (df[steps.TARGET] == 2).sum() == 1


def test_transform_makes_stratified_split(tmp_path):
    raw_path = steps.extract(tmp_path)
    split = steps.transform(raw_path, tmp_path, test_size=0.2, random_state=42)
    assert split["train_rows"] + split["test_rows"] == 569
    assert split["test_rows"] == 114


def test_train_and_deploy(tmp_path):
    raw_path = steps.extract(tmp_path)
    split = steps.transform(raw_path, tmp_path, 0.2, 42)
    result = steps.train(split["train_path"], split["test_path"], tmp_path, 42)
    assert result["metrics"]["recall"] >= 0.9

    model_dir = tmp_path / "lambda_model"
    deployed = steps.deploy(result["model_path"], result["metrics"], model_dir, 0.9, "v-test")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    assert deployed["model_version"] == "v-test"
    assert metadata["n_features"] == 30
    assert (model_dir / "model.joblib").exists()
    assert not list(model_dir.glob("*.tmp"))  # atomic writes leave no temp files


def test_deploy_gate_blocks_weak_model(tmp_path):
    with pytest.raises(steps.ModelQualityError):
        steps.deploy("unused.joblib", {"recall": 0.5}, tmp_path, 0.9, "v-bad")
    assert not (tmp_path / "model.joblib").exists()


def test_validate_gate(tmp_path):
    pytest.importorskip("great_expectations")
    good = steps.validate(steps.extract(tmp_path / "good"), tmp_path / "good", 500)
    bad = steps.validate(steps.extract(tmp_path / "bad", True), tmp_path / "bad", 500)
    assert good["success"] is True
    assert bad["success"] is False
    assert bad["checks_failed"] == 3