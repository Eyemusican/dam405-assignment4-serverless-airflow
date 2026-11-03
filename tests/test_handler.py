"""Unit tests for the Lambda handler (run without Docker)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lambda_fn"))

import app  # noqa: E402

EVENTS = ROOT / "events"


def load_event(name):
    return json.loads((EVENTS / name).read_text())


def body_of(response):
    return json.loads(response["body"])


def test_health_returns_model_info():
    response = app.lambda_handler(load_event("health_api.json"), None)
    body = body_of(response)
    assert response["statusCode"] == 200
    assert body["status"] == "ok"
    assert body["n_features"] == 30


def test_direct_invoke_predicts_both_classes():
    response = app.lambda_handler(load_event("predict_direct.json"), None)
    body = body_of(response)
    assert response["statusCode"] == 200
    labels = [p["label"] for p in body["predictions"]]
    assert labels == ["malignant", "benign"]


def test_api_gateway_event_predicts():
    response = app.lambda_handler(load_event("predict_api.json"), None)
    body = body_of(response)
    assert response["statusCode"] == 200
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0.0 <= p["malignant_probability"] <= 1.0


def test_single_features_row_is_accepted():
    row = load_event("predict_direct.json")["instances"][0]
    response = app.lambda_handler({"features": row}, None)
    assert response["statusCode"] == 200
    assert len(body_of(response)["predictions"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [1.0, 2.0, 3.0]},           # wrong number of features
        {"features": ["a"] * 30},                # non numeric values
        {"features": [True] * 30},               # booleans are not numbers
        {"features": [float("nan")] * 30},       # NaN
        {"instances": []},                       # empty list
        {"wrong_key": [1.0] * 30},               # missing key
        {"instances": [[1.0] * 30] * 101},       # more than MAX_ROWS
    ],
)
def test_invalid_payloads_return_400(payload):
    response = app.lambda_handler(payload, None)
    assert response["statusCode"] == 400
    assert "error" in body_of(response)


def test_invalid_json_body_returns_400():
    event = {"httpMethod": "POST", "path": "/predict", "body": "not json"}
    response = app.lambda_handler(event, None)
    assert response["statusCode"] == 400


def test_empty_body_returns_400():
    event = {"httpMethod": "POST", "path": "/predict", "body": None}
    response = app.lambda_handler(event, None)
    assert response["statusCode"] == 400


def test_huge_integer_returns_400():
    payload = {"features": [10 ** 400] + [1.0] * 29}
    response = app.lambda_handler(payload, None)
    assert response["statusCode"] == 400
    assert "too large" in body_of(response)["error"]