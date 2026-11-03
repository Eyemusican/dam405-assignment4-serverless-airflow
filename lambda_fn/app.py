"""AWS Lambda handler for breast cancer prediction.

Triggers:
  GET  /health   -> model info
  POST /predict  -> predictions
Also works with a direct invoke where the event is the JSON payload.

Payload: {"features": [30 numbers]} or {"instances": [[30 numbers], ...]}
"""
import base64
import json
import logging
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "model"))

# Runs once per container (cold start), reused by warm invocations
_load_start = time.perf_counter()
MODEL = joblib.load(MODEL_DIR / "model.joblib")
METADATA = json.loads((MODEL_DIR / "metadata.json").read_text())
MODEL_LOAD_MS = round((time.perf_counter() - _load_start) * 1000, 2)
N_FEATURES = METADATA["n_features"]
LABELS = METADATA["label_meaning"]
MAX_ROWS = int(os.environ.get("MAX_ROWS", "100"))

_cold_start = True


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _get_payload(event):
    """Return the JSON payload from an API Gateway event or a direct invoke."""
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if body is None:
            raise ValueError("Request body is empty")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        payload = json.loads(body) if isinstance(body, str) else body
    else:
        payload = event

    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    return payload


def _parse_rows(payload):
    """Validate the input and return a 2D numpy array."""
    if "instances" in payload:
        rows = payload["instances"]
    elif "features" in payload:
        rows = [payload["features"]]
    else:
        raise ValueError("Provide 'features' (one row) or 'instances' (list of rows)")

    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("'instances' must be a non-empty list")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"Too many rows, maximum is {MAX_ROWS}")

    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != N_FEATURES:
            raise ValueError(f"Row {i} must be a list of {N_FEATURES} numbers")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Row {i} contains a non numeric value")
            try:
                number = float(value)
            except OverflowError:
                raise ValueError(f"Row {i} contains a number that is too large")
            if not math.isfinite(number):
                raise ValueError(f"Row {i} contains NaN or infinity")

    return np.array(rows, dtype=float)


def _health():
    return _response(200, {
        "status": "ok",
        "model_version": METADATA["model_version"],
        "trained_at": METADATA["trained_at"],
        "sklearn_version": METADATA["sklearn_version"],
        "n_features": N_FEATURES,
        "metrics": METADATA["metrics"],
    })


def lambda_handler(event, context):
    global _cold_start
    cold_start = _cold_start
    _cold_start = False

    start = time.perf_counter()
    request_id = getattr(context, "aws_request_id", "local")

    if isinstance(event, dict) and event.get("httpMethod") == "GET":
        return _health()

    try:
        payload = _get_payload(event)
        X = _parse_rows(payload)
    except ValueError as err:
        logger.warning(json.dumps({"event": "bad_request", "request_id": request_id, "error": str(err)}))
        return _response(400, {"error": str(err)})

    try:
        preds = MODEL.predict(X)
        probs = MODEL.predict_proba(X)[:, 1]
    except Exception:
        logger.exception(json.dumps({"event": "inference_error", "request_id": request_id}))
        return _response(500, {"error": "Internal error during prediction"})

    predictions = [
        {
            "prediction": int(p),
            "label": LABELS[str(int(p))],
            "malignant_probability": round(float(pr), 4),
        }
        for p, pr in zip(preds, probs)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(json.dumps({
        "event": "prediction",
        "request_id": request_id,
        "model_version": METADATA["model_version"],
        "rows": len(predictions),
        "cold_start": cold_start,
        "latency_ms": latency_ms,
    }))

    return _response(200, {
        "model_version": METADATA["model_version"],
        "predictions": predictions,
        "cold_start": cold_start,
        "model_load_ms": MODEL_LOAD_MS,
        "latency_ms": latency_ms,
    })