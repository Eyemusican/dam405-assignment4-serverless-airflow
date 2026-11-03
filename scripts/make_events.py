"""Create sample Lambda events from real dataset rows."""
import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer

EVENTS_DIR = Path(__file__).resolve().parent.parent / "events"


def api_event(method, path, body=None):
    """Minimal API Gateway (REST, proxy) event."""
    return {
        "httpMethod": method,
        "path": path,
        "headers": {"Content-Type": "application/json"},
        "queryStringParameters": None,
        "isBase64Encoded": False,
        "body": json.dumps(body) if body is not None else None,
    }


def main():
    data = load_breast_cancer()
    y = 1 - data.target  # 1 = malignant

    malignant_idx = int((y == 1).nonzero()[0][0])
    benign_idx = int((y == 0).nonzero()[0][0])
    malignant_row = [round(float(v), 6) for v in data.data[malignant_idx]]
    benign_row = [round(float(v), 6) for v in data.data[benign_idx]]

    payload = {"instances": [malignant_row, benign_row]}

    events = {
        "predict_direct.json": payload,
        "predict_api.json": api_event("POST", "/predict", payload),
        "health_api.json": api_event("GET", "/health"),
        "bad_request_api.json": api_event("POST", "/predict", {"features": [1.0, 2.0, 3.0]}),
    }

    EVENTS_DIR.mkdir(exist_ok=True)
    for name, event in events.items():
        (EVENTS_DIR / name).write_text(json.dumps(event, indent=2))
        print(f"Wrote events/{name}")

    print(f"Malignant example is dataset row {malignant_idx}, benign example is row {benign_idx}")


if __name__ == "__main__":
    main()