"""Train the first model for the Lambda function.

Dataset: Breast Cancer Wisconsin (labels flipped so 1 = malignant).
Output:  lambda_fn/model/model.joblib and lambda_fn/model/metadata.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "lambda_fn" / "model"
RANDOM_STATE = 42


def main():
    data = load_breast_cancer(as_frame=True)
    feature_names = list(data.feature_names)
    X = data.data.to_numpy()
    y = 1 - data.target  # flip labels so 1 = malignant

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    metadata = {
        "model_version": "v1-initial",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_by": "scripts/train_model.py",
        "sklearn_version": sklearn.__version__,
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "label_meaning": {"0": "benign", "1": "malignant"},
        "metrics": metrics,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Model saved to {MODEL_DIR / 'model.joblib'}")
    print(f"sklearn version: {sklearn.__version__}")
    print(f"Test metrics: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()