"""
Serving Pipeline.

Real-time anomaly detection: fetches online features from Feast,
runs inference through the trained autoencoder loaded from MLflow.

Exposes a simple HTTP API via Flask for scoring new transactions.
"""

import sys
from pathlib import Path

# Add project root to path so torch can find 'model.autoencoder' when unpickling
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request

from feast import FeatureStore

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_REPO = PROJECT_ROOT / "feature_repo"
MLFLOW_TRACKING_URI = "postgresql+psycopg2://mlops:mlops_secret@localhost:5433/mlflow"

# MinIO as S3-compatible artifact store
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9200"
os.environ["AWS_ACCESS_KEY_ID"] = "mlops_minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mlops_minio_secret"
ARTIFACTS_DIR = PROJECT_ROOT / "mlflow" / "artifacts"

app = Flask(__name__)

# Globals loaded at startup
store = None
model = None
scaler = None
threshold = None


def load_model():
    """Load the latest model from MLflow registry."""
    global model, scaler, threshold, store

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load latest model version from registry
    model_uri = "models:/transaction-anomaly-detector/latest"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    # Load scaler and threshold
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    threshold = joblib.load(ARTIFACTS_DIR / "threshold.pkl")

    # Initialize Feast store
    store = FeatureStore(repo_path=str(FEATURE_REPO))

    print(f"Model loaded. Threshold: {threshold:.6f}")


def get_online_features(customer_id: str) -> dict:
    """Fetch online features for a customer from Feast."""
    features = store.get_online_features(
        features=[
            "customer_profile_features:avg_transaction_amount_30d",
            "customer_profile_features:std_transaction_amount_30d",
            "customer_profile_features:total_transactions_30d",
            "customer_profile_features:avg_transactions_per_day_30d",
            "customer_profile_features:max_transaction_amount_30d",
            "customer_profile_features:unique_merchants_30d",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    return {k: v[0] for k, v in features.items()}


def score_transaction(transaction: dict) -> dict:
    """Score a single transaction for anomaly detection."""
    customer_id = transaction["customer_id"]

    # Get online profile features from Feast
    profile = get_online_features(customer_id)

    # Build feature vector (same order as training)
    feature_vector = np.array([[
        float(transaction.get("amount", 0)),
        int(transaction.get("transaction_hour", 0)),
        int(transaction.get("transaction_day_of_week", 0)),
        int(transaction.get("is_online", 0)),
        float(transaction.get("distance_from_home", 0)),
        float(profile.get("avg_transaction_amount_30d", 0) or 0),
        float(profile.get("std_transaction_amount_30d", 0) or 0),
        int(profile.get("total_transactions_30d", 0) or 0),
        float(profile.get("avg_transactions_per_day_30d", 0) or 0),
        float(profile.get("max_transaction_amount_30d", 0) or 0),
        int(profile.get("unique_merchants_30d", 0) or 0),
    ]])

    # Scale
    feature_scaled = scaler.transform(feature_vector)

    # Predict
    x_tensor = torch.FloatTensor(feature_scaled)
    reconstruction_error = model.get_reconstruction_error(x_tensor).item()
    is_anomaly = reconstruction_error > threshold

    return {
        "customer_id": customer_id,
        "reconstruction_error": round(reconstruction_error, 6),
        "threshold": round(threshold, 6),
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(reconstruction_error / threshold, 4),
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """Score a transaction for anomaly detection.

    Expected JSON body:
    {
        "customer_id": "CUST_0001",
        "amount": 150.0,
        "transaction_hour": 14,
        "transaction_day_of_week": 2,
        "is_online": 1,
        "distance_from_home": 5.3
    }
    """
    data = request.get_json(force=True)

    required_fields = ["customer_id", "amount"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = score_transaction(data)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Score multiple transactions."""
    data = request.get_json(force=True)

    if "transactions" not in data:
        return jsonify({"error": "Expected 'transactions' array"}), 400

    results = [score_transaction(txn) for txn in data["transactions"]]
    return jsonify({"results": results})


def run_server(host: str = "0.0.0.0", port: int = 5001):
    """Start the serving API."""
    load_model()
    print(f"Starting serving API on {host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()
