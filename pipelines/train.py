"""
Training Pipeline.

Fetches historical features from Feast offline store, trains a PyTorch
autoencoder for anomaly detection, and logs everything to MLflow.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from feast import FeatureStore

# Add parent to path for model import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.autoencoder import TransactionAutoencoder

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_REPO = PROJECT_ROOT / "feature_repo"
DATA_DIR = PROJECT_ROOT / "data"
MLFLOW_TRACKING_URI = "postgresql+psycopg2://mlops:mlops_secret@localhost:5433/mlflow"

# MinIO as S3-compatible artifact store
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9200"
os.environ["AWS_ACCESS_KEY_ID"] = "mlops_minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mlops_minio_secret"
MLFLOW_ARTIFACT_ROOT = "s3://mlflow-artifacts"
PG_CONN = "postgresql+psycopg2://mlops:mlops_secret@localhost:5433/feast_offline"


def get_historical_features() -> pd.DataFrame:
    """Retrieve historical features from Feast for training."""
    from sqlalchemy import create_engine

    store = FeatureStore(repo_path=str(FEATURE_REPO))

    # Load entity DataFrame from PostgreSQL offline store
    engine = create_engine(PG_CONN)
    txn_df = pd.read_sql("SELECT customer_id, event_timestamp FROM transactions", engine)

    labels_path = DATA_DIR / "labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError("Labels file not found. Run 'make ingest' first.")
    labels_df = pd.read_parquet(labels_path)
    engine.dispose()
    print(f"Loaded {len(txn_df)} transactions, {len(labels_df)} labels.")

    # Entity DataFrame: customer_id + event_timestamp (required by Feast)
    # Offset entity timestamps by 1 second so point-in-time join finds features
    # (Feast looks for feature_timestamp <= entity_timestamp)
    entity_df = txn_df[["customer_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True) + pd.Timedelta(seconds=1)

    # Get historical features
    print("Fetching historical features from Feast...")
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_features:amount",
            "transaction_features:transaction_hour",
            "transaction_features:transaction_day_of_week",
            "transaction_features:is_online",
            "transaction_features:distance_from_home",
            "customer_profile_features:avg_transaction_amount_30d",
            "customer_profile_features:std_transaction_amount_30d",
            "customer_profile_features:total_transactions_30d",
            "customer_profile_features:avg_transactions_per_day_30d",
            "customer_profile_features:max_transaction_amount_30d",
            "customer_profile_features:unique_merchants_30d",
        ],
    ).to_df()

    # Merge labels by customer_id (many-to-many possible, keep all)
    training_data = training_data.merge(
        labels_df[["customer_id", "_label"]],
        on="customer_id",
        how="left",
    )

    # De-duplicate: keep one row per entity timestamp
    training_data = training_data.drop_duplicates(subset=["customer_id", "event_timestamp"])

    print(f"Retrieved {len(training_data)} feature rows.")
    if len(training_data) == 0:
        raise ValueError("No feature rows retrieved from Feast. Check data and feast-apply.")
    return training_data


def prepare_data(df: pd.DataFrame):
    """Prepare features for autoencoder training."""
    feature_cols = [
        "amount", "transaction_hour", "transaction_day_of_week", "is_online",
        "distance_from_home", "avg_transaction_amount_30d",
        "std_transaction_amount_30d", "total_transactions_30d",
        "avg_transactions_per_day_30d", "max_transaction_amount_30d",
        "unique_merchants_30d",
    ]

    X = df[feature_cols].fillna(0).values
    labels = df["_label"].fillna(0).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train only on normal transactions (label == 0)
    normal_mask = labels == 0
    X_train = X_scaled[normal_mask]
    X_all = X_scaled
    y_all = labels

    return X_train, X_all, y_all, scaler, feature_cols


def train_autoencoder(
    X_train: np.ndarray,
    input_dim: int,
    encoding_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
):
    """Train the autoencoder on normal transactions."""
    model = TransactionAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.FloatTensor(X_train)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        avg_loss = epoch_loss / len(X_train)
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    return model, history


def evaluate_model(
    model: TransactionAutoencoder,
    X_all: np.ndarray,
    y_all: np.ndarray,
    threshold_percentile: float = 95.0,
):
    """Evaluate the model's anomaly detection performance."""
    model.eval()
    X_tensor = torch.FloatTensor(X_all)
    errors = model.get_reconstruction_error(X_tensor).numpy()

    # Set threshold at percentile of reconstruction errors on all data
    threshold = np.percentile(errors, threshold_percentile)
    predictions = (errors > threshold).astype(int)

    # Metrics
    auc = roc_auc_score(y_all, errors) if len(np.unique(y_all)) > 1 else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_all, predictions, average="binary", zero_division=0
    )

    metrics = {
        "auc_roc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "threshold": float(threshold),
        "mean_reconstruction_error": float(errors.mean()),
        "anomaly_rate_predicted": float(predictions.mean()),
    }

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics, threshold


def run_training():
    """Full training pipeline: Feast → Train → MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create or get experiment with MinIO artifact location
    experiment_name = "anomaly-detection-autoencoder"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f"{MLFLOW_ARTIFACT_ROOT}/anomaly-detection",
        )
    mlflow.set_experiment(experiment_name)

    # Hyperparameters
    params = {
        "encoding_dim": 8,
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "threshold_percentile": 95.0,
    }

    with mlflow.start_run(run_name="autoencoder-training"):
        # Log parameters
        mlflow.log_params(params)

        # 1. Fetch features from Feast
        df = get_historical_features()
        X_train, X_all, y_all, scaler, feature_cols = prepare_data(df)
        input_dim = X_train.shape[1]

        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("total_samples", len(X_all))
        mlflow.log_param("features", feature_cols)

        # 2. Train model
        print(f"\nTraining autoencoder (input_dim={input_dim})...")
        model, history = train_autoencoder(
            X_train, input_dim,
            encoding_dim=params["encoding_dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
        )

        # Log training loss curve
        for epoch, loss in enumerate(history):
            mlflow.log_metric("train_loss", loss, step=epoch)

        # 3. Evaluate
        metrics, threshold = evaluate_model(
            model, X_all, y_all, params["threshold_percentile"]
        )
        mlflow.log_metrics(metrics)

        # 4. Log model to MLflow
        mlflow.pytorch.log_model(model, "autoencoder-model")

        # 5. Save scaler and threshold as artifacts
        import joblib
        artifacts_dir = PROJECT_ROOT / "mlflow" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(scaler, artifacts_dir / "scaler.pkl")
        joblib.dump(threshold, artifacts_dir / "threshold.pkl")
        mlflow.log_artifact(str(artifacts_dir / "scaler.pkl"))
        mlflow.log_artifact(str(artifacts_dir / "threshold.pkl"))

        # 6. Register model
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/autoencoder-model"
        mlflow.register_model(model_uri, "transaction-anomaly-detector")

        print(f"\nModel logged to MLflow (run_id: {run_id})")
        print(f"Threshold: {threshold:.6f}")

    return model, scaler, threshold


if __name__ == "__main__":
    run_training()
