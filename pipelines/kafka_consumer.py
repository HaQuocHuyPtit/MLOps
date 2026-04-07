"""
Kafka Consumer ETL Pipeline.

Consumes transactions from Kafka, transforms them into feature-ready
DataFrames, and writes to PostgreSQL (Feast offline store).
Also computes aggregated customer profile features.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy import create_engine

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"
PG_CONN = "postgresql+psycopg2://mlops:mlops_secret@localhost:5433/feast_offline"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def consume_transactions(timeout_ms: int = 10000) -> list[dict]:
    """Consume all available transactions from Kafka."""
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        group_id=None,  # No group → always read from beginning, no offset tracking
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=timeout_ms,
    )

    transactions = []
    print("Consuming transactions from Kafka...")
    for message in consumer:
        transactions.append(message.value)

    consumer.close()
    print(f"Consumed {len(transactions)} transactions.")
    return transactions


def transform_transactions(raw: list[dict]) -> pd.DataFrame:
    """Clean and transform raw transactions into feature DataFrame."""
    df = pd.DataFrame(raw)

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["created_timestamp"] = pd.Timestamp.now(tz="UTC")

    # Ensure correct types
    df["amount"] = df["amount"].astype(float)
    df["transaction_hour"] = df["transaction_hour"].astype(int)
    df["transaction_day_of_week"] = df["transaction_day_of_week"].astype(int)
    df["is_online"] = df["is_online"].astype(int)
    df["distance_from_home"] = df["distance_from_home"].astype(float)

    # Keep labels separate for evaluation
    labels = df[["transaction_id", "customer_id", "_label"]].copy()

    # Drop internal columns not part of features
    feature_cols = [
        "transaction_id", "customer_id", "amount", "merchant_category",
        "transaction_hour", "transaction_day_of_week", "is_online",
        "distance_from_home", "event_timestamp", "created_timestamp",
    ]
    features_df = df[feature_cols].copy()

    return features_df, labels


def compute_customer_profiles(txn_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated customer-level features from transactions."""
    grouped = txn_df.groupby("customer_id")

    profiles = pd.DataFrame({
        "customer_id": grouped["amount"].mean().index,
        "avg_transaction_amount_30d": grouped["amount"].mean().values,
        "std_transaction_amount_30d": grouped["amount"].std().fillna(0).values,
        "total_transactions_30d": grouped["amount"].count().values,
        "avg_transactions_per_day_30d": (grouped["amount"].count() / 30).values.astype(np.float32),
        "max_transaction_amount_30d": grouped["amount"].max().values,
        "unique_merchants_30d": grouped["merchant_category"].nunique().values,
    })

    profiles["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    profiles["created_timestamp"] = pd.Timestamp.now(tz="UTC")

    return profiles


def write_to_postgres(df: pd.DataFrame, table_name: str):
    """Write DataFrame to PostgreSQL table in the offline store."""
    engine = create_engine(PG_CONN)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    engine.dispose()
    print(f"Written {len(df)} rows to PostgreSQL table '{table_name}'")


def run_etl():
    """Full ETL pipeline: consume → transform → write PostgreSQL."""
    raw = consume_transactions()
    if not raw:
        print("No transactions found. Is the producer running?")
        return

    # Transform
    txn_features, labels = transform_transactions(raw)

    # Compute customer profiles
    profiles = compute_customer_profiles(txn_features)

    # Write to PostgreSQL offline store
    write_to_postgres(txn_features, "transactions")
    write_to_postgres(profiles, "customer_profiles")

    # Write labels for evaluation (local Parquet, not part of Feast)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(DATA_DIR / "labels.parquet", index=False)
    print(f"Written {len(labels)} labels to {DATA_DIR / 'labels.parquet'}")

    print(f"\nETL Summary:")
    print(f"  Transactions: {len(txn_features)}")
    print(f"  Customer profiles: {len(profiles)}")
    print(f"  Unique customers: {txn_features['customer_id'].nunique()}")


if __name__ == "__main__":
    run_etl()
