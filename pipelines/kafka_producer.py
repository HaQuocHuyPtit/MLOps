"""
Kafka Producer: Simulates credit card transaction data.

Generates realistic tabular transaction data and streams it to Kafka topic
'transactions'. ~5% of transactions are injected as anomalies (unusually
high amounts, odd hours, far from home).
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "entertainment", "travel", "healthcare", "utilities",
]

CUSTOMER_IDS = [f"CUST_{i:04d}" for i in range(1, 201)]  # 200 customers

# Per-customer home location (lat, lon) for distance simulation
CUSTOMER_HOMES = {
    cid: (round(random.uniform(30.0, 45.0), 4), round(random.uniform(-120.0, -75.0), 4))
    for cid in CUSTOMER_IDS
}


def generate_normal_transaction(customer_id: str) -> dict:
    """Generate a normal transaction."""
    hour = random.choices(range(24), weights=[
        1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 10, 10,
        10, 9, 8, 8, 9, 10, 8, 6, 4, 3, 2, 1,
    ])[0]

    return {
        "transaction_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "amount": round(random.lognormvariate(3.0, 1.0), 2),  # median ~$20
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "transaction_hour": hour,
        "transaction_day_of_week": random.randint(0, 6),
        "is_online": random.choices([0, 1], weights=[0.6, 0.4])[0],
        "distance_from_home": round(random.expovariate(0.1), 2),  # mean ~10 km
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def generate_anomalous_transaction(customer_id: str) -> dict:
    """Generate an anomalous transaction (unusual patterns)."""
    txn = generate_normal_transaction(customer_id)
    anomaly_type = random.choice(["high_amount", "far_location", "odd_time", "combo"])

    if anomaly_type in ("high_amount", "combo"):
        txn["amount"] = round(random.uniform(2000, 15000), 2)
    if anomaly_type in ("far_location", "combo"):
        txn["distance_from_home"] = round(random.uniform(500, 5000), 2)
    if anomaly_type in ("odd_time", "combo"):
        txn["transaction_hour"] = random.choice([2, 3, 4])

    return txn


def produce_transactions(num_transactions: int = 20, anomaly_rate: float = 0.05):
    """Stream transactions to Kafka."""
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    print(f"Producing {num_transactions} transactions to topic '{TOPIC}'...")
    anomaly_count = 0

    for i in range(num_transactions):
        customer_id = random.choice(CUSTOMER_IDS)
        is_anomaly = random.random() < anomaly_rate

        if is_anomaly:
            txn = generate_anomalous_transaction(customer_id)
            txn["_label"] = 1  # anomaly label (for evaluation only)
            anomaly_count += 1
        else:
            txn = generate_normal_transaction(customer_id)
            txn["_label"] = 0

        producer.send(TOPIC, key=customer_id, value=txn)

        if (i + 1) % 1000 == 0:
            print(f"  Sent {i + 1}/{num_transactions} transactions")

    producer.flush()
    producer.close()
    print(f"Done. Total: {num_transactions}, Anomalies: {anomaly_count} "
          f"({anomaly_count / num_transactions * 100:.1f}%)")


if __name__ == "__main__":
    produce_transactions()
