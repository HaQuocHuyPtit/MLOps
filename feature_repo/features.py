from datetime import timedelta

from feast import FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Float64, Int64, String

from entities import customer

# --- Offline source: PostgreSQL tables written by the Kafka consumer ETL ---

transaction_source = PostgreSQLSource(
    name="transaction_source",
    query="SELECT * FROM transactions",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# --- Feature View: transaction-level features ---

transaction_features = FeatureView(
    name="transaction_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="merchant_category", dtype=String),
        Field(name="transaction_hour", dtype=Int64),
        Field(name="transaction_day_of_week", dtype=Int64),
        Field(name="is_online", dtype=Int64),
        Field(name="distance_from_home", dtype=Float64),
    ],
    source=transaction_source,
    online=True,
)

# --- Aggregated customer profile source ---

customer_profile_source = PostgreSQLSource(
    name="customer_profile_source",
    query="SELECT * FROM customer_profiles",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

customer_profile_features = FeatureView(
    name="customer_profile_features",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="avg_transaction_amount_30d", dtype=Float64),
        Field(name="std_transaction_amount_30d", dtype=Float64),
        Field(name="total_transactions_30d", dtype=Int64),
        Field(name="avg_transactions_per_day_30d", dtype=Float32),
        Field(name="max_transaction_amount_30d", dtype=Float64),
        Field(name="unique_merchants_30d", dtype=Int64),
    ],
    source=customer_profile_source,
    online=True,
)
