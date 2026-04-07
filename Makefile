.PHONY: setup infra-up infra-down feast-apply ingest materialize train serve test ui stop help

PYTHON := python3
PROJECT_ROOT := $(shell pwd)
FEATURE_REPO := $(PROJECT_ROOT)/feature_repo

# Infrastructure connection strings
PG_URI := postgresql+psycopg2://mlops:mlops_secret@localhost:5433/mlflow
MINIO_ENDPOINT := http://localhost:9200

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Infrastructure
# ──────────────────────────────────────────────

setup: ## Install Python dependencies
	pip install -r requirements.txt

infra-up: ## Start all services (Kafka, PostgreSQL, Redis, MinIO)
	docker compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 15
	@echo ""
	@echo "Services running:"
	@echo "  Kafka:      localhost:9092"
	@echo "  PostgreSQL: localhost:5433"
	@echo "  Redis:      localhost:6379"
	@echo "  MinIO API:  localhost:9200"
	@echo "  MinIO UI:   localhost:9201"

infra-down: ## Stop all Docker services
	docker compose down

# ──────────────────────────────────────────────
# Feature Store
# ──────────────────────────────────────────────

feast-apply: ## Register Feast feature definitions
	cd $(FEATURE_REPO) && feast apply
	@echo "Feast features registered."

feast-ui: ## Launch Feast UI (port 8890)
	cd $(FEATURE_REPO) && feast ui --host 0.0.0.0 --port 8890

# ──────────────────────────────────────────────
# Data Pipeline
# ──────────────────────────────────────────────

ingest: ## Run Kafka producer + consumer ETL
	@echo "=== Step 1: Producing transactions to Kafka ==="
	$(PYTHON) pipelines/kafka_producer.py
	@echo ""
	@echo "=== Step 2: Consuming & transforming to PostgreSQL ==="
	$(PYTHON) pipelines/kafka_consumer.py

materialize: ## Materialize features from offline → online store (PostgreSQL → Redis)
	cd $(FEATURE_REPO) && feast materialize-incremental $$(date -u +"%Y-%m-%dT%H:%M:%S")
	@echo "Features materialized to Redis online store."

# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────

train: ## Train autoencoder + log to MLflow
	$(PYTHON) pipelines/train.py

mlflow-ui: ## Launch MLflow UI (port 5002)
	MLFLOW_S3_ENDPOINT_URL=$(MINIO_ENDPOINT) \
	AWS_ACCESS_KEY_ID=mlops_minio \
	AWS_SECRET_ACCESS_KEY=mlops_minio_secret \
	mlflow ui --backend-store-uri $(PG_URI) \
		--default-artifact-root s3://mlflow-artifacts \
		--host 0.0.0.0 --port 5002

# ──────────────────────────────────────────────
# Model Serving
# ──────────────────────────────────────────────

serve: ## Start real-time serving API (port 5001)
	$(PYTHON) pipelines/serve.py

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

test: ## Run end-to-end tests against serving API
	$(PYTHON) tests/test_e2e.py

# ──────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────

pipeline: infra-up feast-apply ingest materialize train  ## Run full pipeline (up to training)
	@echo ""
	@echo "=== Pipeline complete ==="
	@echo "Next steps:"
	@echo "  make serve   - Start serving API"
	@echo "  make test    - Run E2E tests (requires serve)"
	@echo "  make ui      - Launch Feast + MLflow UIs"

ui: ## Launch both Feast UI and MLflow UI
	@echo "Starting MLflow UI on port 5000..."
	MLFLOW_S3_ENDPOINT_URL=$(MINIO_ENDPOINT) \
	AWS_ACCESS_KEY_ID=mlops_minio \
	AWS_SECRET_ACCESS_KEY=mlops_minio_secret \
	mlflow ui --backend-store-uri $(PG_URI) \
		--default-artifact-root s3://mlflow-artifacts \
		--host 0.0.0.0 --port 5002 &
	@echo "Starting Feast UI on port 8890..."
	cd $(FEATURE_REPO) && feast ui --host 0.0.0.0 --port 8890 &
	@echo ""
	@echo "UIs running:"
	@echo "  MLflow:    http://localhost:5002"
	@echo "  Feast:     http://localhost:8890"
	@echo "  MinIO:     http://localhost:9201"

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

stop: ## Stop all background services
	-docker compose down
	-pkill -f "feast ui" || true
	-pkill -f "mlflow ui" || true
	-pkill -f "serve.py" || true
	@echo "All services stopped."

clean: ## Remove generated data, artifacts, and Docker volumes
	rm -rf data/
	docker compose down -v
	@echo "Cleaned data and Docker volumes."
