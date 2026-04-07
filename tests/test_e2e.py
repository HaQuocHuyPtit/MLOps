"""
End-to-end pipeline test.

Sends sample transactions to the serving API and validates responses.
Run after `make serve` is up.
"""

import json
import sys
import time

import requests

SERVE_URL = "http://localhost:5001"


def wait_for_server(timeout: int = 30):
    """Wait for serving API to be ready."""
    print("Waiting for serving API...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{SERVE_URL}/health", timeout=2)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("Server did not start in time.")
    return False


def test_health():
    """Test health endpoint."""
    resp = requests.get(f"{SERVE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print("[PASS] Health check")


def test_normal_transaction():
    """Test a normal-looking transaction."""
    txn = {
        "customer_id": "CUST_0001",
        "amount": 25.50,
        "transaction_hour": 14,
        "transaction_day_of_week": 2,
        "is_online": 0,
        "distance_from_home": 3.2,
    }
    resp = requests.post(f"{SERVE_URL}/predict", json=txn)
    assert resp.status_code == 200
    data = resp.json()

    assert "is_anomaly" in data
    assert "reconstruction_error" in data
    assert "anomaly_score" in data
    assert isinstance(data["is_anomaly"], bool)
    print(f"[PASS] Normal transaction -> anomaly={data['is_anomaly']}, "
          f"score={data['anomaly_score']}")


def test_anomalous_transaction():
    """Test a clearly anomalous transaction."""
    txn = {
        "customer_id": "CUST_0001",
        "amount": 9999.99,
        "transaction_hour": 3,
        "transaction_day_of_week": 1,
        "is_online": 1,
        "distance_from_home": 3000.0,
    }
    resp = requests.post(f"{SERVE_URL}/predict", json=txn)
    assert resp.status_code == 200
    data = resp.json()

    assert "is_anomaly" in data
    # We expect this to have a high anomaly score
    print(f"[PASS] Anomalous transaction -> anomaly={data['is_anomaly']}, "
          f"score={data['anomaly_score']}")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    txns = {
        "transactions": [
            {
                "customer_id": "CUST_0010",
                "amount": 15.00,
                "transaction_hour": 10,
                "transaction_day_of_week": 3,
                "is_online": 0,
                "distance_from_home": 2.0,
            },
            {
                "customer_id": "CUST_0020",
                "amount": 12000.00,
                "transaction_hour": 3,
                "transaction_day_of_week": 0,
                "is_online": 1,
                "distance_from_home": 4500.0,
            },
        ]
    }
    resp = requests.post(f"{SERVE_URL}/predict/batch", json=txns)
    assert resp.status_code == 200
    data = resp.json()

    assert "results" in data
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert "is_anomaly" in r
        assert "anomaly_score" in r

    print(f"[PASS] Batch prediction -> {len(data['results'])} results")
    for r in data["results"]:
        print(f"  {r['customer_id']}: anomaly={r['is_anomaly']}, score={r['anomaly_score']}")


def test_missing_fields():
    """Test validation error on missing fields."""
    resp = requests.post(f"{SERVE_URL}/predict", json={"transaction_hour": 10})
    assert resp.status_code == 400
    assert "error" in resp.json()
    print("[PASS] Missing fields validation")


def run_all_tests():
    """Run all end-to-end tests."""
    if not wait_for_server():
        sys.exit(1)

    print("\n=== Running E2E Tests ===\n")
    tests = [
        test_health,
        test_normal_transaction,
        test_anomalous_transaction,
        test_batch_prediction,
        test_missing_fields,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_all_tests()
