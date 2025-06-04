import pandas as pd

from src.features.rfm import compute_rfm


def test_compute_rfm():
    data = pd.DataFrame({
        "customer_id": ["a", "a", "b"],
        "transaction_timestamp": [
            "2024-01-01", "2024-01-10", "2024-01-05"
        ],
        "transaction_amount": [10.0, 20.0, 30.0],
    })
    rfm = compute_rfm(data, "customer_id", "transaction_timestamp", "transaction_amount")
    assert set(rfm.columns) == {"customer_id", "recency", "frequency", "monetary"}
    # two customers -> 2 rows
    assert len(rfm) == 2
