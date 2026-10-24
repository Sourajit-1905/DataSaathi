"""
Shared pytest fixtures for DataSaathi backend tests.
"""

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient


import os
os.environ.setdefault("SESSION_TTL_HOURS", "0")
from main import app


@pytest.fixture(scope="session")
def client():
    """FastAPI test client — one instance for the entire test session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_csv_bytes() -> bytes:
    """Minimal clean CSV as bytes (mimics a real upload)."""
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [30000, 45000, 60000, 75000, 90000],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Purchased": [1, 0, 1, 1, 0],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def dirty_csv_bytes() -> bytes:
    """CSV with deliberate issues: missing values, duplicates, outliers."""
    df = pd.DataFrame({
        "Age": [25, None, 35, 35, 999],
        "Income": [30000, 45000, None, 45000, -100],
        "Gender": ["Male", "male", "MALE", "Female", None],
        "Purchased": [1, 0, 1, 0, 1],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def text_csv_bytes() -> bytes:
    """CSV with a free-form text column for NLP tests."""
    df = pd.DataFrame({
        "ReviewID": range(1, 31),
        "ReviewText": [
            "This product is absolutely amazing! Highly recommend.",
            "Terrible experience. Never buying again.",
            "Decent product but the delivery was slow.",
            "Great quality for the price.",
            "Did not meet my expectations at all.",
        ] * 6,
        "Rating": [5, 1, 3, 4, 2] * 6,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def uploaded_session(client, sample_csv_bytes) -> str:
    """Upload a sample CSV and return the session_id. Used by downstream tests."""
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("test_data.csv", sample_csv_bytes, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["session_id"]
