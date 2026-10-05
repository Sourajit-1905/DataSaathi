"""
Stage 1 Tests — Dataset upload, preview, reset, info.
Covers all items in the Stage 1 verification checklist.
"""

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient


# Health 

def test_health_check(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


# Upload: happy paths 

def test_upload_csv_success(client: TestClient, sample_csv_bytes: bytes):
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("sales.csv", sample_csv_bytes, "text/csv")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["filename"] == "sales.csv"
    assert data["rows"] == 5
    assert data["columns"] == 4
    assert "Age" in data["column_names"]
    assert data["file_size_kb"] > 0
    assert "upload_time" in data


def test_upload_csv_creates_session_files(client: TestClient, sample_csv_bytes: bytes, tmp_path):
    """
    Verify original, working, and history files are created.
    We check indirectly: if preview works, files must exist.
    """
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", sample_csv_bytes, "text/csv")},
    )
    assert upload_resp.status_code == 201
    session_id = upload_resp.json()["session_id"]

    # Preview works → working file exists
    preview_resp = client.get(f"/api/dataset/{session_id}/preview")
    assert preview_resp.status_code == 200


def test_upload_xlsx_converts_to_csv(client: TestClient):
    """XLSX file should upload and parse correctly."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    xlsx_bytes = buf.getvalue()

    response = client.post(
        "/api/dataset/upload",
        files={"file": ("data.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["rows"] == 3
    assert data["columns"] == 2


# Upload: error paths 

def test_upload_invalid_extension_returns_400(client: TestClient):
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("data.json", b'{"a":1}', "application/json")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == 400


def test_upload_file_too_large_returns_413(client: TestClient):
    # Generate a 51 MB payload
    large_bytes = b"a,b\n" + b"1,2\n" * (51 * 1024 * 1024 // 4)
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("huge.csv", large_bytes, "text/csv")},
    )
    assert response.status_code == 413
    assert response.json()["code"] == 413


def test_upload_empty_csv_returns_400(client: TestClient):
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400


# Preview 

def test_preview_returns_ten_rows(client: TestClient, uploaded_session: str):
    # uploaded_session has 5 rows — preview should return all 5
    response = client.get(f"/api/dataset/{uploaded_session}/preview")
    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 5  # only 5 rows in fixture
    assert "column_dtypes" in data
    assert data["total_rows"] == 5
    assert data["total_columns"] == 4


def test_preview_invalid_session_returns_404(client: TestClient):
    fake_id = "00000000-0000-4000-8000-000000000000"
    response = client.get(f"/api/dataset/{fake_id}/preview")
    assert response.status_code == 404


def test_preview_invalid_uuid_returns_400(client: TestClient):
    response = client.get("/api/dataset/not-a-uuid/preview")
    assert response.status_code == 400


# Reset 

def test_reset_returns_correct_response(client: TestClient, uploaded_session: str):
    response = client.post(f"/api/dataset/{uploaded_session}/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reset"
    assert data["session_id"] == uploaded_session


def test_reset_restores_original(client: TestClient, sample_csv_bytes: bytes):
    # Upload
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", sample_csv_bytes, "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]

    # Verify original row count
    info_before = client.get(f"/api/dataset/{session_id}/info").json()
    original_rows = info_before["rows"]

    # Reset
    reset_resp = client.post(f"/api/dataset/{session_id}/reset")
    assert reset_resp.status_code == 200

    # Row count should be unchanged after reset (since we haven't cleaned anything)
    info_after = client.get(f"/api/dataset/{session_id}/info").json()
    assert info_after["rows"] == original_rows


# Info 

def test_info_returns_correct_structure(client: TestClient, uploaded_session: str):
    response = client.get(f"/api/dataset/{uploaded_session}/info")
    assert response.status_code == 200
    data = response.json()
    assert data["rows"] == 5
    assert data["columns"] == 4
    assert "column_names" in data
    assert "column_dtypes" in data
    assert data["memory_usage_mb"] >= 0


# Error format 

def test_error_response_has_correct_shape(client: TestClient):
    """All errors must return { error, detail, code } — never raw tracebacks."""
    response = client.get("/api/dataset/not-a-uuid/info")
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "code" in data
    # Must not contain traceback keywords
    detail_str = str(data)
    assert "Traceback" not in detail_str
    assert "File \"/" not in detail_str
