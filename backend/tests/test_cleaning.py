"""
Stage 3 Tests — Data Cleaning.
Covers all 9 issue detectors + apply + undo + history + router endpoints.
"""

import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


# Fixtures

@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "Age":      [25, None, 35, 35, 999, 22, 30, None, 45, 28],
        "Income":   [30000, 45000, None, 45000, -100, 60000, 75000, 80000, None, 55000],
        "Gender":   ["Male", "male", "MALE", "Female", None, "female", "Male", "Female", "Male", None],
        "Status":   ["Active"] * 10,
        "Purchased": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "Age":       [25, 30, 35, 40, 45],
        "Income":    [30000, 45000, 60000, 75000, 90000],
        "Gender":    ["Male", "Female", "Male", "Female", "Male"],
        "Purchased": [1, 0, 1, 1, 0],
    })


# C1 — Missing Numerical

def test_c1_detect_missing_numerical(dirty_df):
    from services.clean_service import detect_missing_numerical
    result = detect_missing_numerical(dirty_df)
    col_names = [r["column"] for r in result]
    assert "Age" in col_names
    assert "Income" in col_names
    assert "Purchased" not in col_names
    age_entry = next(r for r in result if r["column"] == "Age")
    assert age_entry["missing_count"] == 2
    assert age_entry["missing_pct"] == 20.0


def test_c1_apply_mean(dirty_df):
    from services.clean_service import apply_missing_numerical
    df = apply_missing_numerical(dirty_df.copy(), "Age", "mean")
    assert df["Age"].isna().sum() == 0


def test_c1_apply_median(dirty_df):
    from services.clean_service import apply_missing_numerical
    df = apply_missing_numerical(dirty_df.copy(), "Age", "median")
    assert df["Age"].isna().sum() == 0


def test_c1_apply_constant(dirty_df):
    from services.clean_service import apply_missing_numerical
    df = apply_missing_numerical(dirty_df.copy(), "Age", "constant", fill_value=0.0)
    assert df["Age"].isna().sum() == 0
    assert (df["Age"] == 0.0).sum() == 2


def test_c1_apply_drop_rows(dirty_df):
    from services.clean_service import apply_missing_numerical
    rows_before = len(dirty_df)
    df = apply_missing_numerical(dirty_df.copy(), "Age", "drop_rows")
    assert len(df) == rows_before - 2
    assert df["Age"].isna().sum() == 0


def test_c1_apply_drop_column(dirty_df):
    from services.clean_service import apply_missing_numerical
    df = apply_missing_numerical(dirty_df.copy(), "Age", "drop_column")
    assert "Age" not in df.columns


def test_c1_preview_does_not_modify_df(dirty_df):
    from services.clean_service import preview_missing_numerical
    original_nulls = dirty_df["Age"].isna().sum()
    preview_missing_numerical(dirty_df, "Age", "median")
    assert dirty_df["Age"].isna().sum() == original_nulls


# C2 — Missing Categorical

def test_c2_detect_missing_categorical(dirty_df):
    from services.clean_service import detect_missing_categorical
    result = detect_missing_categorical(dirty_df)
    col_names = [r["column"] for r in result]
    assert "Gender" in col_names


def test_c2_apply_mode(dirty_df):
    from services.clean_service import apply_missing_categorical
    df = apply_missing_categorical(dirty_df.copy(), "Gender", "mode")
    assert df["Gender"].isna().sum() == 0


def test_c2_apply_constant(dirty_df):
    from services.clean_service import apply_missing_categorical
    df = apply_missing_categorical(dirty_df.copy(), "Gender", "constant", fill_value="Unknown")
    assert df["Gender"].isna().sum() == 0
    assert "Unknown" in df["Gender"].values


def test_c2_apply_drop_rows(dirty_df):
    from services.clean_service import apply_missing_categorical
    rows_before = len(dirty_df)
    df = apply_missing_categorical(dirty_df.copy(), "Gender", "drop_rows")
    assert len(df) < rows_before
    assert df["Gender"].isna().sum() == 0


# C3 — Duplicates

def test_c3_detect_duplicates():
    from services.clean_service import detect_duplicates
    df = pd.DataFrame({
        "A": [1, 1, 2, 3],
        "B": ["x", "x", "y", "z"],
    })
    result = detect_duplicates(df)
    assert result["duplicate_count"] == 1
    assert result["duplicate_pct"] == 25.0


def test_c3_apply_duplicates():
    from services.clean_service import apply_duplicates
    df = pd.DataFrame({
        "A": [1, 1, 2, 3],
        "B": ["x", "x", "y", "z"],
    })
    df_clean = apply_duplicates(df)
    assert len(df_clean) == 3
    assert df_clean.duplicated().sum() == 0


def test_c3_no_duplicates_returns_zero():
    from services.clean_service import detect_duplicates
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    result = detect_duplicates(df)
    assert result["duplicate_count"] == 0


# C4 — Outliers

def test_c4_detect_outliers():
    from services.clean_service import detect_outliers
    df = pd.DataFrame({"Value": [10, 12, 11, 13, 10, 12, 1000]})
    result = detect_outliers(df)
    assert len(result) == 1
    assert result[0]["outlier_count"] == 1


def test_c4_apply_iqr_cap():
    from services.clean_service import apply_outliers
    df = pd.DataFrame({"Value": [10, 12, 11, 13, 10, 12, 1000]})
    df_clean = apply_outliers(df.copy(), "Value", "iqr_cap")
    assert df_clean["Value"].max() < 1000


def test_c4_apply_zscore_cap():
    from services.clean_service import apply_outliers
    # Need enough normal values so z-score bound falls well below 1000
    normal = [10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 11]
    df = pd.DataFrame({"Value": normal + [1000]})
    df_clean = apply_outliers(df.copy(), "Value", "zscore_cap")
    assert df_clean["Value"].max() < 1000


def test_c4_apply_drop_rows():
    from services.clean_service import apply_outliers
    df = pd.DataFrame({"Value": [10, 12, 11, 13, 10, 12, 1000]})
    rows_before = len(df)
    df_clean = apply_outliers(df.copy(), "Value", "drop_rows")
    assert len(df_clean) < rows_before


# C5 — Type Conversion

def test_c5_detect_type_inconsistencies():
    from services.clean_service import detect_type_inconsistencies
    df = pd.DataFrame({
        "NumericStr": ["1", "2", "3", "4", "5"],
        "TrueText":   ["apple", "banana", "cherry", "date", "fig"],
    })
    result = detect_type_inconsistencies(df)
    cols = [r["column"] for r in result]
    assert "NumericStr" in cols
    assert "TrueText" not in cols


def test_c5_apply_to_numeric():
    from services.clean_service import apply_type_conversion
    df = pd.DataFrame({"Value": ["1", "2", "3", "4", "5"]})
    df_converted = apply_type_conversion(df.copy(), "Value", "to_numeric")
    assert pd.api.types.is_numeric_dtype(df_converted["Value"])


# C6 — String Standardisation

def test_c6_detect_string_issues(dirty_df):
    from services.clean_service import detect_string_issues
    result = detect_string_issues(dirty_df)
    col_names = [r["column"] for r in result]
    assert "Gender" in col_names


def test_c6_apply_lowercase(dirty_df):
    from services.clean_service import apply_string_standardisation
    df = apply_string_standardisation(dirty_df.copy(), "Gender", "lowercase")
    non_null = df["Gender"].dropna()
    assert all(v == v.lower() for v in non_null)


def test_c6_apply_titlecase(dirty_df):
    from services.clean_service import apply_string_standardisation
    df = apply_string_standardisation(dirty_df.copy(), "Gender", "titlecase")
    non_null = df["Gender"].dropna()
    assert all(v == v.title() for v in non_null)


def test_c6_apply_trim_whitespace():
    from services.clean_service import apply_string_standardisation
    df = pd.DataFrame({"Name": ["  Alice  ", " Bob", "Charlie "]})
    df_clean = apply_string_standardisation(df.copy(), "Name", "trim_whitespace")
    assert df_clean["Name"].tolist() == ["Alice", "Bob", "Charlie"]


# C7 — Constant Columns

def test_c7_detect_constant_columns(dirty_df):
    from services.clean_service import detect_constant_columns
    result = detect_constant_columns(dirty_df)
    col_names = [r["column"] for r in result]
    assert "Status" in col_names


def test_c7_apply_drop_column(dirty_df):
    from services.clean_service import apply_drop_column
    df = apply_drop_column(dirty_df.copy(), "Status")
    assert "Status" not in df.columns


# C8 — Identifier Columns

def test_c8_detect_identifier_columns():
    from services.clean_service import detect_identifier_columns
    df = pd.DataFrame({
        "CustomerID": list(range(1, 101)),
        "Age":        [25] * 100,
    })
    result = detect_identifier_columns(df)
    flagged = [r["column"] for r in result]
    assert "CustomerID" in flagged
    assert "Age" not in flagged


# C9 — Negative Values

def test_c9_detect_non_negative_column():
    from services.clean_service import detect_negative_values
    df = pd.DataFrame({"Revenue": [100, -50, 200, -30, 150]})
    result = detect_negative_values(df)
    assert len(result) == 1
    assert result[0]["column"] == "Revenue"
    assert result[0]["negative_count"] == 2
    assert result[0]["semantic_class"] == "non_negative"


def test_c9_detect_signed_column():
    from services.clean_service import detect_negative_values
    df = pd.DataFrame({"temperature": [-10, -5, 0, 5, 10]})
    result = detect_negative_values(df)
    assert len(result) == 1
    assert result[0]["semantic_class"] == "signed"


def test_c9_detect_encoded_column():
    from services.clean_service import detect_negative_values
    df = pd.DataFrame({"label": [-1, 0, 1, -1, 1]})
    result = detect_negative_values(df)
    assert len(result) == 1
    assert result[0]["semantic_class"] == "encoded"


def test_c9_apply_set_zero():
    from services.clean_service import apply_negative_values
    df = pd.DataFrame({"Revenue": [100, -50, 200, -30, 150]})
    df_clean = apply_negative_values(df.copy(), "Revenue", "set_zero")
    assert (df_clean["Revenue"] < 0).sum() == 0
    assert df_clean["Revenue"].min() == 0


def test_c9_apply_set_nan():
    from services.clean_service import apply_negative_values
    df = pd.DataFrame({"Revenue": [100, -50, 200, -30, 150]})
    df_clean = apply_negative_values(df.copy(), "Revenue", "set_nan")
    assert df_clean["Revenue"].isna().sum() == 2
    assert (df_clean["Revenue"] < 0).sum() == 0


def test_c9_apply_abs_value():
    from services.clean_service import apply_negative_values
    df = pd.DataFrame({"Revenue": [100, -50, 200, -30, 150]})
    df_clean = apply_negative_values(df.copy(), "Revenue", "abs_value")
    assert (df_clean["Revenue"] < 0).sum() == 0
    assert df_clean["Revenue"].tolist() == [100, 50, 200, 30, 150]


def test_c9_apply_drop_rows():
    from services.clean_service import apply_negative_values
    df = pd.DataFrame({"Revenue": [100, -50, 200, -30, 150]})
    df_clean = apply_negative_values(df.copy(), "Revenue", "drop_rows")
    assert len(df_clean) == 3
    assert (df_clean["Revenue"] < 0).sum() == 0


def test_c9_apply_flag_only():
    from services.clean_service import apply_negative_values
    df = pd.DataFrame({"Revenue": [100, -50, 200, -30, 150]})
    df_clean = apply_negative_values(df.copy(), "Revenue", "flag_only")
    assert "Revenue_negative_flag" in df_clean.columns
    assert df_clean["Revenue_negative_flag"].sum() == 2
    assert (df_clean["Revenue"] < 0).sum() == 2


def test_c9_apply_keep():
    from services.clean_service import apply_negative_values
    df = pd.DataFrame({"Revenue": [100, -50, 200, -30, 150]})
    df_clean = apply_negative_values(df.copy(), "Revenue", "keep")
    assert df_clean["Revenue"].tolist() == [100, -50, 200, -30, 150]


def test_c9_issues_endpoint_filters_signed_columns(client):
    """
    Signed and encoded columns must NOT appear in the issues endpoint
    — only non_negative and ambiguous are actionable.
    """
    df = pd.DataFrame({
        "temperature": [-10, -5, 0, 5, 10],
        "Revenue":     [100, -50, 200, -30, 150],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]
    response = client.get(f"/api/clean/{session_id}/issues")
    assert response.status_code == 200
    neg_issues = response.json()["issues"]["negative_values"]
    col_names = [i["column"] for i in neg_issues]
    assert "Revenue" in col_names
    assert "temperature" not in col_names


# Integration Tests — router endpoints

def test_get_issues_returns_correct_shape(client, uploaded_session):
    response = client.get(f"/api/clean/{uploaded_session}/issues")
    assert response.status_code == 200
    data = response.json()
    assert "issues" in data
    assert "total_issues" in data
    expected_keys = [
        "missing_numerical", "missing_categorical", "duplicates",
        "outliers", "type_inconsistencies", "string_issues",
        "constant_columns", "identifier_columns", "negative_values",
    ]
    for key in expected_keys:
        assert key in data["issues"], f"Missing issue key: {key}"


def test_apply_and_undo_restores_original(client):
    dirty = pd.DataFrame({
        "Age":   [25, None, 35, None, 45],
        "Score": [80, 90, 85, 70, 95],
    })
    buf = io.BytesIO()
    dirty.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("dirty.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]

    apply_resp = client.post(
        f"/api/clean/{session_id}/apply",
        json={"issue_type": "missing_numerical", "column": "Age", "strategy": "median"},
    )
    assert apply_resp.status_code == 200
    assert apply_resp.json()["rows_after"] == 5

    undo_resp = client.post(f"/api/clean/{session_id}/undo")
    assert undo_resp.status_code == 200

    info = client.get(f"/api/dataset/{session_id}/info").json()
    assert info["rows"] == 5


def test_apply_logs_to_history(client):
    dirty = pd.DataFrame({
        "Age":   [25, None, 35, None, 45],
        "Score": [80, 90, 85, 70, 95],
    })
    buf = io.BytesIO()
    dirty.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("dirty.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]

    client.post(
        f"/api/clean/{session_id}/apply",
        json={"issue_type": "missing_numerical", "column": "Age", "strategy": "median"},
    )

    history_resp = client.get(f"/api/clean/{session_id}/history")
    assert history_resp.status_code == 200
    history = history_resp.json()["history"]
    assert len(history) == 1
    assert history[0]["issue_type"] == "missing_numerical"
    assert history[0]["strategy"] == "median"


def test_apply_drop_rows_reduces_count(client):
    dirty = pd.DataFrame({
        "Age":   [25, None, 35, None, 45],
        "Score": [80, 90, 85, 70, 95],
    })
    buf = io.BytesIO()
    dirty.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("dirty.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]

    apply_resp = client.post(
        f"/api/clean/{session_id}/apply",
        json={"issue_type": "missing_numerical", "column": "Age", "strategy": "drop_rows"},
    )
    assert apply_resp.status_code == 200
    assert apply_resp.json()["rows_after"] == 3


def test_apply_set_nan_returns_warning(client):
    dirty = pd.DataFrame({
        "Revenue": [100, -50, 200, -30, 150],
        "Score":   [80, 90, 85, 70, 95],
    })
    buf = io.BytesIO()
    dirty.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("dirty.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]

    apply_resp = client.post(
        f"/api/clean/{session_id}/apply",
        json={"issue_type": "negative_values", "column": "Revenue", "strategy": "set_nan"},
    )
    assert apply_resp.status_code == 200
    assert apply_resp.json()["warning"] is not None


def test_apply_flag_only_returns_warning(client):
    dirty = pd.DataFrame({
        "Revenue": [100, -50, 200, -30, 150],
        "Score":   [80, 90, 85, 70, 95],
    })
    buf = io.BytesIO()
    dirty.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("dirty.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]

    apply_resp = client.post(
        f"/api/clean/{session_id}/apply",
        json={"issue_type": "negative_values", "column": "Revenue", "strategy": "flag_only"},
    )
    assert apply_resp.status_code == 200
    assert apply_resp.json()["warning"] is not None


def test_apply_invalid_strategy_returns_400(client, uploaded_session):
    response = client.post(
        f"/api/clean/{uploaded_session}/apply",
        json={"issue_type": "missing_numerical", "column": "Age", "strategy": "invalid_strategy"},
    )
    assert response.status_code == 400


def test_apply_invalid_issue_type_returns_400(client, uploaded_session):
    response = client.post(
        f"/api/clean/{uploaded_session}/apply",
        json={"issue_type": "nonexistent_issue", "column": "Age", "strategy": "median"},
    )
    assert response.status_code == 400


def test_undo_without_snapshot_returns_404(client):
    df = pd.DataFrame({"A": [1, 2, 3]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]
    response = client.post(f"/api/clean/{session_id}/undo")
    assert response.status_code == 404


def test_preview_does_not_modify_working_csv(client):
    dirty = pd.DataFrame({
        "Age":   [25, None, 35, None, 45],
        "Score": [80, 90, 85, 70, 95],
    })
    buf = io.BytesIO()
    dirty.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("dirty.csv", buf.getvalue(), "text/csv")},
    )
    session_id = upload_resp.json()["session_id"]

    client.post(
        f"/api/clean/{session_id}/preview",
        json={"issue_type": "missing_numerical", "column": "Age", "strategy": "median"},
    )

    info = client.get(f"/api/dataset/{session_id}/info").json()
    assert info["rows"] == 5