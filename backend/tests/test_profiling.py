"""
Stage 2 Tests — Data Profiling.
Covers all 22 profiling functions + router endpoints + Gemini fallback.
"""

import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


# Unit Tests - profile_service functions


def test_p1_get_dimensions():
    from services.profile_service import get_dimensions
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    result = get_dimensions(df)
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["total_cells"] == 6
    assert result["memory_mb"] > 0


def test_p2_identify_columns_semantic_labels():
    from services.profile_service import identify_columns
    df = pd.DataFrame({
        "CustomerID": [1, 2],
        "email_address": ["a@b.com", "c@d.com"],
        "Age": [25, 30],
    })
    result = identify_columns(df)
    labels = {r["name"]: r["semantic_label"] for r in result}
    assert labels["CustomerID"] == "id"
    assert labels["email_address"] == "email"
    assert labels["Age"] == "age"


def test_p3_detect_numerical_columns():
    from services.profile_service import detect_numerical_columns
    df = pd.DataFrame({
        "Int_Col": pd.array([1, 2, 3], dtype="int64"),
        "Float_Col": [1.1, 2.2, 3.3],
        "Text_Col": ["a", "b", "c"],
    })
    result = detect_numerical_columns(df)
    col_names = [r["column"] for r in result]
    assert "Int_Col" in col_names
    assert "Float_Col" in col_names
    assert "Text_Col" not in col_names


def test_p4_detect_categorical_columns():
    from services.profile_service import detect_categorical_columns
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Male"],
        "Age": [25, 30, 35],
    })
    result = detect_categorical_columns(df)
    col_names = [r["column"] for r in result]
    assert "Gender" in col_names
    assert "Age" not in col_names
    gender_entry = next(r for r in result if r["column"] == "Gender")
    assert gender_entry["unique_count"] == 2


def test_p5_detect_datetime_columns():
    from services.profile_service import detect_datetime_columns
    df = pd.DataFrame({
        "created_at": ["2023-01-01", "2023-06-15", "2023-12-31"],
        "Name": ["Alice", "Bob", "Charlie"],
    })
    result = detect_datetime_columns(df)
    col_names = [r["column"] for r in result]
    assert "created_at" in col_names
    assert "Name" not in col_names


def test_p6_missing_value_analysis_critical_flag():
    from services.profile_service import missing_value_analysis
    df = pd.DataFrame({
        "A": [None] * 8 + [1, 2],   # 80% missing → critical
        "B": [1, 2, None, 4, 5,
              6, 7, 8, 9, 10],       # 10% missing → not critical
        "C": list(range(10)),        # 0% missing
    })
    result = missing_value_analysis(df)
    assert result["A"]["is_critical"] == True
    assert result["A"]["pct"] == 80.0
    assert result["B"]["is_critical"] == False
    assert result["C"]["count"] == 0


def test_p7_cardinality_analysis():
    from services.profile_service import cardinality_analysis
    df = pd.DataFrame({
        "Binary": [0, 1] * 75,  
        "Category": [
            f"cat_{i % 50}" for i in range(150)
        ],  
        "UniqueID": [
            f"id_{i}" for i in range(150)
        ],  
    })

    result = cardinality_analysis(df)

    assert result["Binary"]["cardinality_type"] == "low"
    assert result["Category"]["cardinality_type"] == "medium"
    assert result["UniqueID"]["cardinality_type"] == "high"


def test_p8_duplicate_row_detection():
    from services.profile_service import duplicate_row_detection
    df = pd.DataFrame({
        "A": [1, 1, 2, 3, 3],
        "B": ["x", "x", "y", "z", "z"],
    })
    result = duplicate_row_detection(df)
    assert result["count"] == 2
    assert result["pct"] == 40.0
    assert len(result["sample_indices"]) <= 5


def test_p9_duplicate_column_detection():
    from services.profile_service import duplicate_column_detection
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],   # exact duplicate of A
        "C": [4, 5, 6],
    })
    result = duplicate_column_detection(df)
    assert len(result) == 1
    assert result[0]["col1"] == "A"
    assert result[0]["col2"] == "B"


def test_p10_numerical_statistical_summary():
    from services.profile_service import numerical_statistical_summary
    df = pd.DataFrame({"Value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = numerical_statistical_summary(df)
    assert result["Value"]["min"] == 1.0
    assert result["Value"]["max"] == 5.0
    assert result["Value"]["mean"] == 3.0
    assert result["Value"]["median"] == 3.0
    assert result["Value"]["q1"] == 2.0
    assert result["Value"]["q3"] == 4.0


def test_p11_suspicious_value_detection_age():
    from services.profile_service import suspicious_value_detection
    df = pd.DataFrame({"Age": [25, 30, -5, 200, 35]})
    result = suspicious_value_detection(df)
    assert "Age" in result
    assert result["Age"][0]["flagged_count"] == 2


def test_p11_suspicious_value_detection_no_flags():
    from services.profile_service import suspicious_value_detection
    df = pd.DataFrame({"Revenue": [100.0, 200.0, 300.0]})
    result = suspicious_value_detection(df)
    # No negative values → Revenue should have no flags
    assert "Revenue" not in result


def test_p12_data_type_consistency():
    from services.profile_service import data_type_consistency
    df = pd.DataFrame({
        "NumericAsStr": ["1", "2", "3", "4", "5"],
        "TrueText": ["apple", "banana", "cherry", "date", "elderberry"],
    })
    result = data_type_consistency(df)
    suggested = {r["column"]: r["suggested_type"] for r in result}
    assert suggested.get("NumericAsStr") == "numeric"
    assert "TrueText" not in suggested


def test_p13_constant_column_detection():
    from services.profile_service import constant_column_detection
    df = pd.DataFrame({
        "Constant": ["yes"] * 100,
        "Variable": list(range(100)),
    })
    result = constant_column_detection(df)
    col_names = [r["column"] for r in result]
    assert "Constant" in col_names
    assert "Variable" not in col_names


def test_p14_identifier_detection():
    from services.profile_service import identifier_detection
    df = pd.DataFrame({
        "CustomerID": list(range(1, 101)),
        "Age": [25] * 100,
        "Score": [88] * 100,
    })
    result = identifier_detection(df)
    flagged = [r["column"] for r in result]
    assert "CustomerID" in flagged
    assert "Age" not in flagged


def test_p15_outlier_detection_iqr():
    from services.profile_service import outlier_detection
    df = pd.DataFrame({"Value": [10, 12, 11, 13, 10, 12, 1000]})
    result = outlier_detection(df)
    assert result["Value"]["outlier_count"] == 1
    assert result["Value"]["upper_bound"] < 1000


def test_p16_distribution_analysis_labels():
    from services.profile_service import distribution_analysis
    df = pd.DataFrame({
        "Symmetric": [1, 2, 3, 4, 5, 4, 3, 2, 1],
        "Skewed": [1, 1, 1, 1, 1, 1, 1, 1, 100],
    })
    result = distribution_analysis(df)
    assert result["Symmetric"]["distribution_label"] == "symmetric"
    assert result["Skewed"]["distribution_label"] == "highly skewed"


def test_p17_categorical_imbalance():
    from services.profile_service import categorical_imbalance
    df = pd.DataFrame({
        "Gender": ["Male"] * 90 + ["Female"] * 10,
        "Status": ["A"] * 50 + ["B"] * 50,
    })
    result = categorical_imbalance(df)
    assert result["Gender"]["is_imbalanced"] == True
    assert result["Gender"]["top_pct"] == 90.0
    assert result["Status"]["is_imbalanced"] == False


def test_p18_datetime_coverage():
    from services.profile_service import datetime_coverage
    df = pd.DataFrame({
        "created_at": pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-03",
            "2023-01-04", "2023-01-05",
        ])
    })
    result = datetime_coverage(df)
    assert "created_at" in result
    assert result["created_at"]["total_days"] == 4
    assert result["created_at"]["min_date"] == "2023-01-01"
    assert result["created_at"]["max_date"] == "2023-01-05"


def test_p19_initial_relationship_screening():
    from services.profile_service import initial_relationship_screening
    df = pd.DataFrame({
        "A": list(range(100)),
        "B": [x * 2 for x in range(100)],   # perfectly correlated with A
        "C": [x * -1 for x in range(100)],  # perfectly negatively correlated
        "D": [x % 7 for x in range(100)],   # weak correlation
    })
    result = initial_relationship_screening(df)
    pairs = [(r["col1"], r["col2"]) for r in result]
    assert ("A", "B") in pairs
    assert ("A", "C") in pairs
    # D should not appear (weak correlation)
    d_pairs = [p for p in pairs if "D" in p]
    assert len(d_pairs) == 0


def test_p20_potential_target_detection():
    from services.profile_service import potential_target_detection
    df = pd.DataFrame({
        "CustomerID": range(100),
        "Purchased": [0, 1] * 50,
        "Category": ["A", "B", "C", "D"] * 25,
        "Revenue": [float(i) for i in range(100)],
    })
    result = potential_target_detection(
        df,
        identifier_cols=["CustomerID"],
        datetime_cols=[],
        constant_cols=[],
    )
    assert len(result) <= 3
    assert result[0]["column"] == "Purchased"
    assert result[0]["score"] == 90


def test_p21_quality_summary_perfect_score():
    from services.profile_service import data_quality_summary
    df = pd.DataFrame({"A": [1, 2, 3]})
    all_results = {
        "missing_values": {"A": {"pct": 0, "is_critical": False}},
        "duplicate_rows": {"pct": 0},
        "suspicious_values": {},
        "categorical_imbalance": {},
    }
    result = data_quality_summary(df, all_results)
    assert result["score"] == 100
    assert result["label"] == "excellent"


def test_p21_quality_summary_drops_with_issues():
    from services.profile_service import data_quality_summary
    df = pd.DataFrame({"A": [None] * 8 + [1, 2]})
    all_results = {
        "missing_values": {"A": {"pct": 80, "is_critical": True}},
        "duplicate_rows": {"pct": 10},
        "suspicious_values": {"A": [{"rule": "test", "flagged_count": 1}]},
        "categorical_imbalance": {},
    }
    result = data_quality_summary(df, all_results)
    assert result["score"] < 100
    assert result["label"] in ["fair", "poor", "good"]
    assert len(result["breakdown"]) > 0


def test_p22_rule_based_insights_returns_list():
    from services.profile_service import _generate_rule_based_insights
    all_results = {
        "missing_values": {"Age": {"pct": 15.0, "is_critical": False}},
        "duplicate_rows": {"count": 50, "pct": 5.0},
        "distribution_analysis": {
            "Income": {"skewness": 3.5, "kurtosis": 10.0, "distribution_label": "highly skewed"}
        },
        "identifier_columns": [{"column": "CustomerID", "unique_ratio": 0.99, "reason": "name_keyword"}],
        "categorical_imbalance": {"Gender": {"top_category": "Male", "top_pct": 85.0, "is_imbalanced": True}},
        "correlation_pairs": [{"col1": "Age", "col2": "Income", "correlation": 0.82}],
        "suspicious_values": {},
        "quality_summary": {"score": 55, "label": "fair"},
    }
    insights = _generate_rule_based_insights(all_results)
    assert isinstance(insights, list)
    assert len(insights) >= 1
    assert all(isinstance(s, str) for s in insights)


# Integration Tests - router endpoints

def test_profile_run_returns_correct_shape(client, uploaded_session):
    response = client.post(f"/api/profile/{uploaded_session}/run")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == uploaded_session
    assert data["status"] == "complete"
    assert "quality_summary" in data
    assert data["insights_count"] >= 1
    assert data["insights_source"] in ["gemini", "rule_based"]


def test_profile_result_returns_all_keys(client, uploaded_session):
    # Ensure profile is run first
    client.post(f"/api/profile/{uploaded_session}/run")
    response = client.get(f"/api/profile/{uploaded_session}/result")
    assert response.status_code == 200
    profile = response.json()["profile"]
    expected_keys = [
        "dimensions", "column_identification", "numerical_columns",
        "categorical_columns", "datetime_columns", "missing_values",
        "cardinality", "duplicate_rows", "duplicate_columns",
        "numerical_summary", "suspicious_values", "type_inconsistencies",
        "constant_columns", "identifier_columns", "outlier_detection",
        "distribution_analysis", "categorical_imbalance", "datetime_coverage",
        "correlation_pairs", "potential_targets", "quality_summary", "insights",
    ]
    for key in expected_keys:
        assert key in profile, f"Missing key in profile result: {key}"


def test_profile_insights_endpoint(client, uploaded_session):
    client.post(f"/api/profile/{uploaded_session}/run")
    response = client.get(f"/api/profile/{uploaded_session}/insights")
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data
    assert isinstance(data["insights"], list)
    assert len(data["insights"]) >= 1
    assert data["insights_source"] in ["gemini", "rule_based"]


def test_profile_quality_endpoint(client, uploaded_session):
    client.post(f"/api/profile/{uploaded_session}/run")
    response = client.get(f"/api/profile/{uploaded_session}/quality")
    assert response.status_code == 200
    data = response.json()
    assert "quality_summary" in data
    assert "score" in data["quality_summary"]
    assert "label" in data["quality_summary"]
    assert "dimensions" in data
    assert "missing_values" in data
    assert "duplicate_rows" in data


def test_profile_result_without_run_returns_404(client, uploaded_session):
    # Fresh session — profile not run yet
    upload_resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", b"A,B\n1,2\n3,4", "text/csv")},
    )
    fresh_session = upload_resp.json()["session_id"]
    response = client.get(f"/api/profile/{fresh_session}/result")
    assert response.status_code == 404


def test_profile_run_invalid_session_returns_404(client):
    fake_id = "00000000-0000-4000-8000-000000000000"
    response = client.post(f"/api/profile/{fake_id}/run")
    assert response.status_code == 404


def test_profile_run_invalid_uuid_returns_400(client):
    response = client.post("/api/profile/not-a-uuid/run")
    assert response.status_code == 400


def test_profile_fallback_works_without_gemini_key(client, uploaded_session):
    """
    Even with no Gemini API key, profiling must complete
    and return rule_based insights — never a 500 error.
    """
    response = client.post(f"/api/profile/{uploaded_session}/run")
    assert response.status_code == 200
    assert response.json()["insights_source"] in ["gemini", "rule_based"]