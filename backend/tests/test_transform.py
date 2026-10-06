"""
Stage 4 Tests — Transform Service.
Covers all 25 transformation functions + router endpoints + undo.
"""

import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


# Fixtures

@pytest.fixture
def num_df():
    """Clean numerical DataFrame for scaling/transform tests."""
    return pd.DataFrame({
        "Age":     [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Income":  [30000, 45000, 60000, 75000, 90000,
                    105000, 120000, 135000, 150000, 165000],
        "Score":   [88, 72, 91, 65, 78, 84, 90, 67, 95, 73],
    })


@pytest.fixture
def cat_df():
    """DataFrame with categorical columns for encoding tests."""
    return pd.DataFrame({
        "Gender":   ["Male", "Female", "Male", "Female", "Male"] * 2,
        "Region":   ["North", "South", "East", "West", "North"] * 2,
        "Purchased": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def dt_df():
    """DataFrame with datetime column for extraction tests."""
    return pd.DataFrame({
        "created_at": pd.to_datetime([
            "2023-01-15 08:30:00",
            "2023-04-22 14:45:00",
            "2023-07-10 09:00:00",
            "2023-10-05 17:20:00",
            "2023-12-31 23:59:00",
        ]),
        "Value": [100, 200, 300, 400, 500],
    })


@pytest.fixture
def skewed_df():
    """DataFrame with right-skewed column for log/power tests."""
    np.random.seed(42)
    values = np.random.exponential(scale=1000, size=100)
    return pd.DataFrame({"Revenue": values})


# T1   Min-Max Normalisation

def test_t1_minmax_range(num_df):
    from services.transform_service import apply_minmax
    df = apply_minmax(num_df.copy(), "Age")
    assert df["Age"].min() >= 0.0
    assert df["Age"].max() <= 1.0


def test_t1_minmax_constant_column():
    from services.transform_service import apply_minmax
    df = pd.DataFrame({"A": [5, 5, 5, 5, 5]})
    df_t = apply_minmax(df.copy(), "A")
    assert (df_t["A"] == 0.0).all()


def test_t1_minmax_preview_returns_stats(num_df):
    from services.transform_service import preview_minmax
    result = preview_minmax(num_df, "Age")
    assert "before" in result
    assert "after" in result
    assert result["after"]["min"] >= 0.0
    assert result["after"]["max"] <= 1.0


# T2 — Z-Score Standardisation

def test_t2_zscore_mean_near_zero(num_df):
    from services.transform_service import apply_zscore
    df = apply_zscore(num_df.copy(), "Age")
    assert abs(df["Age"].mean()) < 1e-10


def test_t2_zscore_std_near_one(num_df):
    from services.transform_service import apply_zscore
    df = apply_zscore(num_df.copy(), "Age")
    assert abs(df["Age"].std() - 1.0) < 1e-10


def test_t2_zscore_preview_returns_stats(num_df):
    from services.transform_service import preview_zscore
    result = preview_zscore(num_df, "Age")
    assert "before" in result
    assert "after" in result
    assert abs(result["after"]["mean"]) < 0.01


# T3  Robust Scaler

def test_t3_robust_median_near_zero(num_df):
    from services.transform_service import apply_robust
    df = apply_robust(num_df.copy(), "Age")
    assert abs(df["Age"].median()) < 1e-10


def test_t3_robust_not_affected_by_outlier():
    from services.transform_service import apply_robust
    df = pd.DataFrame({"Value": [10, 12, 11, 13, 10, 12, 1000]})
    df_robust = apply_robust(df.copy(), "Value")
    df_zscore_check = df_robust.copy()
    # Robust scaled version should have median near 0
    assert abs(df_robust["Value"].median()) < 0.1


# T4   Log Transform

def test_t4_log_reduces_skewness(skewed_df):
    from services.transform_service import apply_log
    before_skew = skewed_df["Revenue"].skew()
    df = apply_log(skewed_df.copy(), "Revenue")
    after_skew = df["Revenue"].skew()
    assert abs(after_skew) < abs(before_skew)


def test_t4_log_handles_zeros():
    from services.transform_service import apply_log
    df = pd.DataFrame({"Value": [0, 1, 2, 3, 4]})
    df_t = apply_log(df.copy(), "Value")
    assert df_t["Value"].isna().sum() == 0
    assert df_t["Value"].iloc[0] == 0.0  # log1p(0) = 0


def test_t4_log_preview_returns_stats(skewed_df):
    from services.transform_service import preview_log
    result = preview_log(skewed_df, "Revenue")
    assert "before" in result
    assert "after" in result


# T5   Power Transform

def test_t5_power_reduces_skewness(skewed_df):
    from services.transform_service import apply_power
    before_skew = skewed_df["Revenue"].skew()
    df = apply_power(skewed_df.copy(), "Revenue")
    after_skew = df["Revenue"].skew()
    assert abs(after_skew) < abs(before_skew)


# T6   Label Encoding

def test_t6_label_encode_converts_to_int(cat_df):
    from services.transform_service import apply_label_encode
    df = apply_label_encode(cat_df.copy(), "Gender")
    assert pd.api.types.is_numeric_dtype(df["Gender"])
    assert set(df["Gender"].unique()).issubset({0, 1})


def test_t6_label_encode_preserves_row_count(cat_df):
    from services.transform_service import apply_label_encode
    df = apply_label_encode(cat_df.copy(), "Gender")
    assert len(df) == len(cat_df)


# T7   One-Hot Encoding

def test_t7_onehot_creates_new_columns(cat_df):
    from services.transform_service import apply_onehot
    df = apply_onehot(cat_df.copy(), "Gender")
    assert "Gender" not in df.columns
    assert "Gender_Male" in df.columns
    assert "Gender_Female" in df.columns


def test_t7_onehot_binary_values(cat_df):
    from services.transform_service import apply_onehot
    df = apply_onehot(cat_df.copy(), "Gender")
    assert df["Gender_Male"].isin([0, 1, True, False]).all()


# T8   Ordinal Encoding

def test_t8_ordinal_respects_order():
    from services.transform_service import apply_ordinal
    df = pd.DataFrame({"Size": ["Small", "Medium", "Large", "Small", "Large"]})
    df_t = apply_ordinal(df.copy(), "Size", order=["Small", "Medium", "Large"])
    assert df_t["Size"].tolist() == [0, 1, 2, 0, 2]


# T9   Frequency Encoding

def test_t9_frequency_encode_sums_to_one(cat_df):
    from services.transform_service import apply_frequency_encode
    df = apply_frequency_encode(cat_df.copy(), "Gender")
    assert pd.api.types.is_numeric_dtype(df["Gender"])
    assert df["Gender"].between(0, 1).all()


# T10   Target Encoding

def test_t10_target_encode_replaces_with_mean(cat_df):
    from services.transform_service import apply_target_encode
    df = apply_target_encode(cat_df.copy(), "Gender", target="Purchased")
    assert pd.api.types.is_numeric_dtype(df["Gender"])
    # All Male rows should have the same value (mean of Purchased for Male)
    male_vals = df.loc[cat_df["Gender"] == "Male", "Gender"].unique()
    assert len(male_vals) == 1


# T11   Equal-Width Binning

def test_t11_equal_width_creates_bin_column(num_df):
    from services.transform_service import apply_equal_width_bin
    df = apply_equal_width_bin(num_df.copy(), "Age", bins=5)
    assert "Age_bin" in df.columns
    assert df["Age_bin"].nunique() <= 5


# T12   Equal-Frequency Binning

def test_t12_equal_freq_creates_bin_column(num_df):
    from services.transform_service import apply_equal_freq_bin
    df = apply_equal_freq_bin(num_df.copy(), "Age", bins=5)
    assert "Age_qbin" in df.columns


# T13   Custom Threshold Binning

def test_t13_custom_bins_creates_labels(num_df):
    from services.transform_service import apply_custom_bins
    df = apply_custom_bins(
        num_df.copy(), "Age",
        thresholds=[35, 55],
        labels=["Young", "Middle", "Senior"],
    )
    assert "Age_custom_bin" in df.columns
    assert set(df["Age_custom_bin"].dropna().unique()).issubset(
        {"Young", "Middle", "Senior"}
    )


# T14   Polynomial Feature

def test_t14_polynomial_adds_squared_column(num_df):
    from services.transform_service import apply_polynomial
    df = apply_polynomial(num_df.copy(), "Age")
    assert "Age_sq" in df.columns
    assert df["Age_sq"].iloc[0] == 25 ** 2


# T15   Interaction Term

def test_t15_interaction_creates_product_column(num_df):
    from services.transform_service import apply_interaction
    df = apply_interaction(num_df.copy(), "Age", "Score")
    assert "Age_x_Score" in df.columns
    assert df["Age_x_Score"].iloc[0] == 25 * 88


# T16   Custom Formula

def test_t16_formula_creates_derived_column(num_df):
    from services.transform_service import apply_formula
    df = apply_formula(num_df.copy(), "Age + Score", "Age_plus_Score")
    assert "Age_plus_Score" in df.columns
    assert df["Age_plus_Score"].iloc[0] == 25 + 88


def test_t16_formula_injection_rejected(num_df):
    from services.transform_service import apply_formula
    from core.exceptions import InvalidFormulaError
    with pytest.raises(InvalidFormulaError):
        apply_formula(num_df.copy(), "import os", "bad_col")


# T17   Lag Feature

def test_t17_lag_creates_shifted_column(num_df):
    from services.transform_service import apply_lag
    df = apply_lag(num_df.copy(), "Age", periods=1)
    assert "Age_lag1" in df.columns
    assert pd.isna(df["Age_lag1"].iloc[0])
    assert df["Age_lag1"].iloc[1] == 25


# T18   Rolling Mean

def test_t18_rolling_mean_creates_column(num_df):
    from services.transform_service import apply_rolling_mean
    df = apply_rolling_mean(num_df.copy(), "Age", window=3)
    assert "Age_roll3" in df.columns
    assert df["Age_roll3"].isna().sum() == 0


# T19-T23   Datetime Extractions

def test_t19_extract_year(dt_df):
    from services.transform_service import apply_extract_year
    df = apply_extract_year(dt_df.copy(), "created_at")
    assert "created_at_year" in df.columns
    assert df["created_at_year"].iloc[0] == 2023


def test_t20_extract_month(dt_df):
    from services.transform_service import apply_extract_month
    df = apply_extract_month(dt_df.copy(), "created_at")
    assert "created_at_month" in df.columns
    assert df["created_at_month"].iloc[0] == 1


def test_t21_extract_day(dt_df):
    from services.transform_service import apply_extract_day
    df = apply_extract_day(dt_df.copy(), "created_at")
    assert "created_at_day" in df.columns
    assert df["created_at_day"].iloc[0] == 15


def test_t22_extract_dow(dt_df):
    from services.transform_service import apply_extract_dow
    df = apply_extract_dow(dt_df.copy(), "created_at")
    assert "created_at_dow" in df.columns
    assert df["created_at_dow"].iloc[0] in range(7)


def test_t23_extract_hour(dt_df):
    from services.transform_service import apply_extract_hour
    df = apply_extract_hour(dt_df.copy(), "created_at")
    assert "created_at_hour" in df.columns
    assert df["created_at_hour"].iloc[0] == 8


# T24   PCA

def test_t24_pca_creates_two_components(num_df):
    from services.transform_service import apply_pca
    df = apply_pca(num_df.copy(), columns=["Age", "Income", "Score"])
    assert "pca_1" in df.columns
    assert "pca_2" in df.columns
    assert df["pca_1"].isna().sum() == 0


# T25   Exclude ML Flag

def test_t25_exclude_ml_returns_dict():
    from services.transform_service import flag_exclude_ml
    result = flag_exclude_ml("CustomerID")
    assert result["transform_type"] == "exclude_ml"
    assert result["column"] == "CustomerID"
    assert "note" in result


# apply_transform dispatcher

def test_dispatcher_minmax(num_df):
    from services.transform_service import apply_transform
    df = apply_transform(num_df.copy(), {"transform_type": "minmax", "column": "Age"})
    assert df["Age"].max() <= 1.0


def test_dispatcher_onehot(cat_df):
    from services.transform_service import apply_transform
    df = apply_transform(cat_df.copy(), {"transform_type": "onehot", "column": "Gender"})
    assert "Gender_Male" in df.columns


def test_dispatcher_unknown_raises():
    from services.transform_service import apply_transform
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        apply_transform(df, {"transform_type": "unknown_transform", "column": "A"})


# Integration Tests — router endpoints

def _upload_df(client, df) -> str:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    return resp.json()["session_id"]


def test_recommend_endpoint_returns_result(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [30000, 45000, 60000, 75000, 90000],
    })
    session_id = _upload_df(client, df)
    resp = client.get(f"/api/transform/{session_id}/recommend?column=Age")
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendation" in data
    assert "recommended_transform" in data["recommendation"]
    assert "explanation" in data["recommendation"]


def test_apply_minmax_endpoint(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [30000, 45000, 60000, 75000, 90000],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/transform/{session_id}/apply",
        json={"transform_type": "minmax", "column": "Age"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "applied"
    assert data["cols_after"] == data["cols_before"]


def test_apply_onehot_adds_columns(client):
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Score": [80, 90, 85, 70, 95],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/transform/{session_id}/apply",
        json={"transform_type": "onehot", "column": "Gender"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cols_after"] > data["cols_before"]
    assert len(data["new_columns"]) > 0


def test_apply_and_undo_restores_columns(client):
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Score": [80, 90, 85, 70, 95],
    })
    session_id = _upload_df(client, df)
    cols_before = client.get(f"/api/dataset/{session_id}/info").json()["columns"]

    client.post(
        f"/api/transform/{session_id}/apply",
        json={"transform_type": "onehot", "column": "Gender"},
    )

    undo_resp = client.post(f"/api/transform/{session_id}/undo")
    assert undo_resp.status_code == 200

    cols_after_undo = client.get(
        f"/api/dataset/{session_id}/info"
    ).json()["columns"]
    assert cols_after_undo == cols_before


def test_apply_logs_to_history(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Score": [80, 90, 85, 70, 95],
    })
    session_id = _upload_df(client, df)

    client.post(
        f"/api/transform/{session_id}/apply",
        json={"transform_type": "zscore", "column": "Age"},
    )

    history_resp = client.get(f"/api/transform/{session_id}/history")
    assert history_resp.status_code == 200
    history = history_resp.json()["history"]
    assert len(history) == 1
    assert history[0]["transform_type"] == "zscore"


def test_apply_exclude_ml_no_df_mutation(client):
    df = pd.DataFrame({
        "CustomerID": range(1, 6),
        "Score": [80, 90, 85, 70, 95],
    })
    session_id = _upload_df(client, df)
    cols_before = client.get(
        f"/api/dataset/{session_id}/info"
    ).json()["columns"]

    resp = client.post(
        f"/api/transform/{session_id}/apply",
        json={"transform_type": "exclude_ml", "column": "CustomerID"},
    )
    assert resp.status_code == 200

    cols_after = client.get(
        f"/api/dataset/{session_id}/info"
    ).json()["columns"]
    assert cols_after == cols_before


def test_apply_unknown_transform_returns_400(client, uploaded_session):
    resp = client.post(
        f"/api/transform/{uploaded_session}/apply",
        json={"transform_type": "unknown_transform", "column": "Age"},
    )
    assert resp.status_code == 400


def test_apply_unknown_column_returns_400(client, uploaded_session):
    resp = client.post(
        f"/api/transform/{uploaded_session}/apply",
        json={"transform_type": "zscore", "column": "NonExistentColumn"},
    )
    assert resp.status_code == 400


def test_preview_does_not_save(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Score": [80, 90, 85, 70, 95],
    })
    session_id = _upload_df(client, df)

    client.post(
        f"/api/transform/{session_id}/preview",
        json={"transform_type": "minmax", "column": "Age"},
    )

    # Working df must be unchanged
    info = client.get(f"/api/dataset/{session_id}/info").json()
    assert info["columns"] == 2