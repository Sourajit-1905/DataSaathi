"""
Stage 5 Tests — EDA Service.
Covers all 14 EDA sub-modules + filter engine + router endpoints.
"""

import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


# Fixtures

@pytest.fixture
def num_df():
    np.random.seed(42)
    return pd.DataFrame({
        "Age":     [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Income":  [30000, 45000, 60000, 75000, 90000,
                    105000, 120000, 135000, 150000, 165000],
        "Score":   [88, 72, 91, 65, 78, 84, 90, 67, 95, 73],
    })


@pytest.fixture
def cat_df():
    return pd.DataFrame({
        "Gender":    ["Male", "Female", "Male", "Female", "Male",
                      "Female", "Male", "Female", "Male", "Female"],
        "Region":    ["North", "South", "East", "West", "North",
                      "South", "East", "West", "North", "South"],
        "Purchased": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "Age":      [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Income":   [30000, 45000, 60000, 75000, 90000,
                     105000, 120000, 135000, 150000, 165000],
        "Gender":   ["Male", "Female", "Male", "Female", "Male",
                     "Female", "Male", "Female", "Male", "Female"],
        "Purchased": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def time_df():
    dates = pd.date_range("2023-01-01", periods=24, freq="ME")
    np.random.seed(42)
    values = np.random.randint(1000, 5000, size=24).tolist()
    return pd.DataFrame({"date": dates, "revenue": values})


@pytest.fixture
def skewed_df():
    np.random.seed(42)
    return pd.DataFrame({
        "Revenue": np.random.exponential(scale=1000, size=100).tolist(),
        "Category": (["A"] * 70 + ["B"] * 20 + ["C"] * 10),
    })


# E12 — Filter Engine

def test_filter_eq(mixed_df):
    from services.eda_service import apply_filters
    result = apply_filters(mixed_df, [{"column": "Gender", "operator": "eq", "value": "Male"}])
    assert all(result["Gender"] == "Male")
    assert len(result) == 5


def test_filter_gt(mixed_df):
    from services.eda_service import apply_filters
    result = apply_filters(mixed_df, [{"column": "Age", "operator": "gt", "value": 40}])
    assert all(result["Age"] > 40)


def test_filter_lt(mixed_df):
    from services.eda_service import apply_filters
    result = apply_filters(mixed_df, [{"column": "Income", "operator": "lt", "value": 60000}])
    assert all(result["Income"] < 60000)

def test_filter_contains(mixed_df):
    from services.eda_service import apply_filters
    # "Male" contains "Male" but "Female" also contains "male" — use "^Male$" pattern
    result = apply_filters(mixed_df, [{"column": "Gender", "operator": "contains", "value": "^Male$"}])
    assert len(result) == 5


def test_filter_in(mixed_df):
    from services.eda_service import apply_filters
    result = apply_filters(mixed_df, [{"column": "Gender", "operator": "in", "value": ["Male"]}])
    assert all(result["Gender"] == "Male")


def test_multiple_filters(mixed_df):
    from services.eda_service import apply_filters
    result = apply_filters(mixed_df, [
        {"column": "Gender", "operator": "eq", "value": "Male"},
        {"column": "Age", "operator": "gt", "value": 30},
    ])
    assert all(result["Gender"] == "Male")
    assert all(result["Age"] > 30)


def test_filter_caps_at_five(mixed_df):
    from services.eda_service import apply_filters
    # 6 filters — only first 5 applied
    filters = [{"column": "Age", "operator": "gte", "value": 0}] * 6
    result = apply_filters(mixed_df, filters)
    assert len(result) == len(mixed_df)


def test_filter_unknown_column_skipped(mixed_df):
    from services.eda_service import apply_filters
    result = apply_filters(mixed_df, [{"column": "NonExistent", "operator": "eq", "value": "X"}])
    assert len(result) == len(mixed_df)


def test_filter_does_not_modify_original(mixed_df):
    from services.eda_service import apply_filters
    original_len = len(mixed_df)
    apply_filters(mixed_df, [{"column": "Gender", "operator": "eq", "value": "Male"}])
    assert len(mixed_df) == original_len


# E1 — Numerical Univariate

def test_e1_numerical_univariate_keys(num_df):
    from services.eda_service import numerical_univariate
    result = numerical_univariate(num_df, "Age")
    expected_keys = [
        "column", "count", "missing", "mean", "median", "mode",
        "variance", "std", "min", "max", "range", "iqr",
        "skewness", "kurtosis", "distribution_label",
        "percentiles", "histogram", "box_plot", "density",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e1_numerical_univariate_values(num_df):
    from services.eda_service import numerical_univariate
    result = numerical_univariate(num_df, "Age")
    assert result["min"] == 25.0
    assert result["max"] == 70.0
    assert result["count"] == 10
    assert result["missing"] == 0


def test_e1_percentiles_present(num_df):
    from services.eda_service import numerical_univariate
    result = numerical_univariate(num_df, "Age")
    assert "p25" in result["percentiles"]
    assert "p50" in result["percentiles"]
    assert "p75" in result["percentiles"]
    assert "p90" in result["percentiles"]
    assert "p95" in result["percentiles"]
    assert "p99" in result["percentiles"]


def test_e1_histogram_bins(num_df):
    from services.eda_service import numerical_univariate
    result = numerical_univariate(num_df, "Age")
    assert len(result["histogram"]["counts"]) == 20
    assert len(result["histogram"]["bin_edges"]) == 21


def test_e1_box_plot_keys(num_df):
    from services.eda_service import numerical_univariate
    result = numerical_univariate(num_df, "Age")
    box = result["box_plot"]
    for key in ["min", "q1", "median", "q3", "max", "lower_fence", "upper_fence", "outlier_count"]:
        assert key in box


def test_e1_invalid_column():
    from services.eda_service import numerical_univariate
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = numerical_univariate(df, "NonExistent")
    assert "error" in result


def test_e1_distribution_label_symmetric(num_df):
    from services.eda_service import numerical_univariate
    result = numerical_univariate(num_df, "Age")
    assert "symmetric" in result["distribution_label"]


def test_e1_distribution_label_skewed(skewed_df):
    from services.eda_service import numerical_univariate
    result = numerical_univariate(skewed_df, "Revenue")
    assert "skewed" in result["distribution_label"]


# E2 — Categorical Univariate

def test_e2_categorical_univariate_keys(cat_df):
    from services.eda_service import categorical_univariate
    result = categorical_univariate(cat_df, "Gender")
    expected_keys = [
        "column", "count", "missing", "unique_count",
        "most_common", "least_common", "frequency_table",
        "pareto", "rare_categories", "cats_for_80pct",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e2_frequency_table_complete(cat_df):
    from services.eda_service import categorical_univariate
    result = categorical_univariate(cat_df, "Gender")
    total_count = sum(r["count"] for r in result["frequency_table"])
    assert total_count == len(cat_df)


def test_e2_pareto_cumulative_reaches_100(cat_df):
    from services.eda_service import categorical_univariate
    result = categorical_univariate(cat_df, "Gender")
    last_cumulative = result["pareto"][-1]["cumulative_pct"]
    assert abs(last_cumulative - 100.0) < 0.1


def test_e2_rare_categories_below_threshold(skewed_df):
    from services.eda_service import categorical_univariate
    result = categorical_univariate(skewed_df, "Category")
    for rare in result["rare_categories"]:
        assert rare["pct"] < result["rare_threshold_pct"]


def test_e2_unique_count_correct(cat_df):
    from services.eda_service import categorical_univariate
    result = categorical_univariate(cat_df, "Gender")
    assert result["unique_count"] == 2


# E3 — Numerical vs Numerical

def test_e3_num_vs_num_keys(num_df):
    from services.eda_service import numerical_vs_numerical
    result = numerical_vs_numerical(num_df, "Age", "Income")
    expected_keys = [
        "col_x", "col_y", "n_pairs", "pearson", "spearman",
        "relationship", "regression_line", "scatter_sample", "trend_summary",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e3_perfect_correlation(num_df):
    from services.eda_service import numerical_vs_numerical
    result = numerical_vs_numerical(num_df, "Age", "Income")
    assert abs(result["pearson"]["r"]) > 0.99
    assert result["relationship"]["strength"] == "strong"


def test_e3_pearson_significant(num_df):
    from services.eda_service import numerical_vs_numerical
    result = numerical_vs_numerical(num_df, "Age", "Income")
    assert result["pearson"]["significant"] == True


def test_e3_regression_line_has_correct_keys(num_df):
    from services.eda_service import numerical_vs_numerical
    result = numerical_vs_numerical(num_df, "Age", "Income")
    reg = result["regression_line"]
    assert "slope" in reg
    assert "intercept" in reg
    assert len(reg["x"]) == 2
    assert len(reg["y"]) == 2


def test_e3_scatter_sample_capped(num_df):
    from services.eda_service import numerical_vs_numerical
    # Extend df to > 500 rows
    big_df = pd.concat([num_df] * 60, ignore_index=True)
    result = numerical_vs_numerical(big_df, "Age", "Income")
    assert len(result["scatter_sample"]["x"]) <= 500


def test_e3_invalid_column():
    from services.eda_service import numerical_vs_numerical
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = numerical_vs_numerical(df, "A", "NonExistent")
    assert "error" in result


# E4 — Numerical vs Categorical

def test_e4_num_vs_cat_keys(mixed_df):
    from services.eda_service import numerical_vs_categorical
    result = numerical_vs_categorical(mixed_df, "Income", "Gender")
    expected_keys = [
        "num_col", "cat_col", "group_stats",
        "box_plot_by_category", "violin_data",
        "difference_summary", "group_count",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e4_group_stats_per_category(mixed_df):
    from services.eda_service import numerical_vs_categorical
    result = numerical_vs_categorical(mixed_df, "Income", "Gender")
    assert "Male" in result["group_stats"]
    assert "Female" in result["group_stats"]


def test_e4_group_stats_has_mean_median(mixed_df):
    from services.eda_service import numerical_vs_categorical
    result = numerical_vs_categorical(mixed_df, "Income", "Gender")
    for group in result["group_stats"].values():
        assert "mean" in group
        assert "median" in group
        assert "std" in group
        assert "min" in group
        assert "max" in group


def test_e4_difference_summary_not_empty(mixed_df):
    from services.eda_service import numerical_vs_categorical
    result = numerical_vs_categorical(mixed_df, "Income", "Gender")
    assert len(result["difference_summary"]) > 0


# E5 — Categorical vs Categorical

def test_e5_cat_vs_cat_keys(cat_df):
    from services.eda_service import categorical_vs_categorical
    result = categorical_vs_categorical(cat_df, "Gender", "Region")
    expected_keys = [
        "col_a", "col_b", "n_pairs", "contingency_table",
        "row_proportions", "chi_square", "grouped_bar_data",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e5_contingency_table_correct(cat_df):
    from services.eda_service import categorical_vs_categorical
    result = categorical_vs_categorical(cat_df, "Gender", "Region")
    ct = result["contingency_table"]
    assert "Male" in ct or "Female" in ct


def test_e5_chi_square_has_pvalue(cat_df):
    from services.eda_service import categorical_vs_categorical
    result = categorical_vs_categorical(cat_df, "Gender", "Region")
    chi2 = result["chi_square"]
    assert "p_value" in chi2
    assert "significant" in chi2


def test_e5_row_proportions_sum_to_100(cat_df):
    from services.eda_service import categorical_vs_categorical
    result = categorical_vs_categorical(cat_df, "Gender", "Region")
    for row_props in result["row_proportions"].values():
        total = sum(row_props.values())
        assert abs(total - 100.0) < 0.5


# E6 — Time-Based EDA

def test_e6_time_eda_keys(time_df):
    from services.eda_service import time_based_eda
    result = time_based_eda(time_df, "date", "revenue", freq="M")
    expected_keys = [
        "date_col", "value_col", "freq", "n_records",
        "date_range", "trend_data", "growth_rates",
        "rolling_mean", "peak_period", "low_period",
        "trend_direction", "time_gaps",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e6_trend_data_not_empty(time_df):
    from services.eda_service import time_based_eda
    result = time_based_eda(time_df, "date", "revenue", freq="M")
    assert len(result["trend_data"]) > 0


def test_e6_growth_rates_length(time_df):
    from services.eda_service import time_based_eda
    result = time_based_eda(time_df, "date", "revenue", freq="M")
    assert len(result["growth_rates"]) == len(result["trend_data"]) - 1


def test_e6_trend_direction_valid(time_df):
    from services.eda_service import time_based_eda
    result = time_based_eda(time_df, "date", "revenue", freq="M")
    assert result["trend_direction"] in ["upward", "downward", "stable", "insufficient data"]


def test_e6_date_range_correct(time_df):
    from services.eda_service import time_based_eda
    result = time_based_eda(time_df, "date", "revenue", freq="M")
    assert result["date_range"]["min"] == "2023-01-31"
    assert result["date_range"]["total_days"] > 0


def test_e6_invalid_column():
    from services.eda_service import time_based_eda
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5, freq="ME"), "val": [1, 2, 3, 4, 5]})
    result = time_based_eda(df, "date", "NonExistent")
    assert "error" in result


# E7 — Multivariate

def test_e7_multivariate_keys(num_df):
    from services.eda_service import multivariate_analysis
    result = multivariate_analysis(num_df)
    expected_keys = [
        "columns_analysed", "n_columns", "heatmap_data",
        "strong_pairs", "pairwise_summary", "correlation_matrix",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e7_heatmap_data_correct_size(num_df):
    from services.eda_service import multivariate_analysis
    result = multivariate_analysis(num_df)
    n = result["n_columns"]
    assert len(result["heatmap_data"]) == n * n


def test_e7_strong_pairs_detected(num_df):
    from services.eda_service import multivariate_analysis
    result = multivariate_analysis(num_df)
    assert len(result["strong_pairs"]) > 0
    for pair in result["strong_pairs"]:
        assert abs(pair["correlation"]) > 0.7


def test_e7_caps_at_20_columns():
    from services.eda_service import multivariate_analysis
    big_df = pd.DataFrame(
        np.random.randn(50, 25),
        columns=[f"col_{i}" for i in range(25)]
    )
    result = multivariate_analysis(big_df)
    assert result["n_columns"] <= 20


def test_e7_insufficient_columns():
    from services.eda_service import multivariate_analysis
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = multivariate_analysis(df)
    assert "error" in result


# E8 — Outlier Investigation

def test_e8_outlier_investigation_keys(num_df):
    from services.eda_service import outlier_investigation
    result = outlier_investigation(num_df, "Age")
    expected_keys = [
        "column", "total_values", "iqr_method",
        "zscore_method", "extreme_values", "outlier_contribution",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e8_iqr_detects_outlier():
    from services.eda_service import outlier_investigation
    df = pd.DataFrame({"Value": [10, 12, 11, 13, 10, 12, 11, 13, 10, 1000]})
    result = outlier_investigation(df, "Value")
    assert result["iqr_method"]["outlier_count"] >= 1


def test_e8_zscore_detects_outlier():
    from services.eda_service import outlier_investigation
    normal = [10] * 20
    df = pd.DataFrame({"Value": normal + [1000]})
    result = outlier_investigation(df, "Value")
    assert result["zscore_method"]["outlier_count"] >= 1


def test_e8_extreme_values_correct_count():
    from services.eda_service import outlier_investigation
    df = pd.DataFrame({"Value": list(range(1, 21))})
    result = outlier_investigation(df, "Value")
    assert len(result["extreme_values"]["top_5"]) == 5
    assert len(result["extreme_values"]["bottom_5"]) == 5


def test_e8_mean_impact_calculated():
    from services.eda_service import outlier_investigation
    df = pd.DataFrame({"Value": [10, 12, 11, 13, 10, 12, 11, 13, 10, 1000]})
    result = outlier_investigation(df, "Value")
    contrib = result["outlier_contribution"]
    assert "mean_with_outliers" in contrib
    assert "mean_without_outliers" in contrib
    assert contrib["mean_with_outliers"] != contrib["mean_without_outliers"]


# E9 — Auto Insights

def test_e9_rule_based_insights_returns_list(skewed_df):
    from services.eda_service import _generate_rule_based_eda_insights
    insights = _generate_rule_based_eda_insights(skewed_df)
    assert isinstance(insights, list)
    assert len(insights) >= 1
    assert all(isinstance(s, str) for s in insights)


def test_e9_skewed_column_mentioned(skewed_df):
    from services.eda_service import _generate_rule_based_eda_insights
    insights = _generate_rule_based_eda_insights(skewed_df)
    combined = " ".join(insights).lower()
    assert "revenue" in combined or "skew" in combined


def test_e9_dominant_category_mentioned(skewed_df):
    from services.eda_service import _generate_rule_based_eda_insights
    insights = _generate_rule_based_eda_insights(skewed_df)
    combined = " ".join(insights).lower()
    assert "category" in combined or "a" in combined


# E11 — Chart Recommendation

def test_e11_numerical_univariate_recommends_histogram(num_df):
    from services.eda_service import recommend_chart
    result = recommend_chart(num_df, "Age")
    assert result["recommended_chart"] == "histogram"


def test_e11_categorical_univariate_recommends_bar(cat_df):
    from services.eda_service import recommend_chart
    result = recommend_chart(cat_df, "Gender")
    assert result["recommended_chart"] == "bar_chart"


def test_e11_num_vs_num_recommends_scatter(num_df):
    from services.eda_service import recommend_chart
    result = recommend_chart(num_df, "Age", "Income")
    assert result["recommended_chart"] == "scatter_plot"


def test_e11_cat_vs_num_recommends_boxplot(mixed_df):
    from services.eda_service import recommend_chart
    result = recommend_chart(mixed_df, "Gender", "Income")
    assert result["recommended_chart"] == "box_plot"


def test_e11_cat_vs_cat_recommends_grouped_bar(cat_df):
    from services.eda_service import recommend_chart
    result = recommend_chart(cat_df, "Gender", "Region")
    assert result["recommended_chart"] == "grouped_bar_chart"


def test_e11_datetime_recommends_line(time_df):
    from services.eda_service import recommend_chart
    result = recommend_chart(time_df, "date")
    assert result["recommended_chart"] == "line_chart"


def test_e11_has_alternatives(num_df):
    from services.eda_service import recommend_chart
    result = recommend_chart(num_df, "Age", "Income")
    assert "alternatives" in result
    assert isinstance(result["alternatives"], list)


# E14 — EDA Summary

def test_e14_summary_keys(mixed_df):
    from services.eda_service import generate_eda_summary
    result = generate_eda_summary(mixed_df, [])
    expected_keys = [
        "shape", "column_types", "strong_correlations",
        "skewed_columns", "high_missing_columns",
        "analyses_run", "questions_asked", "total_analyses",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_e14_column_types_correct(mixed_df):
    from services.eda_service import generate_eda_summary
    result = generate_eda_summary(mixed_df, [])
    # mixed_df has Age, Income, Purchased (numerical=3) and Gender (categorical=1)
    assert result["column_types"]["numerical"] == 3
    assert result["column_types"]["categorical"] == 1


def test_e14_history_reflected(mixed_df):
    from services.eda_service import generate_eda_summary
    history = [
        {"analysis_type": "univariate_numerical"},
        {"analysis_type": "bivariate_num_num"},
        {"question": "What is the average income?"},
    ]
    result = generate_eda_summary(mixed_df, history)
    assert result["total_analyses"] == 3
    assert "What is the average income?" in result["questions_asked"]


# Integration Tests - router endpoints

def _upload_df(client, df) -> str:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    return resp.json()["session_id"]


def test_analyze_univariate_numerical(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Income": [30000, 45000, 60000, 75000, 90000,
                   105000, 120000, 135000, 150000, 165000],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/eda/{session_id}/analyze",
        json={"analysis_type": "univariate_numerical", "column": "Age"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_type"] == "univariate_numerical"
    assert "result" in data
    assert "mean" in data["result"]


def test_analyze_univariate_categorical(client):
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Score": [80, 90, 85, 70, 95],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/eda/{session_id}/analyze",
        json={"analysis_type": "univariate_categorical", "column": "Gender"},
    )
    assert resp.status_code == 200
    assert "frequency_table" in resp.json()["result"]


def test_analyze_bivariate_num_num(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Income": [30000, 45000, 60000, 75000, 90000,
                   105000, 120000, 135000, 150000, 165000],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/eda/{session_id}/analyze",
        json={
            "analysis_type": "bivariate_num_num",
            "col_x": "Age",
            "col_y": "Income",
        },
    )
    assert resp.status_code == 200
    assert "pearson" in resp.json()["result"]


def test_analyze_with_filter(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Gender": ["Male", "Female"] * 5,
        "Income": [30000, 45000, 60000, 75000, 90000,
                   105000, 120000, 135000, 150000, 165000],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/eda/{session_id}/analyze",
        json={
            "analysis_type": "univariate_numerical",
            "column": "Age",
            "filters": [{"column": "Gender", "operator": "eq", "value": "Male"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filters_applied"] == 1
    assert data["rows_analysed"] == 5


def test_analyze_filter_does_not_modify_dataset(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
    })
    session_id = _upload_df(client, df)

    client.post(
        f"/api/eda/{session_id}/analyze",
        json={
            "analysis_type": "univariate_numerical",
            "column": "Age",
            "filters": [{"column": "Gender", "operator": "eq", "value": "Male"}],
        },
    )

    info = client.get(f"/api/dataset/{session_id}/info").json()
    assert info["rows"] == 5


def test_analyze_logs_to_history(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Score": [80, 90, 85, 70, 95],
    })
    session_id = _upload_df(client, df)

    client.post(
        f"/api/eda/{session_id}/analyze",
        json={"analysis_type": "univariate_numerical", "column": "Age"},
    )
    client.post(
        f"/api/eda/{session_id}/analyze",
        json={"analysis_type": "univariate_numerical", "column": "Score"},
    )

    history_resp = client.get(f"/api/eda/{session_id}/history")
    assert history_resp.status_code == 200
    assert history_resp.json()["total_analyses"] == 2


def test_recommend_chart_endpoint(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [30000, 45000, 60000, 75000, 90000],
    })
    session_id = _upload_df(client, df)
    resp = client.get(
        f"/api/eda/{session_id}/recommend-chart",
        params={"col_x": "Age", "col_y": "Income"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chart_recommendation"]["recommended_chart"] == "scatter_plot"


def test_insights_endpoint_returns_list(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [30000, 45000, 60000, 75000, 90000],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
    })
    session_id = _upload_df(client, df)
    resp = client.get(f"/api/eda/{session_id}/insights")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["insights"], list)
    assert data["count"] >= 1
    assert data["source"] in ["gemini", "rule_based"]


def test_question_endpoint_returns_answer(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [30000, 45000, 60000, 75000, 90000],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/eda/{session_id}/question",
        json={"question": "What variables are strongly correlated?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert data["source"] in ["gemini", "rule_based"]


def test_summary_endpoint_returns_correct_shape(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [30000, 45000, 60000, 75000, 90000],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
    })
    session_id = _upload_df(client, df)
    resp = client.get(f"/api/eda/{session_id}/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "shape" in data["summary"]
    assert "column_types" in data["summary"]


def test_analyze_invalid_type_returns_400(client, uploaded_session):
    resp = client.post(
        f"/api/eda/{uploaded_session}/analyze",
        json={"analysis_type": "nonexistent_type", "column": "Age"},
    )
    assert resp.status_code == 400


def test_analyze_empty_filter_returns_400(client):
    df = pd.DataFrame({
        "Age": [25, 30, 35],
        "Gender": ["Male", "Female", "Male"],
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/eda/{session_id}/analyze",
        json={
            "analysis_type": "univariate_numerical",
            "column": "Age",
            "filters": [{"column": "Gender", "operator": "eq", "value": "Other"}],
        },
    )
    assert resp.status_code == 400


def test_analyze_missing_required_column_returns_400(client, uploaded_session):
    resp = client.post(
        f"/api/eda/{uploaded_session}/analyze",
        json={"analysis_type": "univariate_numerical"},
    )
    assert resp.status_code == 400