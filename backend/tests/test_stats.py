"""
Stage 6 Tests — Statistical Analysis Service.
Covers all 21 statistical tests + recommender + assumption checks + router endpoints.
"""

import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


# Fixtures

@pytest.fixture
def normal_df():
    """Normally distributed numerical data."""
    np.random.seed(42)
    return pd.DataFrame({
        "Score":  np.random.normal(75, 10, 50).tolist(),
        "Age":    np.random.normal(35, 5, 50).tolist(),
        "Income": np.random.normal(60000, 10000, 50).tolist(),
    })


@pytest.fixture
def group_df():
    """Numerical + categorical for group comparison tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "Score":  ([70, 75, 80, 72, 78, 74, 76, 79, 71, 77] +
                   [85, 88, 90, 86, 91, 87, 89, 92, 84, 93]),
        "Group":  (["Control"] * 10 + ["Treatment"] * 10),
        "Region": (["North"] * 5 + ["South"] * 5 +
                   ["North"] * 5 + ["South"] * 5),
    })


@pytest.fixture
def cat_df():
    """Categorical data for chi-square tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "Gender":    (["Male"] * 30 + ["Female"] * 30),
        "Purchased": (["Yes"] * 20 + ["No"] * 10 +
                      ["Yes"] * 15 + ["No"] * 15),
        "Region":    (["North", "South"] * 30),
    })


@pytest.fixture
def paired_df():
    """Paired before/after data."""
    np.random.seed(42)
    before = np.random.normal(70, 10, 30).tolist()
    after = [b + np.random.normal(5, 2) for b in before]
    return pd.DataFrame({"Before": before, "After": after})


@pytest.fixture
def multi_group_df():
    """Three groups for ANOVA and Kruskal-Wallis."""
    np.random.seed(42)
    return pd.DataFrame({
        "Score": (
            np.random.normal(70, 5, 20).tolist() +
            np.random.normal(80, 5, 20).tolist() +
            np.random.normal(90, 5, 20).tolist()
        ),
        "Group": (["A"] * 20 + ["B"] * 20 + ["C"] * 20),
    })


# Assumption Checkers

def test_check_normality_normal_data(normal_df):
    from services.stats_service import check_normality
    result = check_normality(normal_df["Score"])
    assert "passed" in result
    assert "p_value" in result
    assert "statistic" in result
    assert result["passed"] == True


def test_check_normality_non_normal():
    from services.stats_service import check_normality
    s = pd.Series([1] * 40 + [100] * 10)
    result = check_normality(s)
    assert result["passed"] == False


def test_check_equal_variance_equal(group_df):
    from services.stats_service import check_equal_variance
    g1 = group_df[group_df["Group"] == "Control"]["Score"]
    g2 = group_df[group_df["Group"] == "Treatment"]["Score"]
    result = check_equal_variance(g1, g2)
    assert "passed" in result
    assert "p_value" in result


def test_check_sample_size_passes():
    from services.stats_service import check_sample_size
    s = pd.Series(range(50))
    result = check_sample_size(s, min_n=30)
    assert result["passed"] == True
    assert result["n"] == 50


def test_check_sample_size_fails():
    from services.stats_service import check_sample_size
    s = pd.Series(range(10))
    result = check_sample_size(s, min_n=30)
    assert result["passed"] == False


# Test Recommender

def test_recommender_two_numerical_normal(normal_df):
    from services.stats_service import recommend_test
    result = recommend_test(
        normal_df,
        "Is Score related to Age?",
        ["Score", "Age"],
    )
    assert result["recommended_test"] in ["pearson", "spearman"]
    assert "rationale" in result
    assert "available_tests" in result


def test_recommender_num_vs_cat_two_groups(group_df):
    from services.stats_service import recommend_test
    result = recommend_test(
        group_df,
        "Does Score differ between groups?",
        ["Score", "Group"],
    )
    assert result["recommended_test"] in ["independent_t", "mann_whitney"]


def test_recommender_num_vs_cat_multi_groups(multi_group_df):
    from services.stats_service import recommend_test
    result = recommend_test(
        multi_group_df,
        "Does Score differ across groups?",
        ["Score", "Group"],
    )
    assert result["recommended_test"] in ["one_way_anova", "kruskal_wallis"]


def test_recommender_two_categorical(cat_df):
    from services.stats_service import recommend_test
    result = recommend_test(
        cat_df,
        "Is Gender associated with purchase?",
        ["Gender", "Purchased"],
    )
    assert result["recommended_test"] in [
        "chi_square_independence", "fisher_exact"
    ]


def test_recommender_normality_question(normal_df):
    from services.stats_service import recommend_test
    result = recommend_test(
        normal_df,
        "Is Score normally distributed?",
        ["Score"],
    )
    assert result["recommended_test"] == "shapiro_wilk"


def test_recommender_unknown_column_returns_error(normal_df):
    from services.stats_service import recommend_test
    result = recommend_test(normal_df, "test", ["NonExistent"])
    assert "error" in result


# S1   One-Sample t-test

@pytest.mark.asyncio
async def test_s1_one_sample_t_keys(normal_df):
    from services.stats_service import one_sample_t
    result = await one_sample_t(normal_df, "Score", pop_mean=70.0)
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "interpretation", "plain_english",
                "assumption_checks"]:
        assert key in result


@pytest.mark.asyncio
async def test_s1_one_sample_t_significant(normal_df):
    from services.stats_service import one_sample_t
    # mean ~75, testing against 0 — should be significant
    result = await one_sample_t(normal_df, "Score", pop_mean=0.0)
    assert result["significant"] == True


@pytest.mark.asyncio
async def test_s1_one_sample_t_not_significant(normal_df):
    from services.stats_service import one_sample_t
    # mean ~75, testing against 75 — should not be significant
    result = await one_sample_t(normal_df, "Score", pop_mean=75.0)
    assert result["significant"] == False


# S2   Independent Samples t-test

@pytest.mark.asyncio
async def test_s2_independent_t_keys(group_df):
    from services.stats_service import independent_t
    result = await independent_t(group_df, "Score", "Group")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "groups", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s2_independent_t_detects_difference(group_df):
    from services.stats_service import independent_t
    result = await independent_t(group_df, "Score", "Group")
    assert result["significant"] == True
    assert result["p_value"] < 0.05


@pytest.mark.asyncio
async def test_s2_independent_t_effect_size_label(group_df):
    from services.stats_service import independent_t
    result = await independent_t(group_df, "Score", "Group")
    assert result["effect_size"]["label"] in [
        "negligible", "small", "medium", "large"
    ]


# S3   Paired Samples t-test

@pytest.mark.asyncio
async def test_s3_paired_t_keys(paired_df):
    from services.stats_service import paired_t
    result = await paired_t(paired_df, "Before", "After")
    for key in ["test", "statistic", "p_value", "significant",
                "mean_before", "mean_after", "mean_difference",
                "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s3_paired_t_detects_change(paired_df):
    from services.stats_service import paired_t
    result = await paired_t(paired_df, "Before", "After")
    assert result["significant"] == True
    assert result["mean_difference"] > 0


# S4   One-Way ANOVA

@pytest.mark.asyncio
async def test_s4_one_way_anova_keys(multi_group_df):
    from services.stats_service import one_way_anova
    result = await one_way_anova(multi_group_df, "Score", "Group")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "group_stats", "posthoc_tukey",
                "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s4_one_way_anova_significant(multi_group_df):
    from services.stats_service import one_way_anova
    result = await one_way_anova(multi_group_df, "Score", "Group")
    assert result["significant"] == True


@pytest.mark.asyncio
async def test_s4_posthoc_tukey_present(multi_group_df):
    from services.stats_service import one_way_anova
    result = await one_way_anova(multi_group_df, "Score", "Group")
    assert isinstance(result["posthoc_tukey"], list)


@pytest.mark.asyncio
async def test_s4_group_stats_all_groups(multi_group_df):
    from services.stats_service import one_way_anova
    result = await one_way_anova(multi_group_df, "Score", "Group")
    assert "A" in result["group_stats"]
    assert "B" in result["group_stats"]
    assert "C" in result["group_stats"]


# S7 — Pearson Correlation

@pytest.mark.asyncio
async def test_s7_pearson_keys(normal_df):
    from services.stats_service import pearson_correlation
    result = await pearson_correlation(normal_df, "Score", "Age")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "plain_english", "assumption_checks"]:
        assert key in result


@pytest.mark.asyncio
async def test_s7_pearson_perfect_correlation():
    from services.stats_service import pearson_correlation
    df = pd.DataFrame({
        "X": list(range(1, 51)),
        "Y": list(range(1, 51)),
    })
    result = await pearson_correlation(df, "X", "Y")
    assert abs(result["statistic"]) > 0.99
    assert result["significant"] == True


@pytest.mark.asyncio
async def test_s7_pearson_r_squared_correct():
    from services.stats_service import pearson_correlation
    df = pd.DataFrame({
        "X": list(range(1, 51)),
        "Y": list(range(1, 51)),
    })
    result = await pearson_correlation(df, "X", "Y")
    r = result["statistic"]
    r_sq = result["effect_size"]["r_squared"]
    assert abs(r_sq - r**2) < 0.001


# S8   Linear Regression

@pytest.mark.asyncio
async def test_s8_linear_regression_keys(normal_df):
    from services.stats_service import linear_regression
    result = await linear_regression(normal_df, "Age", "Score")
    for key in ["test", "statistic", "p_value", "significant",
                "coefficients", "effect_size", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s8_linear_regression_coefficients():
    from services.stats_service import linear_regression
    df = pd.DataFrame({
        "X": list(range(1, 51)),
        "Y": [2 * x + 1 for x in range(1, 51)],
    })
    result = await linear_regression(df, "X", "Y")
    assert abs(result["coefficients"]["slope"] - 2.0) < 0.01
    assert result["significant"] == True


# S10   Mann-Whitney U Test

@pytest.mark.asyncio
async def test_s10_mann_whitney_keys(group_df):
    from services.stats_service import mann_whitney
    result = await mann_whitney(group_df, "Score", "Group")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "groups", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s10_mann_whitney_detects_difference(group_df):
    from services.stats_service import mann_whitney
    result = await mann_whitney(group_df, "Score", "Group")
    assert result["significant"] == True


# S11   Wilcoxon Signed-Rank Test

@pytest.mark.asyncio
async def test_s11_wilcoxon_keys(paired_df):
    from services.stats_service import wilcoxon_test
    result = await wilcoxon_test(paired_df, "Before", "After")
    for key in ["test", "statistic", "p_value", "significant", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s11_wilcoxon_detects_change(paired_df):
    from services.stats_service import wilcoxon_test
    result = await wilcoxon_test(paired_df, "Before", "After")
    assert result["significant"] == True


# S12   Kruskal-Wallis Test

@pytest.mark.asyncio
async def test_s12_kruskal_wallis_keys(multi_group_df):
    from services.stats_service import kruskal_wallis
    result = await kruskal_wallis(multi_group_df, "Score", "Group")
    for key in ["test", "statistic", "p_value", "significant",
                "group_stats", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s12_kruskal_wallis_significant(multi_group_df):
    from services.stats_service import kruskal_wallis
    result = await kruskal_wallis(multi_group_df, "Score", "Group")
    assert result["significant"] == True


# S13   Spearman Correlation

@pytest.mark.asyncio
async def test_s13_spearman_keys(normal_df):
    from services.stats_service import spearman_correlation
    result = await spearman_correlation(normal_df, "Score", "Age")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s13_spearman_monotonic():
    from services.stats_service import spearman_correlation
    df = pd.DataFrame({
        "X": list(range(1, 51)),
        "Y": [x**2 for x in range(1, 51)],
    })
    result = await spearman_correlation(df, "X", "Y")
    assert abs(result["statistic"]) > 0.99
    assert result["significant"] == True


# S14   Kendall's Tau

@pytest.mark.asyncio
async def test_s14_kendall_keys(normal_df):
    from services.stats_service import kendall_tau
    result = await kendall_tau(normal_df, "Score", "Age")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "plain_english"]:
        assert key in result


# S15  Chi-Square Independence

@pytest.mark.asyncio
async def test_s15_chi_square_independence_keys(cat_df):
    from services.stats_service import chi_square_independence
    result = await chi_square_independence(cat_df, "Gender", "Purchased")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "dof", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s15_cramers_v_between_0_and_1(cat_df):
    from services.stats_service import chi_square_independence
    result = await chi_square_independence(cat_df, "Gender", "Purchased")
    v = result["effect_size"]["cramers_v"]
    assert 0.0 <= v <= 1.0


# S17   Fisher's Exact Test

@pytest.mark.asyncio
async def test_s17_fisher_exact_keys():
    from services.stats_service import fisher_exact
    df = pd.DataFrame({
        "Treatment": (["Drug"] * 20 + ["Placebo"] * 20),
        "Outcome":   (["Recovered"] * 15 + ["Not Recovered"] * 5 +
                      ["Recovered"] * 8 + ["Not Recovered"] * 12),
    })
    result = await fisher_exact(df, "Treatment", "Outcome")
    for key in ["test", "statistic", "p_value", "significant",
                "effect_size", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s17_fisher_exact_non_2x2_returns_error():
    from services.stats_service import fisher_exact
    df = pd.DataFrame({
        "A": ["X", "Y", "Z"] * 10,
        "B": ["P", "Q"] * 15,
    })
    result = await fisher_exact(df, "A", "B")
    assert "error" in result


# S18   Shapiro-Wilk

@pytest.mark.asyncio
async def test_s18_shapiro_wilk_normal(normal_df):
    from services.stats_service import shapiro_wilk
    result = await shapiro_wilk(normal_df, "Score")
    assert result["is_normal"] == True
    assert "plain_english" in result


@pytest.mark.asyncio
async def test_s18_shapiro_wilk_non_normal():
    from services.stats_service import shapiro_wilk
    df = pd.DataFrame({"Value": [1] * 40 + [1000] * 10})
    result = await shapiro_wilk(df, "Value")
    assert result["is_normal"] == False


# S19   KS Test

@pytest.mark.asyncio
async def test_s19_ks_test_keys(normal_df):
    from services.stats_service import ks_test
    result = await ks_test(normal_df, "Score")
    for key in ["test", "statistic", "p_value", "is_normal", "plain_english"]:
        assert key in result


# S20   Levene's Test

@pytest.mark.asyncio
async def test_s20_levene_keys(group_df):
    from services.stats_service import levene_test
    result = await levene_test(group_df, "Score", "Group")
    for key in ["test", "statistic", "p_value", "equal_variances", "plain_english"]:
        assert key in result


@pytest.mark.asyncio
async def test_s20_levene_equal_variances(group_df):
    from services.stats_service import levene_test
    result = await levene_test(group_df, "Score", "Group")
    assert isinstance(result["equal_variances"], bool)


# S21 — Bartlett's Test

@pytest.mark.asyncio
async def test_s21_bartlett_keys(group_df):
    from services.stats_service import bartlett_test
    result = await bartlett_test(group_df, "Score", "Group")
    for key in ["test", "statistic", "p_value", "equal_variances", "plain_english"]:
        assert key in result


# Run dispatcher

@pytest.mark.asyncio
async def test_dispatcher_pearson(normal_df):
    from services.stats_service import run_test
    result = await run_test(normal_df, {
        "test_name": "pearson",
        "col_x": "Score",
        "col_y": "Age",
    })
    assert result["test"] == "pearson"
    assert "p_value" in result


@pytest.mark.asyncio
async def test_dispatcher_unknown_test():
    from services.stats_service import run_test
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = await run_test(df, {"test_name": "unknown_test"})
    assert "error" in result


@pytest.mark.asyncio
async def test_dispatcher_shapiro(normal_df):
    from services.stats_service import run_test
    result = await run_test(normal_df, {
        "test_name": "shapiro_wilk",
        "column": "Score",
    })
    assert result["test"] == "shapiro_wilk"


# Integration Tests — router endpoints

def _upload_df(client, df) -> str:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    return resp.json()["session_id"]


def test_recommend_endpoint(client):
    df = pd.DataFrame({
        "Score": list(range(1, 51)),
        "Group": (["A"] * 25 + ["B"] * 25),
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/stats/{session_id}/recommend",
        json={
            "question": "Does Score differ between groups?",
            "columns": ["Score", "Group"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendation" in data
    assert "recommended_test" in data["recommendation"]
    assert "rationale" in data["recommendation"]


def test_run_test_endpoint_shapiro(client):
    np.random.seed(42)
    df = pd.DataFrame({"Score": np.random.normal(75, 10, 50).tolist()})
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/stats/{session_id}/run",
        json={
            "test_name": "shapiro_wilk",
            "options": {"column": "Score"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert "p_value" in data["result"]
    assert "plain_english" in data["result"]


def test_run_test_endpoint_independent_t(client):
    df = pd.DataFrame({
        "Score": ([70, 75, 80, 72, 78, 74, 76, 79, 71, 77] +
                  [85, 88, 90, 86, 91, 87, 89, 92, 84, 93]),
        "Group": (["Control"] * 10 + ["Treatment"] * 10),
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/stats/{session_id}/run",
        json={
            "test_name": "independent_t",
            "options": {"num_col": "Score", "cat_col": "Group"},
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["significant"] == True


def test_run_test_logs_to_history(client):
    np.random.seed(42)
    df = pd.DataFrame({"Score": np.random.normal(75, 10, 50).tolist()})
    session_id = _upload_df(client, df)

    client.post(
        f"/api/stats/{session_id}/run",
        json={
            "test_name": "shapiro_wilk",
            "options": {"column": "Score"},
        },
    )

    history_resp = client.get(f"/api/stats/{session_id}/history")
    assert history_resp.status_code == 200
    history = history_resp.json()["history"]
    assert len(history) == 1
    assert history[0]["test_name"] == "shapiro_wilk"


def test_run_test_invalid_test_name_returns_400(client, uploaded_session):
    resp = client.post(
        f"/api/stats/{uploaded_session}/run",
        json={"test_name": "invalid_test", "options": {}},
    )
    assert resp.status_code == 400


def test_recommend_missing_columns_returns_400(client, uploaded_session):
    resp = client.post(
        f"/api/stats/{uploaded_session}/recommend",
        json={"question": "test question", "columns": []},
    )
    assert resp.status_code == 400


def test_recommend_nonexistent_column_returns_400(client, uploaded_session):
    resp = client.post(
        f"/api/stats/{uploaded_session}/recommend",
        json={
            "question": "Does X differ?",
            "columns": ["NonExistentColumn"],
        },
    )
    assert resp.status_code == 400


def test_plain_english_always_present(client):
    np.random.seed(42)
    df = pd.DataFrame({
        "Score": np.random.normal(75, 10, 50).tolist(),
        "Group": (["A"] * 25 + ["B"] * 25),
    })
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/stats/{session_id}/run",
        json={
            "test_name": "independent_t",
            "options": {"num_col": "Score", "cat_col": "Group"},
        },
    )
    result = resp.json()["result"]
    assert "plain_english" in result
    assert len(result["plain_english"]) > 10


def test_assumption_checks_always_present(client):
    np.random.seed(42)
    df = pd.DataFrame({"Score": np.random.normal(75, 10, 50).tolist()})
    session_id = _upload_df(client, df)
    resp = client.post(
        f"/api/stats/{session_id}/run",
        json={
            "test_name": "shapiro_wilk",
            "options": {"column": "Score"},
        },
    )
    result = resp.json()["result"]
    assert "assumption_checks" in result
    assert isinstance(result["assumption_checks"], dict)





