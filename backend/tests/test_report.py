import io
import json

import numpy as np
import pandas as pd
import pytest

from core.cache import (
    append_eda_history,
    append_nlp_history,
    append_stats_history,
    get_profile,
    set_ml_result,
)
from services import report_service, session_service
from services.report_service import PDFEngineUnavailable


# Helpers & Fixtures

def _df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Age": rng.integers(18, 70, n).astype(float),
        "Income": rng.normal(50000, 10000, n).round(2),
        "Gender": rng.choice(["Male", "Female"], n),
        "Churn": [0, 1] * (n // 2),
    })
    df.loc[[2, 5, 9], "Age"] = np.nan
    df.loc[[1, 4], "Income"] = np.nan
    return df


def _reviews_df(n: int = 40) -> pd.DataFrame:
    pool = [
        "The product quality is excellent and I love the design",
        "Terrible quality and the product broke after one week",
        "The package arrived on Tuesday in a brown box",
        "Great value for the price and the delivery was fast",
        "Very disappointed with the slow delivery and poor service",
        "The store opens at nine and closes at five",
    ]
    return pd.DataFrame({"Review": [f"{pool[i % len(pool)]} number {i}" for i in range(n)]})


def _new_session(df: pd.DataFrame | None = None) -> str:
    df = _df() if df is None else df
    return session_service.create_session(df, "test.csv")["session_id"]


def _log_clean(sid, issue="missing_numerical", column="Age", strategy="median",
               rb=30, ra=30, cb=4, ca=4):
    session_service.log_operation(sid, {
        "stage": "clean", "issue_type": issue, "column": column, "strategy": strategy,
        "rows_before": rb, "rows_after": ra, "cols_before": cb, "cols_after": ca,
    })


def _log_transform(sid, ttype, column, new_columns=None, options=None, cb=4, ca=4):
    session_service.log_operation(sid, {
        "stage": "transform", "transform_type": ttype, "column": column,
        "options": options or {}, "new_columns": new_columns or [],
        "rows_before": 30, "rows_after": 30, "cols_before": cb, "cols_after": ca,
    })


def _strategy_stub(valid: bool = True) -> dict:
    if not valid:
        return {"valid": False}
    return {
        "valid": True,
        "objective": "predict churn",
        "summary": "Start with a gradient boosting model.",
        "reasoning_source": "rule_based",
        "problem": {"problem_type": "binary_classification", "target": "Churn"},
        "readiness": {"score": 85, "label": "ready", "warnings": ["Small dataset."]},
        "imbalance": {"is_imbalanced": True, "severity": "moderate"},
        "models": [
            {"name": f"Model{i}", "score": 90 - i, "suitability": "excellent", "reasoning": f"reason {i}"}
            for i in range(5)
        ],
        "metrics": {"primary": "PR-AUC", "secondary": ["F1", "Recall"], "reason": "Imbalanced target."},
        "cross_validation": {"method": "Stratified K-Fold"},
        "split_strategy": {"strategy": "stratified"},
        "preprocessing": [{"technique": "Impute missing values"}, {"technique": "Scale numeric features"}],
        "roadmap": [
            {"step_number": 1, "title": "Confirm the problem framing", "description": "Check the target."},
            {"step_number": 2, "title": "Train a baseline", "description": "Start simple."},
        ],
    }


def _upload_df(client, df) -> str:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    return resp.json()["session_id"]


@pytest.fixture
def pdf_engine():
    if not report_service.pdf_available():
        pytest.skip("PDF engine (WeasyPrint + system libraries) is not available")
    return True


# Helpers: _plural, _normalise_sections, _sanitize

def test_plural_regular_and_irregular():
    assert report_service._plural(1, "operation") == "1 operation"
    assert report_service._plural(2, "operation") == "2 operations"
    assert report_service._plural(1, "analysis") == "1 analysis"
    assert report_service._plural(2, "analysis") == "2 analyses"
    assert report_service._plural(3, "exploratory analysis") == "3 exploratory analyses"


def test_normalise_sections_defaults_to_all():
    assert report_service._normalise_sections(None) == report_service.SECTION_KEYS
    assert report_service._normalise_sections([]) == report_service.SECTION_KEYS
    assert len(report_service.SECTION_KEYS) == 10


def test_normalise_sections_canonical_order():
    assert report_service._normalise_sections(["history", "cleaning", "executive_summary"]) == \
        ["executive_summary", "cleaning", "history"]


def test_normalise_sections_rejects_unknown():
    with pytest.raises(ValueError):
        report_service._normalise_sections(["cleaning", "nonsense"])


def test_sanitize_is_json_safe():
    messy = {"a": np.float64("nan"), "b": np.int64(3), "c": np.bool_(True), "d": [np.float32(1.5)]}
    json.dumps(report_service._sanitize(messy), allow_nan=False)


# Section availability

def test_availability_fresh_session():
    sid = _new_session()
    by_key = {s["key"]: s for s in report_service.get_section_availability(sid)}
    assert len(by_key) == 10
    for key in ["executive_summary", "profiling", "appendix"]:
        assert by_key[key]["completed"] is True
    for key in ["cleaning", "transformation", "eda", "statistics", "nlp", "ml_strategy", "history"]:
        assert by_key[key]["completed"] is False


def test_availability_after_activity():
    sid = _new_session()
    _log_clean(sid)
    _log_transform(sid, "minmax", "Age")
    append_eda_history(sid, {"analysis_type": "univariate_numerical", "column": "Age"})
    append_eda_history(sid, {"analysis_type": "univariate_numerical", "column": "Income"})
    append_stats_history(sid, {"test_name": "pearson", "significant": True})
    append_nlp_history(sid, {"analysis_type": "sentiment", "column": "Review"})
    set_ml_result(sid, _strategy_stub())

    by_key = {s["key"]: s for s in report_service.get_section_availability(sid)}
    assert all(by_key[k]["completed"] for k in
               ["cleaning", "transformation", "eda", "statistics", "nlp", "ml_strategy", "history"])
    assert by_key["cleaning"]["detail"] == "1 operation applied."
    assert by_key["eda"]["detail"] == "2 analyses run."
    assert by_key["statistics"]["detail"] == "1 test run."
    assert by_key["nlp"]["detail"] == "1 analysis run."
    assert by_key["history"]["detail"] == "2 operations recorded."


def test_availability_invalid_ml_strategy_not_completed():
    sid = _new_session()
    set_ml_result(sid, _strategy_stub(valid=False))
    by_key = {s["key"]: s for s in report_service.get_section_availability(sid)}
    assert by_key["ml_strategy"]["completed"] is False


# Cleaning / Transformation sections

def test_cleaning_not_performed_when_empty():
    sec = report_service.collect_cleaning(_new_session())
    assert sec["performed"] is False
    assert sec["note"] == "No cleaning operations were applied."
    assert sec["data"] == {}


def test_cleaning_summarises_operations():
    sid = _new_session()
    _log_clean(sid, "missing_numerical", "Age", "median", rb=30, ra=28)
    _log_clean(sid, "missing_numerical", "Income", "mean", rb=28, ra=28)
    _log_clean(sid, "duplicates", None, "drop_duplicates", rb=28, ra=27)

    sec = report_service.collect_cleaning(sid)
    d = sec["data"]
    assert sec["performed"] is True
    assert d["total_operations"] == 3
    assert d["net_rows_change"] == -3
    assert d["net_columns_change"] == 0
    assert d["by_issue"][0] == {"issue_type": "missing_numerical", "count": 2}
    assert d["operations"][0]["step"] == 1
    assert d["operations"][2]["column"] is None
    assert d["truncated"] is False


def test_cleaning_truncates_long_logs():
    sid = _new_session()
    for _ in range(105):
        _log_clean(sid)
    d = report_service.collect_cleaning(sid)["data"]
    assert d["total_operations"] == 105
    assert len(d["operations"]) == 100
    assert d["truncated"] is True


def test_cleaning_ignores_transform_operations():
    sid = _new_session()
    _log_transform(sid, "minmax", "Age")
    assert report_service.collect_cleaning(sid)["performed"] is False


def test_transformation_not_performed_when_empty():
    sec = report_service.collect_transformation(_new_session())
    assert sec["performed"] is False
    assert sec["note"] == "No transformations were applied."


def test_transformation_groups_and_new_features():
    sid = _new_session()
    _log_transform(sid, "minmax", "Age")
    _log_transform(sid, "onehot", "Gender", ["Gender_Male", "Gender_Female"], cb=4, ca=5)
    _log_transform(sid, "extract_month", "created_at", ["created_at_month"])
    _log_transform(sid, "exclude_ml", "CustomerID")
    _log_transform(sid, "pca", "Age", ["pca_1", "pca_2"])
    _log_transform(sid, "equal_width", "Age", ["Age_bin"], options={"bins": 5})

    d = report_service.collect_transformation(sid)["data"]
    assert d["total_operations"] == 6
    assert d["counts"] == {
        "scaled": 1, "encoded": 1, "binned": 1, "datetime_extracted": 1,
        "engineered": 0, "dim_reduction": 1, "excluded_from_ml": 1,
    }
    assert d["new_features"] == [
        "Gender_Male", "Gender_Female", "created_at_month", "pca_1", "pca_2", "Age_bin",
    ]
    assert set(d["columns_transformed"]) == {"Age", "Gender", "created_at", "CustomerID"}
    assert len(d["columns_transformed"]) == 4  # Age is only listed once
    assert d["operations"][5]["options"] == '{"bins": 5}'


# EDA / Statistics sections

def test_eda_not_performed_when_empty():
    sid = _new_session()
    sec = report_service.collect_eda(sid, session_service.get_working_df(sid))
    assert sec["performed"] is False
    assert sec["note"] == "No exploratory analyses were run."


def test_eda_summarises_history():
    sid = _new_session()
    append_eda_history(sid, {"analysis_type": "univariate_numerical", "column": "Age",
                             "filter_count": 0, "rows_analysed": 30})
    append_eda_history(sid, {"analysis_type": "bivariate_num_num", "col_x": "Age", "col_y": "Income",
                             "filter_count": 1, "rows_analysed": 15})
    append_eda_history(sid, {"analysis_type": "question", "question": "What drives churn?"})

    df = session_service.get_working_df(sid)
    d = report_service.collect_eda(sid, df)["data"]
    assert d["total_analyses"] == 3
    assert len(d["analyses"]) == 2                     # the question is listed separately
    assert d["analyses"][1]["columns"] == "Age, Income"
    assert d["analyses"][1]["filters"] == 1
    assert d["questions"] == ["What drives churn?"]
    assert d["column_types"]["numerical"] == 3
    assert isinstance(d["observations"], list)


def test_statistics_not_performed_when_empty():
    sec = report_service.collect_statistics(_new_session())
    assert sec["performed"] is False
    assert sec["note"] == "No statistical tests were run."


def test_statistics_counts_significant_tests():
    sid = _new_session()
    append_stats_history(sid, {"test_name": "independent_t", "options": {"num_col": "Age", "cat_col": "Gender"},
                               "significant": True, "p_value": 0.01, "plain_english": "Groups differ."})
    append_stats_history(sid, {"test_name": "shapiro_wilk", "options": {"column": "Age"},
                               "significant": False, "p_value": 0.4, "plain_english": "Looks normal."})
    append_stats_history(sid, {"test_name": "pearson", "options": {}, "significant": True, "p_value": 0.001})

    d = report_service.collect_statistics(sid)["data"]
    assert d["total_tests"] == 3
    assert d["significant_count"] == 2
    assert d["tests"][0]["plain_english"] == "Groups differ."
    assert isinstance(d["tests"][0]["options"], str)


# NLP section

@pytest.mark.asyncio
async def test_nlp_not_performed_when_empty():
    sid = _new_session()
    sec = await report_service.collect_nlp(sid, session_service.get_working_df(sid))
    assert sec["performed"] is False
    assert sec["note"] == "No NLP analyses were run."


@pytest.mark.asyncio
async def test_nlp_missing_column_is_reported_not_raised():
    sid = _new_session()
    append_nlp_history(sid, {"analysis_type": "quality", "column": "Gone"})
    sec = await report_service.collect_nlp(sid, session_service.get_working_df(sid))
    assert sec["performed"] is True
    assert "no longer exists" in sec["data"]["columns"][0]["error"]


@pytest.mark.asyncio
async def test_nlp_caps_summarised_columns():
    sid = _new_session()
    for col in ["A", "B", "C"]:
        append_nlp_history(sid, {"analysis_type": "quality", "column": col})
    sec = await report_service.collect_nlp(sid, session_service.get_working_df(sid))
    assert len(sec["data"]["columns"]) == 2
    assert sec["data"]["columns_not_summarised"] == 1


@pytest.mark.asyncio
async def test_nlp_summarises_analysed_text_column():
    pytest.importorskip("vaderSentiment")
    df = _reviews_df()
    sid = _new_session(df)
    append_nlp_history(sid, {"analysis_type": "sentiment", "column": "Review"})
    append_nlp_history(sid, {"analysis_type": "keywords", "column": "Review"})

    sec = await report_service.collect_nlp(sid, session_service.get_working_df(sid))
    d = sec["data"]
    assert d["total_analyses"] == 2
    assert {a["analysis_type"]: a["count"] for a in d["analysis_counts"]} == {"sentiment": 1, "keywords": 1}
    col = d["columns"][0]
    assert col["column"] == "Review"
    assert "error" not in col
    assert col["sentiment"]["overall"] in ["positive", "negative", "neutral"]
    assert col["quality"]["total_documents"] == len(df)
    assert col["top_keywords"]


# ML section

def test_ml_not_performed_without_strategy():
    sec = report_service.collect_ml(_new_session())
    assert sec["performed"] is False
    assert sec["note"] == "No ML strategy was generated."


def test_ml_not_performed_for_invalid_strategy():
    sid = _new_session()
    set_ml_result(sid, _strategy_stub(valid=False))
    assert report_service.collect_ml(sid)["performed"] is False


def test_ml_summarises_strategy():
    sid = _new_session()
    set_ml_result(sid, _strategy_stub())
    d = report_service.collect_ml(sid)["data"]
    assert d["problem_type"] == "binary_classification"
    assert d["target"] == "Churn"
    assert d["objective"] == "predict churn"
    assert len(d["top_models"]) == 3                  # capped at the top three
    assert d["top_models"][0]["name"] == "Model0"
    assert d["metrics"]["primary"] == "PR-AUC"
    assert d["validation"] == "Stratified K-Fold"
    assert d["split"] == "stratified"
    assert d["imbalance"]["is_imbalanced"] is True
    assert len(d["roadmap"]) == 2


# History and Appendix sections

def test_history_not_performed_when_empty():
    sec = report_service.collect_history(_new_session())
    assert sec["performed"] is False
    assert sec["note"] == "No cleaning or transformation operations were recorded."


def test_history_lists_every_operation_in_order():
    sid = _new_session()
    _log_clean(sid, "missing_numerical", "Age", "median")
    _log_clean(sid, "duplicates", None, "drop_duplicates")
    _log_transform(sid, "minmax", "Age")
    append_eda_history(sid, {"analysis_type": "univariate_numerical"})

    d = report_service.collect_history(sid)["data"]
    assert d["total_operations"] == 3
    assert [o["step"] for o in d["operations"]] == [1, 2, 3]
    assert d["operations"][0]["action"] == "missing_numerical → median"
    assert d["operations"][2]["action"] == "minmax"
    assert d["operations"][2]["stage"] == "transform"
    assert d["analysis_counts"] == {"eda": 1, "statistics": 0, "nlp": 0}


def test_appendix_column_map_and_numeric_summary():
    sid = _new_session()
    df = session_service.get_working_df(sid)
    d = report_service.collect_appendix(sid, df)["data"]
    assert d["total_columns"] == 4
    assert d["truncated"] is False
    by_col = {c["column"]: c for c in d["column_map"]}
    assert set(by_col) == {"Age", "Income", "Gender", "Churn"}
    assert by_col["Age"]["missing_pct"] == 10.0       # 3 of 30
    assert by_col["Gender"]["unique_count"] == 2
    assert {c["column"] for c in d["numeric_summary"]} == {"Age", "Income", "Churn"}


# Executive summary

def _facts(**over) -> dict:
    base = {
        "rows": 1200, "columns": 8,
        "quality": {"score": 72, "label": "good"},
        "profile_insights": ["'Age' has 12% missing values.", "Dataset has 3 duplicate rows."],
        "cleaning_operations": 3,
        "transformation_operations": 2,
        "eda_analyses": 4,
        "strong_correlations": [{"col1": "A", "col2": "B", "r": 0.91}],
        "stats_tests": 3, "significant_tests": 2, "significant_test_names": ["pearson", "independent_t"],
        "ml": {"problem_type": "binary_classification", "target": "Churn",
               "top_model": "Gradient Boosting", "readiness": "ready", "readiness_warnings": ["Small dataset."]},
        "not_performed": ["NLP Analysis Summary"],
    }
    base.update(over)
    return base


def test_rule_based_summary_full_facts():
    out = report_service._rule_based_summary(_facts())
    s = out["summary"]
    assert out["source"] == "rule_based"
    assert "1,200 rows" in s and "8 columns" in s
    assert "good (72/100)" in s
    assert "3 cleaning operations and 2 transformations" in s
    assert "4 exploratory analyses were recorded" in s
    assert "2 of 3 statistical tests were significant" in s
    assert "binary classification" in s and "'Churn'" in s and "Gradient Boosting" in s
    assert "Not performed: NLP Analysis Summary." in s
    assert 0 < len(out["highlights"]) <= 6
    assert any("strongly correlated" in h for h in out["highlights"])


def test_rule_based_summary_singular_wording():
    s = report_service._rule_based_summary(
        _facts(eda_analyses=1, cleaning_operations=1, transformation_operations=0,
               stats_tests=1, significant_tests=1))["summary"]
    assert "1 exploratory analysis was recorded" in s
    assert "1 cleaning operation." in s
    assert "The statistical test was significant" in s
    assert "analysiss" not in s


def test_rule_based_summary_minimal_facts():
    out = report_service._rule_based_summary({"rows": 10, "columns": 2, "not_performed": []})
    assert "10 rows" in out["summary"]
    assert out["highlights"] == []


def test_build_facts_uses_only_included_sections():
    sid = _new_session()
    _log_clean(sid)
    df = session_service.get_working_df(sid)
    collected = {
        "cleaning": report_service.collect_cleaning(sid),
        "eda": report_service.collect_eda(sid, df),       # not performed
    }
    facts = report_service._build_facts(df, collected)
    assert facts["cleaning_operations"] == 1
    assert "quality" not in facts and "ml" not in facts   # profiling / ML were not selected
    assert facts["not_performed"] == ["Exploratory Data Analysis Summary"]


@pytest.mark.asyncio
async def test_executive_summary_always_produces_text():
    sid = _new_session()
    df = session_service.get_working_df(sid)
    collected = {"cleaning": report_service.collect_cleaning(sid)}
    sec = await report_service.collect_executive_summary(df, collected)
    assert sec["performed"] is True
    assert sec["data"]["source"] in ["gemini", "rule_based"]
    assert len(sec["data"]["summary"]) > 30
    assert "Not performed" in sec["data"]["summary"] or sec["data"]["source"] == "gemini"


# build_report

async def _full_session() -> str:
    """A session where every section except NLP has data."""
    from services import ml_service

    df = _df(120)
    sid = _new_session(df)
    _log_clean(sid, "missing_numerical", "Age", "median", rb=120, ra=120)
    _log_transform(sid, "minmax", "Income")
    append_eda_history(sid, {"analysis_type": "univariate_numerical", "column": "Age",
                             "filter_count": 0, "rows_analysed": 120})
    append_stats_history(sid, {"test_name": "pearson", "options": {"col_x": "Age", "col_y": "Income"},
                               "significant": True, "p_value": 0.02, "plain_english": "Age and Income move together."})
    strategy = await ml_service.build_ml_strategy(
        session_service.get_working_df(sid), "Churn", "predict churn")
    set_ml_result(sid, strategy)
    return sid


@pytest.mark.asyncio
async def test_build_report_full_structure():
    sid = await _full_session()
    df = session_service.get_working_df(sid)
    report = await report_service.build_report(sid, df, None)

    assert [s["key"] for s in report["sections"]] == report_service.SECTION_KEYS
    assert [s["number"] for s in report["sections"]] == list(range(1, 11))
    assert report["not_performed"] == ["NLP Analysis Summary"]
    assert report["dataset"] == {"rows": 120, "columns": 4}
    assert report["generated_at"].endswith("UTC")
    json.dumps(report, allow_nan=False)


@pytest.mark.asyncio
async def test_build_report_subset_keeps_canonical_order():
    sid = _new_session()
    _log_clean(sid)
    df = session_service.get_working_df(sid)
    report = await report_service.build_report(sid, df, ["history", "cleaning", "executive_summary"])
    assert [s["key"] for s in report["sections"]] == ["executive_summary", "cleaning", "history"]
    assert [s["number"] for s in report["sections"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_build_report_skips_unrun_sections_gracefully():
    sid = _new_session()
    df = session_service.get_working_df(sid)
    sections = ["cleaning", "transformation", "eda", "statistics", "nlp", "ml_strategy", "history"]
    report = await report_service.build_report(sid, df, sections)
    assert len(report["not_performed"]) == 7
    assert all(not s["performed"] and s["note"] for s in report["sections"])


@pytest.mark.asyncio
async def test_build_report_isolates_a_failing_section(monkeypatch):
    def boom(session_id):
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(report_service, "collect_cleaning", boom)
    sid = _new_session()
    df = session_service.get_working_df(sid)
    report = await report_service.build_report(sid, df, ["cleaning", "appendix"])

    cleaning, appendix = report["sections"]
    assert cleaning["performed"] is False
    assert cleaning["note"] == "This section could not be generated."
    assert appendix["performed"] is True               # the rest of the report still builds
    assert "collector exploded" not in json.dumps(report)


@pytest.mark.asyncio
async def test_profiling_is_recomputed_when_cache_is_cleared():
    sid = _new_session()
    assert get_profile(sid) is None                    # as after any clean/transform apply
    df = session_service.get_working_df(sid)
    report = await report_service.build_report(sid, df, ["profiling"])
    sec = report["sections"][0]
    assert sec["performed"] is True
    assert sec["data"]["dimensions"]["rows"] == 30
    assert get_profile(sid) is not None


@pytest.mark.asyncio
async def test_build_report_does_not_mutate_dataset():
    sid = await _full_session()
    df = session_service.get_working_df(sid)
    before = df.copy(deep=True)
    await report_service.build_report(sid, df, None)
    pd.testing.assert_frame_equal(before, df)


# Template rendering

@pytest.mark.asyncio
async def test_render_full_report_contains_every_section():
    sid = await _full_session()
    df = session_service.get_working_df(sid)
    report = await report_service.build_report(sid, df, None)
    html = report_service.render_report_html(report)

    assert html.lstrip().lower().startswith("<!doctype html")
    assert "DataSaathi Analysis Report" in html
    for title in report_service.SECTION_TITLES.values():
        assert title in html
    assert "Age and Income move together." in html     # statistics plain English
    assert "binary classification" in html             # ML section
    assert "Not performed" in html                     # the NLP section
    assert "missing numerical" in html                 # cleaning / history rows


@pytest.mark.asyncio
async def test_render_all_skipped_report():
    sid = _new_session()
    df = session_service.get_working_df(sid)
    sections = ["cleaning", "transformation", "eda", "statistics", "nlp", "ml_strategy", "history"]
    report = await report_service.build_report(sid, df, sections)
    html = report_service.render_report_html(report)
    assert html.count("Not performed") >= 7
    assert "No cleaning operations were applied." in html


@pytest.mark.asyncio
async def test_render_escapes_column_names():
    df = pd.DataFrame({"<script>alert(1)</script>": [1, 2, 3], "B": [4, 5, 6]})
    sid = _new_session(df)
    report = await report_service.build_report(sid, session_service.get_working_df(sid), ["appendix"])
    html = report_service.render_report_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


@pytest.mark.asyncio
async def test_render_escapes_text_fields():
    sid = _new_session()
    append_stats_history(sid, {"test_name": "pearson", "significant": True, "p_value": 0.01,
                               "plain_english": "<img src=x onerror=alert(1)>"})
    report = await report_service.build_report(
        sid, session_service.get_working_df(sid), ["statistics"])
    html = report_service.render_report_html(report)
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_template_has_no_external_resources():
    text = (report_service.TEMPLATE_DIR / report_service.TEMPLATE_NAME).read_text(encoding="utf-8")
    assert "<script" not in text.lower()
    assert "http://" not in text and "https://" not in text
    assert "|safe" not in text


# Generation & export (service level)

@pytest.mark.asyncio
async def test_generate_html_writes_file_and_describes_it():
    sid = _new_session()
    _log_clean(sid)
    result = await report_service.generate_report(sid, ["executive_summary", "cleaning"], "html")

    assert result["format"] == "html"
    assert result["file_size_kb"] > 0
    assert [s["key"] for s in result["sections_included"]] == ["executive_summary", "cleaning"]
    assert result["sections_not_performed"] == []
    path = report_service.get_report_file(sid, "html")
    assert path is not None and path.exists()
    assert "DataSaathi Analysis Report" in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_generate_rejects_invalid_format():
    with pytest.raises(ValueError):
        await report_service.generate_report(_new_session(), None, "docx")


@pytest.mark.asyncio
async def test_generate_pdf_without_engine_raises_clear_error(monkeypatch):
    monkeypatch.setattr(report_service, "pdf_available", lambda: False)
    with pytest.raises(PDFEngineUnavailable):
        await report_service.generate_report(_new_session(), None, "pdf")


@pytest.mark.asyncio
async def test_generate_html_removes_stale_pdf():
    sid = _new_session()
    report_service.TEMPLATE_DIR  # noqa: B018 - template dir must exist
    stale = report_service._report_path(sid, "pdf")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"%PDF-1.4 stale")
    await report_service.generate_report(sid, ["cleaning"], "html")
    assert report_service.get_report_file(sid, "pdf") is None


def test_get_report_file_rejects_unknown_format_and_missing_files():
    sid = _new_session()
    assert report_service.get_report_file(sid, "docx") is None
    assert report_service.get_report_file(sid, "html") is None


@pytest.mark.asyncio
async def test_generate_pdf_when_engine_available(pdf_engine):
    sid = _new_session()
    _log_clean(sid)
    result = await report_service.generate_report(sid, None, "pdf")
    assert result["format"] == "pdf"
    pdf_path = report_service.get_report_file(sid, "pdf")
    assert pdf_path is not None
    assert pdf_path.read_bytes()[:4] == b"%PDF"
    assert report_service.get_report_file(sid, "html") is not None  # preview is always written


# History export

def test_history_export_contents():
    sid = _new_session()
    _log_clean(sid)
    _log_transform(sid, "minmax", "Age")
    append_eda_history(sid, {"analysis_type": "univariate_numerical"})
    append_stats_history(sid, {"test_name": "pearson", "significant": True})
    append_nlp_history(sid, {"analysis_type": "sentiment", "column": "Review"})

    export = report_service.build_history_export(sid)
    assert export["session_id"] == sid
    assert len(export["operations"]) == 2
    assert len(export["eda_analyses"]) == 1
    assert len(export["statistical_tests"]) == 1
    assert len(export["nlp_analyses"]) == 1
    assert "exported_at" in export
    json.dumps(export, allow_nan=False)


def test_history_export_empty_session():
    export = report_service.build_history_export(_new_session())
    assert export["operations"] == []
    assert export["eda_analyses"] == []


# Router: GET /sections

def test_sections_endpoint_shape(client):
    sid = _upload_df(client, _df())
    resp = client.get(f"/api/report/{sid}/sections")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sections"]) == 10
    assert {"key", "title", "completed", "detail"} <= set(data["sections"][0])
    assert data["formats"]["html"] is True
    assert isinstance(data["formats"]["pdf"], bool)


def test_sections_endpoint_reflects_activity(client):
    sid = _upload_df(client, _df())
    client.post(f"/api/clean/{sid}/apply",
                json={"issue_type": "missing_numerical", "column": "Age", "strategy": "median"})
    client.post(f"/api/eda/{sid}/analyze",
                json={"analysis_type": "univariate_numerical", "column": "Income"})
    by_key = {s["key"]: s for s in client.get(f"/api/report/{sid}/sections").json()["sections"]}
    assert by_key["cleaning"]["completed"] is True
    assert by_key["history"]["completed"] is True
    assert by_key["eda"]["completed"] is True
    assert by_key["statistics"]["completed"] is False


def test_sections_endpoint_reports_pdf_unavailable(client, monkeypatch):
    monkeypatch.setattr(report_service, "pdf_available", lambda: False)
    sid = _upload_df(client, _df())
    assert client.get(f"/api/report/{sid}/sections").json()["formats"]["pdf"] is False


# Router: POST /generate

def test_generate_default_is_html_with_all_sections(client):
    sid = _upload_df(client, _df())
    resp = client.post(f"/api/report/{sid}/generate", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "generated"
    assert data["format"] == "html"
    assert len(data["sections_included"]) == 10
    assert data["download_url"] == f"/api/report/{sid}/download/html"
    assert data["preview_url"].endswith("?inline=true")
    assert "Data Cleaning Summary" in data["sections_not_performed"]
    assert "Data Profiling Summary" not in data["sections_not_performed"]


def test_generate_response_never_exposes_file_paths(client):
    sid = _upload_df(client, _df())
    text = client.post(f"/api/report/{sid}/generate", json={"format": "html"}).text.lower()
    assert "uploads" not in text
    assert ".html" not in text.replace("/download/html", "")
    assert "\\" not in text


def test_generate_subset_of_sections(client):
    sid = _upload_df(client, _df())
    resp = client.post(f"/api/report/{sid}/generate",
                       json={"sections": ["history", "executive_summary"], "format": "html"})
    assert resp.status_code == 200
    assert [s["key"] for s in resp.json()["sections_included"]] == ["executive_summary", "history"]


def test_generate_validation_errors(client):
    sid = _upload_df(client, _df())
    assert client.post(f"/api/report/{sid}/generate", json={"format": "docx"}).status_code == 400
    assert client.post(f"/api/report/{sid}/generate", json={"sections": ["nonsense"]}).status_code == 400
    assert client.post(f"/api/report/{sid}/generate", json={"sections": []}).status_code == 400


def test_generate_invalid_session_and_uuid(client):
    assert client.post("/api/report/00000000-0000-4000-8000-000000000000/generate",
                       json={}).status_code == 404
    assert client.post("/api/report/not-a-uuid/generate", json={}).status_code == 400


def test_generate_pdf_without_engine_returns_400(client, monkeypatch):
    monkeypatch.setattr(report_service, "pdf_available", lambda: False)
    sid = _upload_df(client, _df())
    resp = client.post(f"/api/report/{sid}/generate", json={"format": "pdf"})
    assert resp.status_code == 400
    assert "html" in resp.json()["detail"].lower()


def test_generate_does_not_modify_dataset(client):
    sid = _upload_df(client, _df())
    before = client.get(f"/api/dataset/{sid}/info").json()
    client.post(f"/api/report/{sid}/generate", json={})
    after = client.get(f"/api/dataset/{sid}/info").json()
    assert (after["rows"], after["columns"]) == (before["rows"], before["columns"])
    assert client.get(f"/api/clean/{sid}/history").json()["total_operations"] == 0


def test_profiling_section_survives_cache_invalidation(client):
    sid = _upload_df(client, _df())
    client.post(f"/api/profile/{sid}/run")
    client.post(f"/api/clean/{sid}/apply",
                json={"issue_type": "missing_numerical", "column": "Age", "strategy": "median"})
    assert get_profile(sid) is None                    # cleaning cleared it
    resp = client.post(f"/api/report/{sid}/generate", json={"sections": ["profiling"]})
    assert resp.status_code == 200
    assert resp.json()["sections_included"][0]["completed"] is True


# Router: GET /download/{format}

def test_download_html_headers(client):
    sid = _upload_df(client, _df())
    client.post(f"/api/report/{sid}/generate", json={})
    resp = client.get(f"/api/report/{sid}/download/html")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    cd = resp.headers["content-disposition"]
    assert cd.startswith("attachment") and "datasaathi_report.html" in cd
    assert "default-src 'none'" in resp.headers["content-security-policy"]
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "DataSaathi Analysis Report" in resp.text


def test_download_inline_for_preview(client):
    sid = _upload_df(client, _df())
    client.post(f"/api/report/{sid}/generate", json={})
    resp = client.get(f"/api/report/{sid}/download/html", params={"inline": "true"})
    assert resp.status_code == 200
    assert resp.headers["content-disposition"].startswith("inline")


def test_download_before_generate_returns_404(client):
    sid = _upload_df(client, _df())
    assert client.get(f"/api/report/{sid}/download/html").status_code == 404
    assert client.get(f"/api/report/{sid}/download/pdf").status_code == 404


def test_download_invalid_format_returns_400(client):
    sid = _upload_df(client, _df())
    assert client.get(f"/api/report/{sid}/download/docx").status_code == 400


def test_download_invalid_session(client):
    assert client.get("/api/report/00000000-0000-4000-8000-000000000000/download/html").status_code == 404
    assert client.get("/api/report/not-a-uuid/download/html").status_code == 400


def test_download_pdf_when_engine_available(client, pdf_engine):
    sid = _upload_df(client, _df())
    assert client.post(f"/api/report/{sid}/generate", json={"format": "pdf"}).status_code == 200
    resp = client.get(f"/api/report/{sid}/download/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    cd = resp.headers["content-disposition"]
    assert cd.startswith("attachment") and "datasaathi_report.pdf" in cd
    assert resp.content[:4] == b"%PDF"


# Router: history completeness and export

def test_report_history_section_is_complete(client):
    sid = _upload_df(client, _df())
    client.post(f"/api/clean/{sid}/apply",
                json={"issue_type": "missing_numerical", "column": "Age", "strategy": "median"})
    client.post(f"/api/clean/{sid}/apply",
                json={"issue_type": "missing_numerical", "column": "Income", "strategy": "mean"})
    client.post(f"/api/report/{sid}/generate", json={"sections": ["history"]})
    html = client.get(f"/api/report/{sid}/download/html").text
    assert "missing numerical → median" in html
    assert "missing numerical → mean" in html


def test_history_export_endpoint(client):
    sid = _upload_df(client, _df())
    client.post(f"/api/clean/{sid}/apply",
                json={"issue_type": "missing_numerical", "column": "Age", "strategy": "median"})
    client.post(f"/api/eda/{sid}/analyze",
                json={"analysis_type": "univariate_numerical", "column": "Income"})

    resp = client.get(f"/api/report/{sid}/history-export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    cd = resp.headers["content-disposition"]
    assert cd.startswith("attachment") and cd.rstrip('"').endswith(".json")
    body = resp.json()
    assert body["session_id"] == sid
    assert len(body["operations"]) == 1
    assert body["operations"][0]["strategy"] == "median"
    assert len(body["eda_analyses"]) == 1


def test_history_export_empty_session_endpoint(client):
    sid = _upload_df(client, _df())
    body = client.get(f"/api/report/{sid}/history-export").json()
    assert body["operations"] == []


def test_history_export_invalid_session(client):
    assert client.get("/api/report/00000000-0000-4000-8000-000000000000/history-export").status_code == 404
    assert client.get("/api/report/not-a-uuid/history-export").status_code == 400