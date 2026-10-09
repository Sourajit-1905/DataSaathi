"""
Stage 8 Tests — ML Strategy Service.
Advisory only: these tests verify recommendations, never model training.

Covers problem-type detection, target/imbalance/readiness/leakage/collinearity
analysis, model scoring, preprocessing / feature-engineering / imbalance advice,
split / CV / metric choice, architecture + roadmap, reasoning fallback, the master
strategy builder, JSON-safety, and the router endpoints.

Nothing here needs Gemini or any heavy library. If GEMINI_API_KEY is set, the
reasoning source may be "gemini" instead of "rule_based"; tests accept both.
"""

import io
import json

import numpy as np
import pandas as pd
import pytest

from core.exceptions import ColumnNotFoundError


# Helpers & Fixtures

def _churn_df(n: int = 300, churn_rate: float = 0.5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = int(n * churn_rate)
    churn = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(churn)
    return pd.DataFrame({
        "Tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.normal(70, 20, n).round(2),
        "Plan": rng.choice(["Basic", "Standard", "Premium"], n),
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Churn": churn,
    })


def _regression_df(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = rng.uniform(50, 200, n)
    rooms = rng.integers(1, 7, n)
    price = 50 * size + 1000 * rooms + rng.normal(0, 500, n)
    return pd.DataFrame({
        "Size": size.round(1),
        "Rooms": rooms,
        "Age": rng.integers(1, 40, n),
        "Price": price.round(2),
    })


def _ctx(**over) -> dict:
    base = dict(
        n_rows=6000, n_features=10, n_numeric=10, n_categorical=0,
        n_datetime=0, n_text=0, missing_pct=0.0, skewed_cols=[],
        outlier_cols=[], high_cardinality=[], multicollinear=False,
        imbalanced=False,
    )
    base.update(over)
    return base


def _roles(**over) -> dict:
    base = dict(numeric=[], categorical=[], datetime=[], text=[], high_cardinality=[])
    base.update(over)
    return base


def _score(models: list[dict], prefix: str) -> int:
    return next(m for m in models if m["name"].startswith(prefix))["score"]


def _upload_df(client, df) -> str:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    return resp.json()["session_id"]


@pytest.fixture
def churn_df():
    return _churn_df()


@pytest.fixture
def imbalanced_df():
    return _churn_df(churn_rate=0.1)


@pytest.fixture
def regression_df():
    return _regression_df()


# Sanitizer

def test_sanitize_makes_everything_json_safe():
    from services.ml_service import _sanitize
    messy = {
        "b": np.bool_(True),
        "i": np.int64(5),
        "f": np.float64(1.5),
        "nan": float("nan"),
        "inf": float("inf"),
        "arr": np.array([1, 2, 3]),
        "set": {1},
        "nested": [{"x": np.float32(2.5)}],
    }
    clean = _sanitize(messy)
    json.dumps(clean, allow_nan=False)  # raises if anything unsafe remains
    assert clean["b"] is True
    assert clean["nan"] is None and clean["inf"] is None
    assert clean["arr"] == [1, 2, 3]


# 1 - Problem Type Detection

def test_m1_binary_numeric_target(churn_df):
    from services.ml_service import detect_problem_type
    r = detect_problem_type(churn_df, "Churn")
    assert r["problem_type"] == "binary_classification"
    assert r["n_classes"] == 2
    assert r["confidence"] == "high"
    assert r["source"] == "target_analysis"


def test_m1_binary_string_target():
    from services.ml_service import detect_problem_type
    df = pd.DataFrame({"Outcome": ["Yes", "No"] * 20, "X": range(40)})
    assert detect_problem_type(df, "Outcome")["problem_type"] == "binary_classification"


def test_m1_continuous_target_is_regression(regression_df):
    from services.ml_service import detect_problem_type
    r = detect_problem_type(regression_df, "Price")
    assert r["problem_type"] == "regression"
    assert r["n_classes"] is None


def test_m1_string_multiclass():
    from services.ml_service import detect_problem_type
    df = pd.DataFrame({"Segment": ["Gold", "Silver", "Bronze", "Basic"] * 10, "X": range(40)})
    r = detect_problem_type(df, "Segment")
    assert r["problem_type"] == "multiclass_classification"
    assert r["n_classes"] == 4


def test_m1_low_cardinality_integer_defaults_to_multiclass():
    from services.ml_service import detect_problem_type
    df = pd.DataFrame({"Grade": [1, 2, 3, 4, 5] * 20, "X": range(100)})
    r = detect_problem_type(df, "Grade")
    assert r["problem_type"] == "multiclass_classification"
    assert r["confidence"] == "medium"


def test_m1_classification_hint_raises_confidence():
    from services.ml_service import detect_problem_type
    df = pd.DataFrame({"Grade": [1, 2, 3, 4, 5] * 20, "X": range(100)})
    r = detect_problem_type(df, "Grade", "which grade will the student get")
    assert r["problem_type"] == "multiclass_classification"
    assert r["confidence"] == "high"


def test_m1_quantity_objective_flips_integer_target_to_regression():
    from services.ml_service import detect_problem_type
    df = pd.DataFrame({"Units": [1, 2, 3, 4, 5] * 20, "X": range(100)})
    r = detect_problem_type(df, "Units", "how many units will we sell")
    assert r["problem_type"] == "regression"
    assert r["source"] == "objective"


def test_m1_too_many_classes_is_invalid():
    from services.ml_service import detect_problem_type
    df = pd.DataFrame({"Label": [f"class_{i % 30}" for i in range(90)]})
    r = detect_problem_type(df, "Label")
    assert r["problem_type"] == "invalid"
    assert "too many classes" in r["reasoning"]


def test_m1_constant_target_is_invalid():
    from services.ml_service import detect_problem_type
    df = pd.DataFrame({"Flag": [1] * 50, "X": range(50)})
    assert detect_problem_type(df, "Flag")["problem_type"] == "invalid"


def test_m1_no_target_is_clustering(churn_df):
    from services.ml_service import detect_problem_type
    r = detect_problem_type(churn_df, None)
    assert r["problem_type"] == "clustering"
    assert r["source"] == "no_target"
    assert r["target"] is None


def test_m1_override_wins(churn_df):
    from services.ml_service import detect_problem_type
    r = detect_problem_type(churn_df, "Churn", override="regression")
    assert r["problem_type"] == "regression"
    assert r["source"] == "user_override"


def test_m1_override_clustering_drops_target(churn_df):
    from services.ml_service import detect_problem_type
    r = detect_problem_type(churn_df, "Churn", override="clustering")
    assert r["problem_type"] == "clustering"
    assert r["target"] is None


def test_m1_invalid_override_raises(churn_df):
    from services.ml_service import detect_problem_type
    with pytest.raises(ValueError):
        detect_problem_type(churn_df, "Churn", override="time_travel")


def test_m1_supervised_override_without_target_raises(churn_df):
    from services.ml_service import detect_problem_type
    with pytest.raises(ValueError):
        detect_problem_type(churn_df, None, override="regression")


def test_m1_unknown_target_raises(churn_df):
    from services.ml_service import detect_problem_type
    with pytest.raises(ColumnNotFoundError):
        detect_problem_type(churn_df, "NoSuchColumn")


# 2 - Target Analysis

def test_m2_classification_class_distribution(churn_df):
    from services.ml_service import analyze_target
    info = analyze_target(churn_df, "Churn", "binary_classification")
    assert len(info["class_distribution"]) == 2
    assert abs(sum(c["pct"] for c in info["class_distribution"]) - 100.0) < 0.1


def test_m2_regression_stats(regression_df):
    from services.ml_service import analyze_target
    info = analyze_target(regression_df, "Price", "regression")
    for key in ["mean", "std", "min", "max", "skewness", "outlier_pct"]:
        assert key in info


def test_m2_no_target():
    from services.ml_service import analyze_target
    assert analyze_target(pd.DataFrame({"A": [1]}), None, "clustering") == {"target": None}


# 3 - Class Imbalance

def test_m3_balanced(churn_df):
    from services.ml_service import detect_class_imbalance
    r = detect_class_imbalance(churn_df, "Churn", "binary_classification")
    assert r["applicable"] is True
    assert r["is_imbalanced"] is False
    assert r["severity"] == "none"


def test_m3_moderate_imbalance(imbalanced_df):
    from services.ml_service import detect_class_imbalance
    r = detect_class_imbalance(imbalanced_df, "Churn", "binary_classification")
    assert r["is_imbalanced"] is True
    assert r["severity"] == "moderate"
    assert r["minority_pct"] == 10.0
    assert r["majority_pct"] == 90.0
    assert r["imbalance_ratio"] == 9.0


def test_m3_severe_imbalance():
    from services.ml_service import detect_class_imbalance
    df = pd.DataFrame({"Y": [0] * 190 + [1] * 10})
    r = detect_class_imbalance(df, "Y", "binary_classification")
    assert r["severity"] == "severe"


def test_m3_not_applicable_for_regression(regression_df):
    from services.ml_service import detect_class_imbalance
    r = detect_class_imbalance(regression_df, "Price", "regression")
    assert r == {"applicable": False, "is_imbalanced": False}


# Feature columns & roles

def test_feature_columns_exclude_target_identifiers_constants():
    from services.ml_service import _feature_columns
    n = 100
    df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Country": ["IN"] * n,
        "Tenure": np.random.default_rng(0).integers(1, 50, n),
        "Churn": [0, 1] * (n // 2),
    })
    features, ids, consts = _feature_columns(df, "Churn")
    assert features == ["Tenure"]
    assert ids == ["CustomerID"]
    assert consts == ["Country"]


def test_column_roles_splits_types():
    from services.ml_service import _column_roles
    n = 120
    df = pd.DataFrame({
        "Age": np.random.default_rng(1).integers(18, 70, n),
        "Plan": ["Basic", "Standard", "Premium"] * (n // 3),
        "Joined": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "Comment": [f"this product is really great and works well number {i}" for i in range(n)],
        "City": [f"city_{i % 60}" for i in range(n)],
    })
    roles = _column_roles(df, list(df.columns))
    assert roles["numeric"] == ["Age"]
    assert "Plan" in roles["categorical"]
    assert "Joined" in roles["datetime"]
    assert "Comment" in roles["text"]
    assert "City" in roles["high_cardinality"]
    assert "Comment" not in roles["categorical"]
    assert "Joined" not in roles["categorical"]


# 4 - Dataset Readiness

def _numeric_df(n: int, cols: int = 4, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"F{i}": rng.normal(size=n) for i in range(cols)})


def test_m4_large_clean_dataset_is_ready():
    from services.ml_service import assess_dataset_readiness
    df = _numeric_df(1500)
    r = assess_dataset_readiness(df, list(df.columns), _roles())
    assert r["score"] == 100
    assert r["label"] == "ready"
    assert r["warnings"] == []
    assert r["rows_per_feature"] == 375.0


def test_m4_tiny_dataset_needs_work():
    from services.ml_service import assess_dataset_readiness
    df = _numeric_df(50)
    r = assess_dataset_readiness(df, list(df.columns), _roles())
    assert r["score"] == 60
    assert r["label"] == "needs_work"
    assert any("50 rows" in w for w in r["warnings"])


def test_m4_no_features_flagged():
    from services.ml_service import assess_dataset_readiness
    df = _numeric_df(1500)
    r = assess_dataset_readiness(df, [], _roles())
    assert r["score"] <= 50
    assert r["rows_per_feature"] is None
    assert any("No usable feature" in w for w in r["warnings"])


def test_m4_heavy_missingness_penalised():
    from services.ml_service import assess_dataset_readiness
    df = _numeric_df(1500, cols=2)
    df.loc[:749, "F0"] = np.nan  # F0 50% missing -> 25% average
    r = assess_dataset_readiness(df, ["F0", "F1"], _roles())
    assert r["score"] == 80
    assert any("missing" in w for w in r["warnings"])


# 5 - Leakage

def test_m5_clean_features_low_risk(churn_df):
    from services.ml_service import detect_leakage_risk
    r = detect_leakage_risk(churn_df, "Churn", ["Tenure", "MonthlyCharges", "Plan"], [])
    assert r["risk"] == "low"
    assert r["flagged_features"] == []


def test_m5_target_copy_is_flagged_high(churn_df):
    from services.ml_service import detect_leakage_risk
    df = churn_df.copy()
    df["Spend"] = df["Churn"] * 100
    r = detect_leakage_risk(df, "Churn", ["Tenure", "Spend"], [])
    assert r["risk"] == "high"
    flagged = {f["column"]: f for f in r["flagged_features"]}
    assert "Spend" in flagged
    assert flagged["Spend"]["correlation"] > 0.95


def test_m5_name_overlap_is_medium_risk(churn_df):
    from services.ml_service import detect_leakage_risk
    df = churn_df.copy()
    df["ChurnScoreX"] = np.random.default_rng(5).normal(size=len(df))
    r = detect_leakage_risk(df, "Churn", ["Tenure", "ChurnScoreX"], [])
    assert r["risk"] == "medium"
    assert r["flagged_features"][0]["column"] == "ChurnScoreX"
    assert r["flagged_features"][0]["correlation"] is None


def test_m5_no_target(churn_df):
    from services.ml_service import detect_leakage_risk
    r = detect_leakage_risk(churn_df, None, ["Tenure"], ["CustomerID"])
    assert r["risk"] == "none"
    assert r["identifier_columns"] == ["CustomerID"]


# 6 - Multicollinearity

def test_m6_detects_correlated_pair():
    from services.ml_service import detect_multicollinearity
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    df = pd.DataFrame({
        "A": a,
        "B": 2 * a + rng.normal(scale=0.05, size=200),
        "C": rng.normal(size=200),
    })
    r = detect_multicollinearity(df, ["A", "B", "C"])
    assert r["has_multicollinearity"] is True
    assert {r["pairs"][0]["col1"], r["pairs"][0]["col2"]} == {"A", "B"}


def test_m6_independent_columns_clean():
    from services.ml_service import detect_multicollinearity
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"A": rng.normal(size=200), "B": rng.normal(size=200)})
    assert detect_multicollinearity(df, ["A", "B"])["has_multicollinearity"] is False


def test_m6_single_column():
    from services.ml_service import detect_multicollinearity
    r = detect_multicollinearity(pd.DataFrame({"A": [1, 2, 3]}), ["A"])
    assert r == {"has_multicollinearity": False, "pairs": []}


# Context builder

def test_context_flags_skew_and_outliers():
    from services.ml_service import _build_context
    rng = np.random.default_rng(2)
    spiky = rng.normal(0, 1, 300)
    spiky[:30] = 100  # 10% extreme values
    df = pd.DataFrame({"Normal": rng.normal(size=300), "Spiky": spiky})
    ctx = _build_context(df, ["Normal", "Spiky"], _roles(numeric=["Normal", "Spiky"]),
                         {"is_imbalanced": False}, {"has_multicollinearity": False})
    assert "Spiky" in ctx["outlier_cols"]
    assert "Spiky" in ctx["skewed_cols"]
    assert "Normal" not in ctx["outlier_cols"]
    assert ctx["n_rows"] == 300 and ctx["n_features"] == 2
    assert ctx["imbalanced"] is False


# 7 - Model Recommendation

def test_m7_catalog_sizes():
    from services.ml_service import recommend_models
    assert len(recommend_models("binary_classification", _ctx())) == 7
    assert len(recommend_models("multiclass_classification", _ctx())) == 7
    assert len(recommend_models("regression", _ctx())) == 7
    assert len(recommend_models("clustering", _ctx())) == 4


def test_m7_sorted_ranked_and_bounded():
    from services.ml_service import recommend_models
    models = recommend_models("binary_classification", _ctx())
    scores = [m["score"] for m in models]
    assert scores == sorted(scores, reverse=True)
    assert [m["rank"] for m in models] == list(range(1, 8))
    assert all(0 <= s <= 100 for s in scores)
    assert all(1 <= m["dots"] <= 5 for m in models)
    assert all(m["strengths"] and m["weaknesses"] for m in models)


def test_m7_top_three_recommended():
    from services.ml_service import recommend_models
    models = recommend_models("regression", _ctx())
    assert [m["recommended"] for m in models] == [True] * 3 + [False] * 4


def test_m7_large_clean_data_favours_gradient_boosting():
    from services.ml_service import recommend_models
    cls = recommend_models("binary_classification", _ctx())
    reg = recommend_models("regression", _ctx())
    assert cls[0]["name"].startswith("Gradient Boosting")
    assert reg[0]["name"].startswith("Gradient Boosting")


def test_m7_small_data_penalises_complex_models():
    from services.ml_service import recommend_models
    models = recommend_models("binary_classification", _ctx(n_rows=100, n_features=5))
    assert _score(models, "Gradient Boosting") < _score(models, "Random Forest")
    assert _score(models, "Gradient Boosting") < _score(models, "Logistic")
    assert models[-1]["name"].startswith("Neural Network")


def test_m7_huge_data_penalises_poor_scalers():
    from services.ml_service import recommend_models
    big = recommend_models("binary_classification", _ctx(n_rows=60000))
    mid = recommend_models("binary_classification", _ctx(n_rows=6000))
    assert _score(big, "Support Vector") < _score(mid, "Support Vector")


def test_m7_imbalance_helps_class_weight_models():
    from services.ml_service import recommend_models
    bal = recommend_models("binary_classification", _ctx())
    imb = recommend_models("binary_classification", _ctx(imbalanced=True))
    assert _score(imb, "Logistic") > _score(bal, "Logistic")
    assert _score(imb, "K-Nearest") < _score(bal, "K-Nearest")


def test_m7_multicollinearity_hurts_linear_models():
    from services.ml_service import recommend_models
    clean = recommend_models("binary_classification", _ctx())
    collinear = recommend_models("binary_classification", _ctx(multicollinear=True))
    assert _score(collinear, "Logistic") < _score(clean, "Logistic")


def test_m7_missing_values_favour_native_handlers():
    from services.ml_service import recommend_models
    clean = recommend_models("binary_classification", _ctx())
    missing = recommend_models("binary_classification", _ctx(missing_pct=10.0))
    assert _score(missing, "Gradient Boosting") > _score(clean, "Gradient Boosting")
    assert _score(missing, "Logistic") < _score(clean, "Logistic")


def test_m7_high_feature_ratio_hurts_knn():
    from services.ml_service import recommend_models
    ok = recommend_models("binary_classification", _ctx(n_rows=200, n_features=5))
    wide = recommend_models("binary_classification", _ctx(n_rows=200, n_features=100))
    assert _score(wide, "K-Nearest") < _score(ok, "K-Nearest")


def test_m7_outliers_help_robust_models():
    from services.ml_service import recommend_models
    clean = recommend_models("binary_classification", _ctx())
    skewed = recommend_models("binary_classification", _ctx(skewed_cols=["x"]))
    assert _score(skewed, "Random Forest") > _score(clean, "Random Forest")
    assert _score(skewed, "Logistic") < _score(clean, "Logistic")


def test_m7_clustering_prefers_kmeans():
    from services.ml_service import recommend_models
    models = recommend_models("clustering", _ctx(n_rows=1000))
    assert models[0]["name"] == "K-Means"


# 8 - Comparison Table

def test_m8_comparison_rows():
    from services.ml_service import recommend_models, build_model_comparison
    models = recommend_models("binary_classification", _ctx())
    table = build_model_comparison(models)
    assert len(table) == len(models)
    assert table[0]["model"] == models[0]["name"]
    assert table[0]["rank"] == 1
    assert all(r["needs_scaling"] in ("Yes", "No") for r in table)


# 9 - Preprocessing

_NO_SCALE = [{"name": "X", "needs_scaling": False}] * 3
_SCALE = [{"name": "X", "needs_scaling": True}] * 3


def _steps(df, roles, ctx, models=_NO_SCALE, ids=None, consts=None, coll=None):
    from services.ml_service import recommend_preprocessing
    return recommend_preprocessing(
        df, roles, ctx, models, ids or [], consts or [],
        coll or {"has_multicollinearity": False, "pairs": []},
    )


def _techniques(steps):
    return [s["technique"] for s in steps]


def test_m9_drops_identifier_and_constant_columns():
    df = pd.DataFrame({"A": [1, 2, 3]})
    steps = _steps(df, _roles(), _ctx(), ids=["CustomerID"], consts=["Country"])
    step = next(s for s in steps if s["technique"].startswith("Drop identifier"))
    assert set(step["applies_to"]) == {"CustomerID", "Country"}


def test_m9_imputation_priority_follows_missing_pct():
    df = pd.DataFrame({"A": [1.0, None, 3.0]})
    high = _steps(df, _roles(numeric=["A"]), _ctx(missing_pct=10.0))
    low = _steps(df, _roles(numeric=["A"]), _ctx(missing_pct=3.0))
    assert next(s for s in high if "Impute" in s["technique"])["priority"] == "high"
    assert next(s for s in low if "Impute" in s["technique"])["priority"] == "medium"


def test_m9_no_imputation_when_nothing_missing():
    df = pd.DataFrame({"A": [1, 2, 3]})
    assert not any("Impute" in t for t in _techniques(_steps(df, _roles(numeric=["A"]), _ctx())))


def test_m9_encoding_steps_split_by_cardinality():
    df = pd.DataFrame({"Plan": ["a"] * 3, "City": ["b"] * 3})
    roles = _roles(categorical=["Plan", "City"], high_cardinality=["City"])
    steps = _steps(df, roles, _ctx())
    one_hot = next(s for s in steps if s["technique"].startswith("One-hot"))
    target = next(s for s in steps if s["technique"].startswith("Target or frequency"))
    assert one_hot["applies_to"] == ["Plan"]
    assert target["applies_to"] == ["City"]


def test_m9_scaling_when_top_models_need_it():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0]})
    steps = _steps(df, _roles(numeric=["A", "B"]), _ctx(), models=_SCALE)
    scale = next(s for s in steps if "Scale" in s["technique"])
    assert scale["priority"] == "high"


def test_m9_scaling_when_ranges_differ_even_if_models_dont_need_it():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": rng.normal(0, 1, 100), "B": rng.normal(0, 1000, 100)})
    steps = _steps(df, _roles(numeric=["A", "B"]), _ctx(), models=_NO_SCALE)
    scale = next(s for s in steps if "Scale" in s["technique"])
    assert scale["priority"] == "medium"


def test_m9_no_scaling_when_not_needed():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": rng.normal(0, 1, 100), "B": rng.normal(0, 1, 100)})
    steps = _steps(df, _roles(numeric=["A", "B"]), _ctx(), models=_NO_SCALE)
    assert not any("Scale" in t for t in _techniques(steps))


def test_m9_skew_outlier_and_collinearity_steps():
    df = pd.DataFrame({"A": [1, 2, 3]})
    coll = {"has_multicollinearity": True,
            "pairs": [{"col1": "A", "col2": "B", "correlation": 0.95}]}
    steps = _steps(df, _roles(), _ctx(skewed_cols=["A"], outlier_cols=["A"]), coll=coll)
    techs = _techniques(steps)
    assert any("log / power" in t for t in techs)
    assert any("Cap or robust-scale" in t for t in techs)
    assert any("correlated" in t for t in techs)


# 10 - Feature Engineering

def _fe(roles, ctx, problem="binary_classification", models=None):
    from services.ml_service import recommend_feature_engineering
    return recommend_feature_engineering(roles, ctx, problem, models or [{"name": "Random Forest"}])


def test_m10_datetime_parts_and_lags():
    recs = _fe(_roles(datetime=["Date"]), _ctx(), problem="regression")
    techs = [r["technique"] for r in recs]
    assert any("date parts" in t for t in techs)
    assert any("lag" in t.lower() for t in techs)


def test_m10_no_lags_for_classification():
    recs = _fe(_roles(datetime=["Date"]), _ctx())
    assert not any("lag" in r["technique"].lower() for r in recs)


def test_m10_text_vectorisation():
    recs = _fe(_roles(text=["Review"]), _ctx())
    assert any("Vectorise" in r["technique"] for r in recs)


def test_m10_high_cardinality_grouping():
    recs = _fe(_roles(high_cardinality=["City"]), _ctx())
    assert any("Group rare categories" in r["technique"] for r in recs)


def test_m10_interactions_when_linear_model_on_top():
    recs = _fe(_roles(), _ctx(n_numeric=4), models=[{"name": "Logistic Regression"}])
    assert any("interaction" in r["technique"].lower() for r in recs)


def test_m10_pca_for_many_features():
    recs = _fe(_roles(), _ctx(n_features=60))
    assert any("PCA" in r["technique"] for r in recs)


def test_m10_nothing_to_suggest_returns_empty():
    assert _fe(_roles(), _ctx(n_numeric=1, n_features=5)) == []


# 11 - Imbalance Handling

def _imb(severity="moderate", minority_count=100, pct=10.0):
    return {"is_imbalanced": True, "severity": severity,
            "minority_count": minority_count, "minority_pct": pct}


def test_m11_balanced_returns_nothing():
    from services.ml_service import recommend_imbalance_handling
    assert recommend_imbalance_handling({"is_imbalanced": False}, 1000) == []


def test_m11_core_techniques():
    from services.ml_service import recommend_imbalance_handling
    techs = [r["technique"] for r in recommend_imbalance_handling(_imb(), 1000)]
    assert "Use class weights" in techs
    assert any("Stratified" in t for t in techs)
    assert any("threshold" in t for t in techs)
    assert any("SMOTE" in t for t in techs)


def test_m11_no_smote_with_tiny_minority():
    from services.ml_service import recommend_imbalance_handling
    techs = [r["technique"] for r in recommend_imbalance_handling(_imb(minority_count=3), 1000)]
    assert not any("SMOTE" in t for t in techs)


def test_m11_undersampling_only_for_severe_and_large():
    from services.ml_service import recommend_imbalance_handling
    large = recommend_imbalance_handling(_imb(severity="severe"), 60000)
    small = recommend_imbalance_handling(_imb(severity="severe"), 1000)
    assert any("undersampling" in r["technique"] for r in large)
    assert not any("undersampling" in r["technique"] for r in small)


# 12 - Split Strategy

def test_m12_classification_is_stratified():
    from services.ml_service import recommend_split_strategy
    r = recommend_split_strategy("binary_classification", 500, False, False)
    assert r["strategy"] == "stratified"
    assert r["ratios_pct"] == {"train": 80, "validation": 0, "test": 20}
    assert "cross-validation" in r["caution"]  # small dataset note


def test_m12_imbalance_mentioned():
    from services.ml_service import recommend_split_strategy
    r = recommend_split_strategy("binary_classification", 5000, True, False)
    assert "imbalanced" in r["description"]


def test_m12_ratios_scale_with_size():
    from services.ml_service import recommend_split_strategy
    assert recommend_split_strategy("regression", 10000, False, False)["ratios_pct"] == \
        {"train": 70, "validation": 15, "test": 15}
    assert recommend_split_strategy("regression", 100000, False, False)["ratios_pct"] == \
        {"train": 90, "validation": 0, "test": 10}


def test_m12_regression_with_datetime_is_chronological():
    from services.ml_service import recommend_split_strategy
    assert recommend_split_strategy("regression", 1000, False, True)["strategy"] == "chronological"


def test_m12_plain_regression_is_random():
    from services.ml_service import recommend_split_strategy
    assert recommend_split_strategy("regression", 1000, False, False)["strategy"] == "random"


def test_m12_clustering_needs_no_split():
    from services.ml_service import recommend_split_strategy
    assert recommend_split_strategy("clustering", 1000, False, False)["strategy"] == "none"


# 13 - Cross-Validation

def test_m13_small_classification_uses_repeated_stratified():
    from services.ml_service import recommend_cross_validation
    r = recommend_cross_validation("binary_classification", 500, False, False)
    assert r["method"] == "Repeated Stratified K-Fold"
    assert r["folds"] == 5 and r["repeats"] == 3


def test_m13_small_regression_uses_repeated_kfold():
    from services.ml_service import recommend_cross_validation
    assert recommend_cross_validation("regression", 500, False, False)["method"] == "Repeated K-Fold"


def test_m13_standard_and_large():
    from services.ml_service import recommend_cross_validation
    std = recommend_cross_validation("binary_classification", 5000, False, False)
    big = recommend_cross_validation("binary_classification", 80000, False, False)
    assert std["method"] == "Stratified K-Fold" and std["folds"] == 5
    assert big["folds"] == 3


def test_m13_time_series_split():
    from services.ml_service import recommend_cross_validation
    r = recommend_cross_validation("regression", 5000, False, True)
    assert r["method"] == "TimeSeriesSplit"


def test_m13_clustering_has_no_cv():
    from services.ml_service import recommend_cross_validation
    r = recommend_cross_validation("clustering", 1000, False, False)
    assert r["method"] == "none" and r["folds"] is None


# 14 - Metrics

def test_m14_binary_balanced_uses_roc_auc():
    from services.ml_service import recommend_metrics
    r = recommend_metrics("binary_classification", False, {})
    assert r["primary"] == "ROC-AUC"
    assert r["avoid"] == []


def test_m14_binary_imbalanced_avoids_accuracy():
    from services.ml_service import recommend_metrics
    r = recommend_metrics("binary_classification", True, {})
    assert r["primary"] == "PR-AUC"
    assert "Accuracy" in r["avoid"]


def test_m14_multiclass_uses_macro_f1():
    from services.ml_service import recommend_metrics
    assert recommend_metrics("multiclass_classification", False, {})["primary"] == "Macro F1"
    assert recommend_metrics("multiclass_classification", True, {})["avoid"] == ["Accuracy alone"]


def test_m14_regression_metric_depends_on_target_shape():
    from services.ml_service import recommend_metrics
    assert recommend_metrics("regression", False, {})["primary"] == "RMSE"
    assert recommend_metrics("regression", False, {"skewness": 2.5})["primary"] == "MAE"
    assert recommend_metrics("regression", False, {"outlier_pct": 8.0})["primary"] == "MAE"


def test_m14_clustering_metrics():
    from services.ml_service import recommend_metrics
    assert recommend_metrics("clustering", False, {})["primary"] == "Silhouette Score"


# 15 - Architecture


def test_m15_full_pipeline_steps():
    from services.ml_service import build_architecture
    steps = build_architecture(
        "binary_classification",
        [{"name": "Random Forest"}],
        [{"technique": "Impute missing values"}],
        [{"technique": "Extract date parts"}],
        [{"technique": "Use class weights"}],
        {"method": "Stratified K-Fold", "folds": 5},
    )
    names = [s["name"] for s in steps]
    assert names[0] == "Load cleaned data"
    assert "Feature engineering" in names
    assert "Preprocessing (ColumnTransformer)" in names
    assert "Train/test split" in names
    assert "Imbalance handling" in names
    assert "Model: Random Forest" in names
    assert names[-1] == "Hyperparameter tuning"
    assert [s["step"] for s in steps] == list(range(1, len(steps) + 1))
    evaluation = next(s for s in steps if s["name"] == "Evaluation")
    assert "Stratified K-Fold (5 folds)" in evaluation["description"]


def test_m15_clustering_has_no_split_step():
    from services.ml_service import build_architecture
    steps = build_architecture("clustering", [{"name": "K-Means"}], [], [], [],
                               {"method": "none", "folds": None})
    assert "Train/test split" not in [s["name"] for s in steps]

 
# 16 - Roadmap


def _road(problem="binary_classification", warnings=None, flagged=None,
          imbalance=None, preprocessing=None):
    from services.ml_service import build_roadmap
    return build_roadmap(
        problem,
        [{"name": "Random Forest"}, {"name": "Logistic Regression"}],
        {"warnings": warnings or []},
        {"flagged_features": flagged or []},
        preprocessing or [{"technique": "Impute missing values"}],
        imbalance or [],
        {"primary": "ROC-AUC", "reason": "Balanced classes."},
    )


def test_m16_steps_sequential_and_bookended():
    road = _road()
    assert [s["step_number"] for s in road] == list(range(1, len(road) + 1))
    assert road[0]["title"] == "Confirm the problem framing"
    assert road[-1]["title"] == "Validate before deployment"


def test_m16_baseline_and_metric_mentioned():
    road = _road()
    baseline = next(s for s in road if s["title"] == "Train a baseline")
    evaluate = next(s for s in road if s["title"] == "Evaluate properly")
    assert "Random Forest" in baseline["description"]
    assert "Logistic Regression" in baseline["description"]
    assert "ROC-AUC" in evaluate["description"]


def test_m16_readiness_step_only_when_needed():
    assert not any(s["title"] == "Resolve data-readiness concerns" for s in _road())
    with_warn = _road(warnings=["Only 50 rows."])
    step = next(s for s in with_warn if s["title"] == "Resolve data-readiness concerns")
    assert "Only 50 rows." in step["description"]


def test_m16_leakage_named_in_readiness_step():
    road = _road(flagged=[{"column": "Spend"}])
    step = next(s for s in road if s["title"] == "Resolve data-readiness concerns")
    assert "Spend" in step["description"]


def test_m16_imbalance_step():
    road = _road(imbalance=[{"technique": "Use class weights"}])
    assert any(s["title"] == "Address class imbalance" for s in road)


def test_m16_clustering_skips_validation_split():
    road = _road(problem="clustering")
    assert not any(s["title"] == "Create a validation split" for s in road)


# 17 - Reasoning

def _sample_models():
    from services.ml_service import recommend_models
    return recommend_models("binary_classification", _ctx())


def test_m17_rule_based_reasoning_mentions_goal():
    from services.ml_service import _rule_based_reasoning
    models = _sample_models()
    problem = {"problem_type": "binary_classification", "target": "Churn"}
    r = _rule_based_reasoning(problem, models, "predict which customers will churn", _ctx())
    assert r["source"] == "rule_based"
    assert "binary classification" in r["summary"]
    assert "predict which customers will churn" in r["summary"]
    assert set(r["model_reasons"]) == {m["name"] for m in models[:5]}


def test_m17_rule_based_reasoning_without_goal():
    from services.ml_service import _rule_based_reasoning
    problem = {"problem_type": "regression", "target": "Price"}
    r = _rule_based_reasoning(problem, _sample_models(), "", _ctx())
    assert "For your goal" not in r["summary"]


@pytest.mark.asyncio
async def test_m17_generate_reasoning_always_returns_usable_text():
    from services.ml_service import generate_reasoning
    models = _sample_models()
    problem = {"problem_type": "binary_classification", "target": "Churn", "n_classes": 2}
    r = await generate_reasoning(problem, models, "predict churn", _ctx(), {"is_imbalanced": False})
    assert r["source"] in ["gemini", "rule_based"]
    assert isinstance(r["summary"], str) and len(r["summary"]) > 20
    for m in models[:5]:
        assert m["name"] in r["model_reasons"]


#18  Master builder

EXPECTED_KEYS = [
    "valid", "objective", "problem", "target_analysis", "feature_columns",
    "excluded_columns", "column_roles", "readiness", "leakage",
    "multicollinearity", "imbalance", "models", "model_comparison",
    "preprocessing", "feature_engineering", "imbalance_handling",
    "split_strategy", "cross_validation", "metrics", "architecture",
    "roadmap", "summary", "reasoning_source",
]


@pytest.mark.asyncio
async def test_m18_binary_strategy_shape(churn_df):
    from services.ml_service import build_ml_strategy
    s = await build_ml_strategy(churn_df, "Churn", "predict which customers will churn")
    for key in EXPECTED_KEYS:
        assert key in s, f"Missing key: {key}"
    assert s["valid"] is True
    assert s["problem"]["problem_type"] == "binary_classification"
    assert "Churn" not in s["feature_columns"]
    assert len(s["models"]) == 7
    assert [m["recommended"] for m in s["models"][:3]] == [True] * 3
    assert s["reasoning_source"] in ["gemini", "rule_based"]
    for m in s["models"][:5]:
        assert m["reasoning"]


@pytest.mark.asyncio
async def test_m18_output_is_strictly_json_safe(imbalanced_df):
    from services.ml_service import build_ml_strategy
    s = await build_ml_strategy(imbalanced_df, "Churn", "")
    json.dumps(s, allow_nan=False)  # fails on NaN / inf / numpy types


@pytest.mark.asyncio
async def test_m18_imbalanced_strategy(imbalanced_df):
    from services.ml_service import build_ml_strategy
    s = await build_ml_strategy(imbalanced_df, "Churn", "")
    assert s["imbalance"]["is_imbalanced"] is True
    assert s["imbalance_handling"]
    assert s["metrics"]["primary"] == "PR-AUC"
    assert "Accuracy" in s["metrics"]["avoid"]
    assert s["split_strategy"]["strategy"] == "stratified"
    assert any(r["title"] == "Address class imbalance" for r in s["roadmap"])


@pytest.mark.asyncio
async def test_m18_regression_strategy(regression_df):
    from services.ml_service import build_ml_strategy
    s = await build_ml_strategy(regression_df, "Price", "predict the price of a house")
    assert s["problem"]["problem_type"] == "regression"
    assert s["imbalance"]["applicable"] is False
    assert s["imbalance_handling"] == []
    assert s["split_strategy"]["strategy"] == "random"
    assert s["metrics"]["primary"] in ["RMSE", "MAE"]
    assert any("Regress" in m["name"] for m in s["models"])


@pytest.mark.asyncio
async def test_m18_regression_with_dates_uses_time_aware_validation():
    from services.ml_service import build_ml_strategy
    df = _regression_df()
    df["SignupDate"] = pd.date_range("2022-01-01", periods=len(df)).strftime("%Y-%m-%d")
    s = await build_ml_strategy(df, "Price", "")
    assert "SignupDate" in s["column_roles"]["datetime"]
    assert s["split_strategy"]["strategy"] == "chronological"
    assert s["cross_validation"]["method"] == "TimeSeriesSplit"
    assert any("lag" in f["technique"].lower() for f in s["feature_engineering"])


@pytest.mark.asyncio
async def test_m18_clustering_strategy(churn_df):
    from services.ml_service import build_ml_strategy
    s = await build_ml_strategy(churn_df.drop(columns=["Churn"]), None, "segment customers")
    assert s["problem"]["problem_type"] == "clustering"
    assert s["split_strategy"]["strategy"] == "none"
    assert s["cross_validation"]["method"] == "none"
    assert s["metrics"]["primary"] == "Silhouette Score"
    assert s["imbalance_handling"] == []
    assert s["models"][0]["name"] == "K-Means"


@pytest.mark.asyncio
async def test_m18_identifiers_and_constants_excluded(churn_df):
    from services.ml_service import build_ml_strategy
    df = churn_df.copy()
    df["CustomerID"] = range(1, len(df) + 1)
    df["Country"] = "IN"
    s = await build_ml_strategy(df, "Churn", "")
    assert "CustomerID" in s["excluded_columns"]["identifiers"]
    assert "Country" in s["excluded_columns"]["constants"]
    assert "CustomerID" not in s["feature_columns"]
    assert "Country" not in s["feature_columns"]
    assert any("Drop identifier" in p["technique"] for p in s["preprocessing"])


@pytest.mark.asyncio
async def test_m18_unusable_target_returns_invalid():
    from services.ml_service import build_ml_strategy
    df = pd.DataFrame({"Flag": [1] * 50, "X": range(50)})
    s = await build_ml_strategy(df, "Flag", "")
    assert s["valid"] is False
    assert "message" in s


@pytest.mark.asyncio
async def test_m18_does_not_mutate_input(churn_df):
    from services.ml_service import build_ml_strategy
    before = churn_df.copy(deep=True)
    await build_ml_strategy(churn_df, "Churn", "")
    pd.testing.assert_frame_equal(before, churn_df)


# 19 - Report Summary

@pytest.mark.asyncio
async def test_m19_summary_of_valid_strategy(churn_df):
    from services.ml_service import build_ml_strategy, ml_summary
    s = await build_ml_strategy(churn_df, "Churn", "predict churn")
    summary = ml_summary(s)
    assert summary["available"] is True
    assert summary["problem_type"] == "binary_classification"
    assert len(summary["top_models"]) == 3
    assert summary["primary_metric"] == "ROC-AUC"
    assert summary["roadmap"][0] == "Confirm the problem framing"
    json.dumps(summary, allow_nan=False)


def test_m19_summary_unavailable_for_missing_or_invalid():
    from services.ml_service import ml_summary
    assert ml_summary(None) == {"available": False}
    assert ml_summary({"valid": False}) == {"available": False}


# Integration Tests - router endpoints

def test_recommend_endpoint_binary(client):
    sid = _upload_df(client, _churn_df())
    resp = client.post(
        f"/api/ml/{sid}/recommend",
        json={"target": "Churn", "objective": "predict which customers will churn"},
    )
    assert resp.status_code == 200
    strategy = resp.json()["strategy"]
    assert strategy["problem"]["problem_type"] == "binary_classification"
    assert len(strategy["models"]) == 7
    assert strategy["roadmap"]


def test_recommend_endpoint_regression(client):
    sid = _upload_df(client, _regression_df())
    resp = client.post(f"/api/ml/{sid}/recommend", json={"target": "Price"})
    assert resp.status_code == 200
    assert resp.json()["strategy"]["problem"]["problem_type"] == "regression"


def test_recommend_endpoint_clustering_without_target(client):
    sid = _upload_df(client, _churn_df().drop(columns=["Churn"]))
    resp = client.post(f"/api/ml/{sid}/recommend", json={})
    assert resp.status_code == 200
    assert resp.json()["strategy"]["problem"]["problem_type"] == "clustering"


def test_recommend_endpoint_override(client):
    sid = _upload_df(client, _churn_df())
    resp = client.post(
        f"/api/ml/{sid}/recommend",
        json={"target": "Churn", "problem_type": "regression"},
    )
    assert resp.status_code == 200
    assert resp.json()["strategy"]["problem"]["source"] == "user_override"


def test_recommend_endpoint_imbalanced(client):
    sid = _upload_df(client, _churn_df(churn_rate=0.1))
    resp = client.post(f"/api/ml/{sid}/recommend", json={"target": "Churn"})
    assert resp.status_code == 200
    assert resp.json()["strategy"]["imbalance"]["is_imbalanced"] is True


def test_recommend_does_not_modify_dataset(client):
    sid = _upload_df(client, _churn_df())
    before = client.get(f"/api/dataset/{sid}/info").json()
    client.post(f"/api/ml/{sid}/recommend", json={"target": "Churn"})
    after = client.get(f"/api/dataset/{sid}/info").json()
    assert after["rows"] == before["rows"]
    assert after["columns"] == before["columns"]


def test_recommend_unknown_target_returns_400(client):
    sid = _upload_df(client, _churn_df())
    resp = client.post(f"/api/ml/{sid}/recommend", json={"target": "NoSuchColumn"})
    assert resp.status_code == 400


def test_recommend_invalid_override_returns_400(client):
    sid = _upload_df(client, _churn_df())
    resp = client.post(
        f"/api/ml/{sid}/recommend",
        json={"target": "Churn", "problem_type": "time_travel"},
    )
    assert resp.status_code == 400


def test_recommend_supervised_override_without_target_returns_400(client):
    sid = _upload_df(client, _churn_df())
    resp = client.post(f"/api/ml/{sid}/recommend", json={"problem_type": "regression"})
    assert resp.status_code == 400


def test_recommend_unusable_target_returns_400(client):
    df = pd.DataFrame({"Flag": [1] * 60, "X": range(60)})
    sid = _upload_df(client, df)
    resp = client.post(f"/api/ml/{sid}/recommend", json={"target": "Flag"})
    assert resp.status_code == 400


def test_recommend_overlong_objective_returns_400(client):
    sid = _upload_df(client, _churn_df())
    resp = client.post(
        f"/api/ml/{sid}/recommend",
        json={"target": "Churn", "objective": "x" * 501},
    )
    assert resp.status_code == 400


def test_recommend_invalid_session_returns_404(client):
    resp = client.post(
        "/api/ml/00000000-0000-4000-8000-000000000000/recommend",
        json={"target": "Churn"},
    )
    assert resp.status_code == 404


def test_recommend_invalid_uuid_returns_400(client):
    resp = client.post("/api/ml/not-a-uuid/recommend", json={})
    assert resp.status_code == 400


def test_read_endpoints_404_before_recommend(client):
    sid = _upload_df(client, _churn_df())
    for path in ["summary", "models", "roadmap"]:
        assert client.get(f"/api/ml/{sid}/{path}").status_code == 404


def test_summary_endpoint(client):
    sid = _upload_df(client, _churn_df())
    client.post(f"/api/ml/{sid}/recommend", json={"target": "Churn"})
    resp = client.get(f"/api/ml/{sid}/summary")
    assert resp.status_code == 200
    summary = resp.json()["summary"]
    assert summary["available"] is True
    assert summary["problem_type"] == "binary_classification"
    assert len(summary["top_models"]) == 3


def test_models_endpoint(client):
    sid = _upload_df(client, _churn_df())
    client.post(f"/api/ml/{sid}/recommend", json={"target": "Churn"})
    resp = client.get(f"/api/ml/{sid}/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["problem_type"] == "binary_classification"
    assert len(data["models"]) == 7
    assert len(data["model_comparison"]) == 7
    assert data["model_comparison"][0]["rank"] == 1


def test_roadmap_endpoint(client):
    sid = _upload_df(client, _churn_df())
    client.post(f"/api/ml/{sid}/recommend", json={"target": "Churn"})
    resp = client.get(f"/api/ml/{sid}/roadmap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["roadmap"] and data["architecture"]
    assert data["roadmap"][0]["step_number"] == 1
    assert data["architecture"][0]["step"] == 1


def test_recommend_again_replaces_cached_strategy(client):
    sid = _upload_df(client, _churn_df())
    client.post(f"/api/ml/{sid}/recommend", json={"target": "Churn"})
    first = client.get(f"/api/ml/{sid}/models").json()["problem_type"]
    client.post(f"/api/ml/{sid}/recommend", json={"target": "Churn", "problem_type": "regression"})
    second = client.get(f"/api/ml/{sid}/models").json()["problem_type"]
    assert first == "binary_classification"
    assert second == "regression"