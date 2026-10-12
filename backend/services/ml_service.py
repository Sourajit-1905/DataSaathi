import json
import logging
import math
import re
from pathlib import Path
from core.gemini_client import generate_json, render_prompt
import numpy as np
import pandas as pd

from core.config import settings
from core.exceptions import ColumnNotFoundError

logger = logging.getLogger(__name__)

VALID_PROBLEM_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
}

CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}

REGRESSION_HINTS = [
    "how much", "how many", "forecast", "estimate the",
    "predict the price", "predict the amount", "predict the revenue",
    "predict the value", "predict the sales",
]

CLASSIFICATION_HINTS = [
    "which", "classify", "whether", "churn", "will they", "will it",
    "yes or no", "fraud", "default", "categor",
]




# Helpers

def _sanitize(obj):
    """Recursively convert numpy types / NaN / inf to JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _integer_like(s: pd.Series) -> bool:
    s = s.dropna()
    if len(s) == 0:
        return False
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        return bool((s % 1 == 0).all())
    return False


def _feature_columns(df: pd.DataFrame, target: str | None):
    """Feature columns = everything except target, identifiers and constants."""
    from services.profile_service import identifier_detection, constant_column_detection

    ids = [i["column"] for i in identifier_detection(df)]
    consts = [c["column"] for c in constant_column_detection(df)]
    excluded = set(ids) | set(consts) | ({target} if target else set())
    features = [c for c in df.columns if c not in excluded]
    return features, sorted(set(ids) - {target}), sorted(set(consts) - {target})


def _column_roles(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Split feature columns into numeric / categorical / datetime / text."""
    dt_cols: set = set()
    text_cols: set = set()
    if feature_cols:
        sub = df[feature_cols]
        try:
            from services.profile_service import detect_datetime_columns
            dt_cols = {d["column"] for d in detect_datetime_columns(sub)}
        except Exception:
            pass
        try:
            from services.nlp_service import detect_text_columns
            text_cols = {
                t["column"] for t in detect_text_columns(sub) if t["is_text_column"]
            }
        except Exception:
            pass

    numeric = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and c not in dt_cols
    ]
    categorical = [
        c for c in feature_cols
        if c not in numeric and c not in dt_cols and c not in text_cols
    ]
    high_card = [c for c in categorical if df[c].nunique() > 50]

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": sorted(dt_cols),
        "text": sorted(text_cols),
        "high_cardinality": high_card,
    }

# 1. Problem Type Detection

def detect_problem_type(df: pd.DataFrame, target: str | None = None, objective: str = "", override: str | None = None,) -> dict:
    """
    Detect the ML problem type from the target column and stated objective.
    A user override always wins.
    """
    obj = (objective or "").lower()

    if target and target not in df.columns:
        raise ColumnNotFoundError(target)

    if override:
        if override not in VALID_PROBLEM_TYPES:
            raise ValueError(
                f"Invalid problem_type '{override}'. Valid: {sorted(VALID_PROBLEM_TYPES)}"
            )
        if override != "clustering" and not target:
            raise ValueError("A target column is required for supervised problems.")
        return {
            "problem_type": override,
            "target": target if override != "clustering" else None,
            "n_classes": int(df[target].nunique()) if target and override in CLASSIFICATION_TYPES else None,
            "confidence": "high",
            "source": "user_override",
            "reasoning": f"Problem type set manually to {override.replace('_', ' ')}.",
        }

    if not target:
        return {
            "problem_type": "clustering",
            "target": None,
            "n_classes": None,
            "confidence": "medium",
            "source": "no_target",
            "reasoning": "No target column was selected, so the task is treated as "
                         "unsupervised (clustering / segmentation).",
        }

    s = df[target].dropna()
    nunique = int(s.nunique())

    if nunique <= 1:
        return {
            "problem_type": "invalid",
            "target": target,
            "n_classes": nunique,
            "confidence": "high",
            "source": "target_analysis",
            "reasoning": f"'{target}' has {nunique} distinct value(s) — it cannot be "
                         f"used as a prediction target.",
        }

    if nunique == 2:
        return {
            "problem_type": "binary_classification",
            "target": target,
            "n_classes": 2,
            "confidence": "high",
            "source": "target_analysis",
            "reasoning": f"'{target}' has exactly 2 distinct values — a binary "
                         f"classification problem.",
        }

    is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

    if is_numeric:
        if _integer_like(s) and nunique <= 10:
            # Ambiguous: low-cardinality integers. Use the objective as a tiebreaker.
            if any(h in obj for h in REGRESSION_HINTS):
                return {
                    "problem_type": "regression",
                    "target": target,
                    "n_classes": None,
                    "confidence": "medium",
                    "source": "objective",
                    "reasoning": f"'{target}' has {nunique} integer values, but your "
                                 f"objective suggests predicting a quantity — treated as regression.",
                }
            return {
                "problem_type": "multiclass_classification",
                "target": target,
                "n_classes": nunique,
                "confidence": "medium" if not any(h in obj for h in CLASSIFICATION_HINTS) else "high",
                "source": "target_analysis",
                "reasoning": f"'{target}' has {nunique} integer values — treated as "
                             f"multi-class classification. Switch to regression if it is a count or quantity.",
            }
        return {
            "problem_type": "regression",
            "target": target,
            "n_classes": None,
            "confidence": "high",
            "source": "target_analysis",
            "reasoning": f"'{target}' is continuous numerical with {nunique} distinct "
                         f"values — a regression problem.",
        }

    # Non-numeric target
    if nunique <= 20:
        return {
            "problem_type": "multiclass_classification",
            "target": target,
            "n_classes": nunique,
            "confidence": "high",
            "source": "target_analysis",
            "reasoning": f"'{target}' is categorical with {nunique} classes — a "
                         f"multi-class classification problem.",
        }

    return {
        "problem_type": "invalid",
        "target": target,
        "n_classes": nunique,
        "confidence": "high",
        "source": "target_analysis",
        "reasoning": f"'{target}' is categorical with {nunique} distinct values — too "
                     f"many classes to classify directly. Consider grouping categories first.",
    }



# 2. Target Analysis

def analyze_target(df: pd.DataFrame, target: str | None, problem_type: str) -> dict:
    if not target or target not in df.columns:
        return {"target": None}

    s = df[target]
    missing_pct = round(float(s.isna().mean() * 100), 2)
    s = s.dropna()
    info = {"target": target, "missing_pct": missing_pct, "n_valid": int(len(s))}

    if problem_type in CLASSIFICATION_TYPES:
        vc = s.value_counts()
        info["class_distribution"] = [
            {"class": str(k), "count": int(v), "pct": round(float(v / len(s) * 100), 2)}
            for k, v in vc.items()
        ]
    elif problem_type == "regression" and len(s) > 3:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier_pct = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100)
        info.update({
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
            "outlier_pct": round(outlier_pct, 2),
        })
    return info


# 3. class Imbalance

def detect_class_imbalance(df: pd.DataFrame, target: str | None, problem_type: str) -> dict:
    if problem_type not in CLASSIFICATION_TYPES or not target:
        return {"applicable": False, "is_imbalanced": False}

    vc = df[target].dropna().value_counts()
    if len(vc) < 2:
        return {"applicable": False, "is_imbalanced": False}

    majority, minority = int(vc.iloc[0]), int(vc.iloc[-1])
    ratio = round(majority / minority, 2) if minority > 0 else float("inf")

    if ratio <= 1.5: severity = "none"

    elif ratio <= 3: severity = "mild"

    elif ratio <= 10: severity = "moderate"

    else: severity = "severe"

    total = int(vc.sum())

    return {
        "applicable": True,
        "is_imbalanced": ratio > 3,
        "severity": severity,
        "imbalance_ratio": ratio if not math.isinf(ratio) else None,
        "majority_class": str(vc.index[0]),
        "majority_pct": round(majority / total * 100, 2),
        "minority_class": str(vc.index[-1]),
        "minority_pct": round(minority / total * 100, 2),
        "minority_count": minority,
    }


# 4.  Dataset Readiness
def assess_dataset_readiness(df: pd.DataFrame, feature_cols: list[str], roles: dict) -> dict:
    n_rows = len(df)
    n_features = len(feature_cols)
    rows_per_feature = round(n_rows / n_features, 1) if n_features else None
    missing_pct = (
        round(float(df[feature_cols].isna().mean().mean() * 100), 2) if feature_cols else 0.0
    )

    warnings: list[str] = []
    score = 100

    if n_rows < 100:
        score -= 40
        warnings.append(f"Only {n_rows} rows — too few for reliable model training.")

    elif n_rows < 500:
        score -= 20
        warnings.append(f"{n_rows} rows is a small dataset — prefer simple models and cross-validation.")

    elif n_rows < 1000: score -= 10

    if n_features == 0:
        score -= 50
        warnings.append("No usable feature columns remain after excluding the target, identifiers and constants.")
    elif rows_per_feature is not None and rows_per_feature < 10:
        score -= 20
        warnings.append(f"Only {rows_per_feature} rows per feature — high risk of overfitting; consider feature selection.")

    if missing_pct > 20:
        score -= 20
        warnings.append(f"{missing_pct}% of feature values are missing.")

    elif missing_pct > 5:
        score -= 8
        warnings.append(f"{missing_pct}% of feature values are missing — imputation is needed.")

    if roles["high_cardinality"]:
        score -= 5
        warnings.append(f"High-cardinality categorical columns: {roles['high_cardinality'][:5]}.")

    score = max(0, score)
    label = "ready" if score >= 80 else "needs_work" if score >= 50 else "not_ready"

    return {
        "score": score,
        "label": label,
        "n_rows": n_rows,
        "n_features": n_features,
        "rows_per_feature": rows_per_feature,
        "feature_missing_pct": missing_pct,
        "warnings": warnings,
    }

# 5. Leakage Risk

def detect_leakage_risk(
    df: pd.DataFrame, target: str | None, feature_cols: list[str], identifier_cols: list[str]
) -> dict:
    """Flag features that may leak the target (near-perfect correlation, name overlap)."""
    flagged: list[dict] = []
    if not target:
        return {"flagged_features": [], "identifier_columns": identifier_cols, "risk": "none"}

    y = df[target]
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        y = pd.Series(pd.factorize(y)[0], index=df.index).where(y.notna())
    y = y.astype(float)

    num_features = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ][:50]

    for col in num_features:
        pair = pd.concat([df[col].astype(float), y], axis=1).dropna()
        if len(pair) < 10 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            continue
        r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if r is not None and not math.isnan(r) and abs(r) > 0.95:
            flagged.append({
                "column": col,
                "reason": f"Correlation with target is {r:.3f} — possible leakage.",
                "correlation": round(float(r), 4),
            })

    tgt_name = str(target).lower()
    for col in feature_cols:
        if col in [f["column"] for f in flagged]:
            continue
        if tgt_name and tgt_name in str(col).lower():
            flagged.append({
                "column": col,
                "reason": f"Column name contains the target name '{target}' — it may be derived from it.",
                "correlation": None,
            })

    risk = "high" if any(f["correlation"] is not None for f in flagged) else (
        "medium" if flagged else "low"
    )

    return {
        "flagged_features": flagged,
        "identifier_columns": identifier_cols,
        "risk": risk,
    }

# 6. Multicollinearity

def detect_multicollinearity(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    cols = numeric_cols[:30]
    if len(cols) < 2:
        return {"has_multicollinearity": False, "pairs": []}

    corr = df[cols].corr()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if not math.isnan(r) and abs(r) > 0.85:
                pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": round(float(r), 4),
                })
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return {"has_multicollinearity": len(pairs) > 0, "pairs": pairs[:10]}


# Context builder

def _build_context(
    df: pd.DataFrame,
    feature_cols: list[str],
    roles: dict,
    imbalance: dict,
    collinearity: dict,
) -> dict:
    skewed, outlier_heavy = [], []
    for col in roles["numeric"][:50]:
        s = df[col].dropna()
        if len(s) <= 3:
            continue
        if abs(float(s.skew())) > 1:
            skewed.append(col)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0 and float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100) > 5:
            outlier_heavy.append(col)

    return {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "n_numeric": len(roles["numeric"]),
        "n_categorical": len(roles["categorical"]),
        "n_datetime": len(roles["datetime"]),
        "n_text": len(roles["text"]),
        "missing_pct": round(float(df[feature_cols].isna().mean().mean() * 100), 2) if feature_cols else 0.0,
        "skewed_cols": skewed,
        "outlier_cols": outlier_heavy,
        "high_cardinality": roles["high_cardinality"],
        "multicollinear": bool(collinearity.get("has_multicollinearity")),
        "imbalanced": bool(imbalance.get("is_imbalanced")),
    }


# 7. Model Recommendation

def _m(name, base, complexity, speed, interp, strengths, weaknesses, *,
       scaling=False, missing=False, categorical=False, class_weight=False,
       collinear_sensitive=False, dim_sensitive=False, outlier_robust=False,
       scales_poorly=False):
    return {
        "name": name, "base_score": base, "complexity": complexity,
        "training_speed": speed, "interpretability": interp,
        "strengths": strengths, "weaknesses": weaknesses,
        "needs_scaling": scaling, "handles_missing": missing,
        "handles_categorical": categorical, "supports_class_weight": class_weight,
        "collinear_sensitive": collinear_sensitive, "dim_sensitive": dim_sensitive,
        "outlier_robust": outlier_robust, "scales_poorly": scales_poorly,
    }


MODEL_CATALOG = {
    "classification": [
        _m("Logistic Regression", 70, "low", "fast", "high",
           ["Fast baseline", "Highly interpretable coefficients", "Outputs calibrated probabilities"],
           ["Assumes linear decision boundary", "Sensitive to correlated features"],
           scaling=True, class_weight=True, collinear_sensitive=True),
        _m("Random Forest", 78, "medium", "medium", "medium",
           ["Handles non-linear patterns", "Robust to outliers", "Little tuning needed"],
           ["Larger memory footprint", "Less interpretable than linear models"],
           class_weight=True, outlier_robust=True),
        _m("Gradient Boosting (XGBoost / LightGBM)", 82, "high", "medium", "low",
           ["Often the best accuracy on tabular data", "Handles missing values", "Supports categorical features"],
           ["Needs hyperparameter tuning", "Can overfit small datasets"],
           missing=True, categorical=True, class_weight=True, outlier_robust=True),
        _m("Support Vector Machine", 60, "high", "slow", "low",
           ["Effective in high-dimensional spaces", "Flexible kernels"],
           ["Slow on large datasets", "Requires feature scaling"],
           scaling=True, class_weight=True, scales_poorly=True),
        _m("K-Nearest Neighbours", 50, "low", "fast", "medium",
           ["No training phase", "Simple to understand"],
           ["Slow predictions on large data", "Suffers in high dimensions"],
           scaling=True, dim_sensitive=True, scales_poorly=True),
        _m("Decision Tree", 55, "low", "fast", "high",
           ["Easy to visualise", "Handles mixed feature types"],
           ["Prone to overfitting", "Unstable to small data changes"],
           class_weight=True, outlier_robust=True),
        _m("Neural Network (MLP)", 50, "high", "slow", "low",
           ["Can learn complex patterns"],
           ["Needs lots of data", "Hard to interpret", "Requires scaling and tuning"],
           scaling=True, dim_sensitive=False),
    ],
    "regression": [
        _m("Linear / Ridge Regression", 68, "low", "fast", "high",
           ["Fast and interpretable", "Ridge handles correlated features"],
           ["Assumes linear relationships", "Sensitive to outliers"],
           scaling=True, collinear_sensitive=True),
        _m("Lasso / ElasticNet", 66, "low", "fast", "high",
           ["Built-in feature selection", "Works with many features"],
           ["Assumes linear relationships", "Needs scaling"],
           scaling=True),
        _m("Random Forest Regressor", 78, "medium", "medium", "medium",
           ["Captures non-linear relationships", "Robust to outliers"],
           ["Cannot extrapolate beyond training range", "Larger memory footprint"],
           outlier_robust=True),
        _m("Gradient Boosting Regressor (XGBoost / LightGBM)", 82, "high", "medium", "low",
           ["Often the best accuracy on tabular data", "Handles missing values"],
           ["Needs hyperparameter tuning", "Can overfit small datasets"],
           missing=True, categorical=True, outlier_robust=True),
        _m("Support Vector Regression", 55, "high", "slow", "low",
           ["Effective with clear margins", "Flexible kernels"],
           ["Slow on large datasets", "Requires scaling"],
           scaling=True, scales_poorly=True),
        _m("K-Nearest Neighbours Regressor", 48, "low", "fast", "medium",
           ["No training phase"],
           ["Slow predictions on large data", "Suffers in high dimensions"],
           scaling=True, dim_sensitive=True, scales_poorly=True),
        _m("Decision Tree Regressor", 52, "low", "fast", "high",
           ["Easy to visualise"],
           ["Prone to overfitting"],
           outlier_robust=True),
    ],
    "clustering": [
        _m("K-Means", 75, "low", "fast", "high",
           ["Fast and scalable", "Easy to interpret"],
           ["Must choose k in advance", "Assumes spherical clusters", "Sensitive to outliers"],
           scaling=True),
        _m("Hierarchical (Agglomerative)", 62, "medium", "slow", "high",
           ["No need to fix k in advance", "Dendrogram is informative"],
           ["Does not scale to large datasets"],
           scaling=True, scales_poorly=True),
        _m("DBSCAN", 60, "medium", "medium", "medium",
           ["Finds arbitrary-shaped clusters", "Flags noise points"],
           ["Sensitive to eps / min_samples", "Struggles in high dimensions"],
           scaling=True, dim_sensitive=True, outlier_robust=True),
        _m("Gaussian Mixture Model", 60, "medium", "medium", "medium",
           ["Soft cluster assignments", "Handles elliptical clusters"],
           ["Assumes Gaussian components", "Needs scaling"],
           scaling=True),
    ],
}


def _score_model(m: dict, ctx: dict, problem_type: str) -> tuple[int, list[str]]:
    score = m["base_score"]
    reasons: list[str] = []
    n_rows, n_feat = ctx["n_rows"], ctx["n_features"]

    if n_rows < 500:
        if m["complexity"] == "high":
            score -= 15
            reasons.append(f"Only {n_rows} rows — high-complexity models tend to overfit.")
        elif m["complexity"] == "low":
            score += 5
            reasons.append(f"Small dataset ({n_rows} rows) suits simple models.")
    elif n_rows >= 5000 and m["complexity"] == "high" and not m["scales_poorly"]:
        score += 4
        reasons.append(f"{n_rows} rows is enough data for a more complex model to pay off.")

    if n_rows > 50000 and m["scales_poorly"]:
        score -= 15
        reasons.append(f"Does not scale well to {n_rows} rows.")

    if ctx["missing_pct"] > 5:
        if m["handles_missing"]:
            score += 4
            reasons.append("Handles missing values natively.")
        else:
            score -= 3
            reasons.append("Requires imputation because features contain missing values.")

    if ctx["n_categorical"] > 0 and m["handles_categorical"]:
        score += 3
        reasons.append("Supports categorical features natively.")

    if ctx["multicollinear"] and m["collinear_sensitive"]:
        score -= 8
        reasons.append("Correlated features reduce reliability of its coefficients.")

    if n_feat > max(1, n_rows / 10) and m["dim_sensitive"]:
        score -= 8
        reasons.append("High feature-to-row ratio hurts distance-based methods.")

    if ctx["outlier_cols"] or ctx["skewed_cols"]:
        if m["outlier_robust"]:
            score += 3
            reasons.append("Robust to outliers and skewed features.")
        elif not m["needs_scaling"]:
            pass
        else:
            score -= 2
            reasons.append("Outliers and skew can distort it unless features are transformed.")

    if problem_type in CLASSIFICATION_TYPES and ctx["imbalanced"]:
        if m["supports_class_weight"]:
            score += 4
            reasons.append("Supports class weights for the imbalanced target.")
        else:
            score -= 4
            reasons.append("No built-in class weighting for the imbalanced target.")

    return max(0, min(100, int(score))), reasons


def recommend_models(problem_type: str, ctx: dict) -> list[dict]:
    key = "classification" if problem_type in CLASSIFICATION_TYPES else problem_type
    catalog = MODEL_CATALOG.get(key, [])

    scored = []
    for m in catalog:
        score, reasons = _score_model(m, ctx, problem_type)
        label = (
            "excellent" if score >= 80 else "good" if score >= 65
            else "fair" if score >= 50 else "poor"
        )
        scored.append({
            "name": m["name"],
            "score": score,
            "suitability": label,
            "dots": max(1, min(5, round(score / 20))),
            "complexity": m["complexity"],
            "training_speed": m["training_speed"],
            "interpretability": m["interpretability"],
            "strengths": m["strengths"],
            "weaknesses": m["weaknesses"],
            "needs_scaling": m["needs_scaling"],
            "handles_missing": m["handles_missing"],
            "handles_categorical": m["handles_categorical"],
            "reasons": reasons,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank
        item["recommended"] = rank <= 3
    return scored


# 8. Model Comparison Table

def build_model_comparison(models: list[dict]) -> list[dict]:
    yn = lambda b: "Yes" if b else "No"
    return [
        {
            "rank": m["rank"],
            "model": m["name"],
            "score": m["score"],
            "suitability": m["suitability"],
            "complexity": m["complexity"],
            "training_speed": m["training_speed"],
            "interpretability": m["interpretability"],
            "needs_scaling": yn(m["needs_scaling"]),
            "handles_missing": yn(m["handles_missing"]),
            "handles_categorical": yn(m["handles_categorical"]),
        }
        for m in models
    ]


# 9. Preprocessing Recommendations

def recommend_preprocessing(
    df: pd.DataFrame,
    roles: dict,
    ctx: dict,
    models: list[dict],
    identifier_cols: list[str],
    constant_cols: list[str],
    collinearity: dict,
) -> list[dict]:
    steps: list[dict] = []

    drop_cols = identifier_cols + constant_cols
    if drop_cols:
        steps.append({
            "technique": "Drop identifier and constant columns",
            "priority": "high",
            "applies_to": drop_cols[:10],
            "trigger": f"{len(drop_cols)} column(s) are unique identifiers or constant.",
            "reason": "They carry no predictive signal and can cause overfitting.",
        })

    if ctx["missing_pct"] > 0:
        cols_missing = [c for c in df.columns if df[c].isna().any()][:10]
        steps.append({
            "technique": "Impute missing values",
            "priority": "high" if ctx["missing_pct"] > 5 else "medium",
            "applies_to": cols_missing,
            "trigger": f"{ctx['missing_pct']}% of feature values are missing.",
            "reason": "Most models cannot accept missing values; use median for numeric and mode for categorical, fit on training data only.",
        })

    if roles["categorical"]:
        low = [c for c in roles["categorical"] if c not in roles["high_cardinality"]]
        if low:
            steps.append({
                "technique": "One-hot encode low-cardinality categoricals",
                "priority": "high",
                "applies_to": low[:10],
                "trigger": f"{len(low)} categorical column(s) with 50 or fewer categories.",
                "reason": "Converts categories into numeric columns that models can use.",
            })
        if roles["high_cardinality"]:
            steps.append({
                "technique": "Target or frequency encode high-cardinality categoricals",
                "priority": "high",
                "applies_to": roles["high_cardinality"][:10],
                "trigger": "Categorical column(s) with more than 50 categories.",
                "reason": "One-hot encoding would create too many columns; use cross-validated target encoding to avoid leakage.",
            })

    top3 = models[:3]
    ranges_differ = False
    if len(roles["numeric"]) >= 2:
        stds = df[roles["numeric"][:50]].std().replace(0, np.nan).dropna()
        ranges_differ = len(stds) >= 2 and float(stds.max() / stds.min()) > 10
    if any(m["needs_scaling"] for m in top3) or ranges_differ:
        steps.append({
            "technique": "Scale numeric features (standardise)",
            "priority": "high" if any(m["needs_scaling"] for m in top3) else "medium",
            "applies_to": roles["numeric"][:10],
            "trigger": "Top-ranked models are scale-sensitive" if any(m["needs_scaling"] for m in top3)
                       else "Numeric features have very different scales.",
            "reason": "Prevents large-valued features from dominating distance- and gradient-based models.",
        })

    if ctx["skewed_cols"]:
        steps.append({
            "technique": "Apply log / power transform to skewed features",
            "priority": "medium",
            "applies_to": ctx["skewed_cols"][:10],
            "trigger": f"{len(ctx['skewed_cols'])} numeric column(s) with |skewness| > 1.",
            "reason": "Makes distributions more symmetric, which helps linear and distance-based models.",
        })

    if ctx["outlier_cols"]:
        steps.append({
            "technique": "Cap or robust-scale outliers",
            "priority": "medium",
            "applies_to": ctx["outlier_cols"][:10],
            "trigger": f"{len(ctx['outlier_cols'])} numeric column(s) with more than 5% IQR outliers.",
            "reason": "Extreme values can distort linear models; tree ensembles are largely unaffected.",
        })

    if collinearity.get("has_multicollinearity"):
        steps.append({
            "technique": "Remove or combine highly correlated features",
            "priority": "medium",
            "applies_to": sorted({p["col2"] for p in collinearity["pairs"]})[:10],
            "trigger": f"{len(collinearity['pairs'])} feature pair(s) with |r| > 0.85.",
            "reason": "Redundant features inflate coefficient variance in linear models and add noise elsewhere.",
        })

    return steps


# 10. Feature Engineering Recommendations

def recommend_feature_engineering(
    roles: dict, ctx: dict, problem_type: str, models: list[dict]
) -> list[dict]:
    recs: list[dict] = []

    if roles["datetime"]:
        recs.append({
            "technique": "Extract date parts (year, month, day-of-week, quarter)",
            "applies_to": roles["datetime"],
            "trigger": f"{len(roles['datetime'])} datetime column(s) detected.",
            "reason": "Raw timestamps are not usable by most models; calendar parts expose seasonality.",
        })
        if problem_type == "regression":
            recs.append({
                "technique": "Add lag and rolling-mean features",
                "applies_to": roles["datetime"],
                "trigger": "Datetime column present in a regression task.",
                "reason": "Past values often predict the next value; split chronologically to avoid leakage.",
            })

    if roles["text"]:
        recs.append({
            "technique": "Vectorise text (TF-IDF or sentence embeddings)",
            "applies_to": roles["text"],
            "trigger": f"{len(roles['text'])} free-text column(s) detected.",
            "reason": "Models need numeric input; TF-IDF is a strong, fast baseline.",
        })

    linear_in_top = any("Regression" in m["name"] or "Lasso" in m["name"] for m in models[:3])
    if linear_in_top and ctx["n_numeric"] >= 2:
        recs.append({
            "technique": "Add interaction and polynomial terms",
            "applies_to": [],
            "trigger": "A linear model is among the top recommendations.",
            "reason": "Lets a linear model capture simple non-linear and combined effects.",
        })

    if roles["high_cardinality"]:
        recs.append({
            "technique": "Group rare categories into 'Other'",
            "applies_to": roles["high_cardinality"][:10],
            "trigger": "High-cardinality categorical column(s).",
            "reason": "Reduces sparsity and noise from categories with very few examples.",
        })

    if ctx["n_features"] > 50 or ctx["multicollinear"] and ctx["n_features"] > 20:
        recs.append({
            "technique": "Dimensionality reduction (PCA) or feature selection",
            "applies_to": [],
            "trigger": f"{ctx['n_features']} features" + (" with correlated pairs." if ctx["multicollinear"] else "."),
            "reason": "Reduces overfitting risk and training time.",
        })

    return recs


# 11. Imbalance Handling

def recommend_imbalance_handling(imbalance: dict, n_rows: int) -> list[dict]:
    if not imbalance.get("is_imbalanced"):
        return []

    sev = imbalance["severity"]
    minority = imbalance["minority_count"]
    recs = [
        {
            "technique": "Use class weights",
            "priority": "high",
            "trigger": f"{sev.capitalize()} imbalance — minority class is {imbalance['minority_pct']}% of rows.",
            "reason": "Penalises mistakes on the minority class more heavily without altering the data.",
        },
        {
            "technique": "Stratified train/test split and cross-validation",
            "priority": "high",
            "trigger": "Imbalanced target.",
            "reason": "Keeps the class ratio the same in every fold so the minority class is always represented.",
        },
        {
            "technique": "Tune the decision threshold",
            "priority": "medium",
            "trigger": "Imbalanced target.",
            "reason": "The default 0.5 threshold is rarely optimal; pick one using a precision-recall curve.",
        },
    ]
    if minority >= 6:
        recs.append({
            "technique": "SMOTE oversampling (training data only)",
            "priority": "medium",
            "trigger": f"Minority class has {minority} examples.",
            "reason": "Synthesises new minority examples; apply inside cross-validation folds to avoid leakage.",
        })
    if sev == "severe" and n_rows > 50000:
        recs.append({
            "technique": "Random undersampling of the majority class",
            "priority": "low",
            "trigger": f"Severe imbalance with {n_rows} rows.",
            "reason": "Large datasets can afford to drop majority rows, which also speeds up training.",
        })
    return recs


# 12. Split Strategy

def recommend_split_strategy(
    problem_type: str, n_rows: int, imbalanced: bool, has_datetime: bool
) -> dict:
    if problem_type == "clustering":
        return {
            "strategy": "none",
            "description": "Clustering is unsupervised — no train/test split is needed. Validate with internal metrics instead.",
        }

    if n_rows >= 100000:
        ratios = {"train": 90, "validation": 0, "test": 10}
    elif n_rows >= 10000:
        ratios = {"train": 70, "validation": 15, "test": 15}
    else:
        ratios = {"train": 80, "validation": 0, "test": 20}

    if has_datetime and problem_type == "regression":
        method, note = "chronological", "Split by time (oldest to train, newest to test) to avoid look-ahead leakage."
    elif problem_type in CLASSIFICATION_TYPES:
        method = "stratified"
        note = ("Stratify on the target — essential because the target is imbalanced."
                if imbalanced else "Stratify on the target to keep class proportions equal.")
    else:
        method, note = "random", "A random shuffle split is appropriate."

    return {
        "strategy": method,
        "ratios_pct": ratios,
        "random_state": 42,
        "description": note,
        "caution": "Fit all preprocessing (imputation, scaling, encoding) on the training set only." +
                   (" Small dataset — rely on cross-validation rather than a single hold-out." if n_rows < 1000 else ""),
    }


# 13. Cross-Validation

def recommend_cross_validation(
    problem_type: str, n_rows: int, imbalanced: bool, has_datetime: bool
) -> dict:
    if problem_type == "clustering":
        return {
            "method": "none",
            "folds": None,
            "description": "Compare several values of k using silhouette score; no CV is needed.",
        }

    if has_datetime and problem_type == "regression":
        return {
            "method": "TimeSeriesSplit",
            "folds": 5,
            "description": "Expanding-window validation preserves time order.",
        }

    classification = problem_type in CLASSIFICATION_TYPES
    prefix = "Stratified " if classification else ""

    if n_rows < 1000:
        return {
            "method": f"Repeated {prefix}K-Fold",
            "folds": 5,
            "repeats": 3,
            "description": "Small dataset — repeat 5-fold CV three times for a more stable estimate.",
        }
    if n_rows <= 50000:
        return {
            "method": f"{prefix}K-Fold",
            "folds": 5,
            "description": "Standard 5-fold cross-validation.",
        }
    return {
        "method": f"{prefix}K-Fold",
        "folds": 3,
        "description": f"{n_rows} rows — 3 folds keeps training time manageable.",
    }


# 14. Evaluation Metrics

def recommend_metrics(problem_type: str, imbalanced: bool, target_info: dict) -> dict:
    if problem_type == "binary_classification":
        if imbalanced:
            return {
                "primary": "PR-AUC",
                "secondary": ["F1", "Recall", "Precision", "Balanced Accuracy", "MCC"],
                "avoid": ["Accuracy"],
                "reason": "With an imbalanced target, accuracy is misleading — a model that always predicts the majority class scores highly.",
            }
        return {
            "primary": "ROC-AUC",
            "secondary": ["F1", "Accuracy", "Precision", "Recall"],
            "avoid": [],
            "reason": "Balanced classes make ROC-AUC and accuracy reliable.",
        }

    if problem_type == "multiclass_classification":
        return {
            "primary": "Macro F1",
            "secondary": ["Accuracy", "Per-class Recall", "Confusion Matrix"],
            "avoid": ["Accuracy alone"] if imbalanced else [],
            "reason": "Macro F1 weights every class equally, so small classes are not ignored.",
        }

    if problem_type == "regression":
        heavy_tail = (target_info.get("skewness") is not None and abs(target_info["skewness"]) > 1) or \
                     (target_info.get("outlier_pct") or 0) > 5
        if heavy_tail:
            return {
                "primary": "MAE",
                "secondary": ["RMSE", "R²", "RMSLE"],
                "avoid": [],
                "reason": "The target is skewed or has outliers, so MAE is less dominated by extreme errors than RMSE.",
            }
        return {
            "primary": "RMSE",
            "secondary": ["MAE", "R²"],
            "avoid": [],
            "reason": "A well-behaved target makes RMSE and R² dependable.",
        }

    return {
        "primary": "Silhouette Score",
        "secondary": ["Davies-Bouldin Index", "Calinski-Harabasz Index", "Inertia (elbow plot)"],
        "avoid": [],
        "reason": "No labels exist, so use internal cluster-quality metrics and sanity-check clusters manually.",
    }


# 15. Pipeline Architecture

def build_architecture(
    problem_type: str,
    models: list[dict],
    preprocessing: list[dict],
    feature_eng: list[dict],
    imbalance_handling: list[dict],
    cv: dict,
) -> list[dict]:
    steps = [{"step": 1, "name": "Load cleaned data", "description": "Use the working dataset produced by the Clean and Transform stages."}]

    if feature_eng:
        steps.append({"step": len(steps) + 1, "name": "Feature engineering",
                      "description": "; ".join(f["technique"] for f in feature_eng[:3])})
    if preprocessing:
        steps.append({"step": len(steps) + 1, "name": "Preprocessing (ColumnTransformer)",
                      "description": "; ".join(p["technique"] for p in preprocessing[:4])})
    if problem_type != "clustering":
        steps.append({"step": len(steps) + 1, "name": "Train/test split",
                      "description": "Split before fitting any transformer to prevent leakage."})
    if imbalance_handling:
        steps.append({"step": len(steps) + 1, "name": "Imbalance handling",
                      "description": imbalance_handling[0]["technique"] + " (training folds only)"})

    best = models[0]["name"] if models else "Model"
    steps.append({"step": len(steps) + 1, "name": f"Model: {best}", "description": "Start with the top-ranked model; compare against the runner-ups."})
    steps.append({"step": len(steps) + 1, "name": "Evaluation",
                  "description": f"Validate with {cv.get('method', 'appropriate validation')}"
                                 + (f" ({cv['folds']} folds)" if cv.get("folds") else "") + "."})
    steps.append({"step": len(steps) + 1, "name": "Hyperparameter tuning", "description": "Use randomised or Bayesian search on the best one or two models."})
    return steps


# 16. Roadmap

def build_roadmap(
    problem_type: str,
    models: list[dict],
    readiness: dict,
    leakage: dict,
    preprocessing: list[dict],
    imbalance_handling: list[dict],
    metrics: dict,
) -> list[dict]:
    pt = problem_type.replace("_", " ")
    top = models[0]["name"] if models else "the top model"
    second = models[1]["name"] if len(models) > 1 else None

    road = []

    def add(title, description):
        road.append({"step_number": len(road) + 1, "title": title, "description": description})

    add("Confirm the problem framing",
        f"Treat this as {pt}. Check the target definition with domain owners before modelling.")

    if readiness["warnings"] or leakage["flagged_features"]:
        issues = readiness["warnings"][:2] + [
            f"Possible leakage in {', '.join(f['column'] for f in leakage['flagged_features'][:3])}."
        ] if leakage["flagged_features"] else readiness["warnings"][:2]
        add("Resolve data-readiness concerns", " ".join(issues))

    if preprocessing:
        add("Build the preprocessing pipeline",
            "Implement: " + "; ".join(p["technique"] for p in preprocessing[:4]) + ".")

    if problem_type != "clustering":
        add("Create a validation split",
            "Hold out a test set first and keep it untouched until the final evaluation.")

    if imbalance_handling:
        add("Address class imbalance", imbalance_handling[0]["technique"] + ", then re-check minority-class recall.")

    add("Train a baseline",
        f"Start with {top}" + (f" and compare against {second}." if second else "."))
    add("Evaluate properly", f"Use {metrics['primary']} as the primary metric. {metrics['reason']}")
    add("Tune and iterate", "Tune the best one or two models; inspect errors and feature importance.")
    add("Validate before deployment",
        "Check performance on the untouched test set, review fairness and stability, and document assumptions.")
    return road


# 17. Reasoning (Gemini + rule-based fallback)

def _rule_based_reasoning(problem: dict, models: list[dict], objective: str, ctx: dict) -> dict:
    pt = problem["problem_type"].replace("_", " ")
    goal = f" For your goal ('{objective.strip()}'), " if objective and objective.strip() else " "
    top = models[0] if models else None

    summary = (
        f"This is a {pt} problem based on your selected target.{goal}"
        + (f"{top['name']} is the strongest starting point (suitability {top['score']}/100). " if top else "")
        + f"The dataset has {ctx['n_rows']} rows and {ctx['n_features']} usable features."
    )

    model_reasons = {
        m["name"]: (" ".join(m["reasons"][:2]) if m["reasons"]
                    else f"A reasonable {m['complexity']}-complexity option for {pt}.")
        for m in models[:5]
    }

    return {"summary": summary, "model_reasons": model_reasons, "source": "rule_based"}

async def generate_reasoning(problem: dict, models: list[dict], objective: str, ctx: dict,
                             imbalance: dict) -> dict:
    fallback = _rule_based_reasoning(problem, models, objective, ctx)

    try:
        context = {
            "problem_type": problem["problem_type"],
            "target": problem.get("target"),
            "n_classes": problem.get("n_classes"),
            "dataset": {k: ctx[k] for k in ["n_rows", "n_features", "n_numeric", "n_categorical", "missing_pct"]},
            "has_skewed_features": bool(ctx["skewed_cols"]),
            "has_outliers": bool(ctx["outlier_cols"]),
            "multicollinearity": ctx["multicollinear"],
            "imbalance": {
                "is_imbalanced": imbalance.get("is_imbalanced", False),
                "severity": imbalance.get("severity"),
                "minority_pct": imbalance.get("minority_pct"),
            },
            "top_models": [
                {"name": m["name"], "score": m["score"], "reasons": m["reasons"][:3]}
                for m in models[:5]
            ],
        }

        prompt = render_prompt(
            "ml_strategy.txt",
            ML_CONTEXT=json.dumps(_sanitize(context), indent=2, default=str),
            USER_OBJECTIVE=(objective or "Not stated").strip()[:500],
        )
        if prompt is None:
            return fallback

        parsed = await generate_json(prompt)
        if not parsed:
            return fallback

        summary = parsed.get("summary", "")
        model_reasons = parsed.get("model_reasons", {})
        if isinstance(summary, str) and len(summary) > 20 and isinstance(model_reasons, dict):
            # Keep rule-based reasons for any model Gemini skipped
            merged = {**fallback["model_reasons"], **{str(k): str(v) for k, v in model_reasons.items()}}
            return {"summary": summary, "model_reasons": merged, "source": "gemini"}

    except Exception as e:
        logger.warning(f"Gemini ML reasoning failed: {e}")

    return fallback

# 18. Master: build the full strategy

async def build_ml_strategy(df: pd.DataFrame, target: str | None = None, objective: str = "", problem_type_override: str | None = None,) -> dict:
    """
    Run every analysis and return the complete ML strategy.
    """
    problem = detect_problem_type(df, target, objective, problem_type_override)

    if problem["problem_type"] == "invalid":
        return _sanitize({
            "problem": problem,
            "valid": False,
            "message": problem["reasoning"],
        })

    pt = problem["problem_type"]
    tgt = problem.get("target")

    feature_cols, identifier_cols, constant_cols = _feature_columns(df, tgt)
    roles = _column_roles(df, feature_cols)

    target_info = analyze_target(df, tgt, pt)
    imbalance = detect_class_imbalance(df, tgt, pt)
    readiness = assess_dataset_readiness(df, feature_cols, roles)
    leakage = detect_leakage_risk(df, tgt, feature_cols, identifier_cols)
    collinearity = detect_multicollinearity(df, roles["numeric"])

    ctx = _build_context(df, feature_cols, roles, imbalance, collinearity)
    models = recommend_models(pt, ctx)
    comparison = build_model_comparison(models)

    preprocessing = recommend_preprocessing(
        df, roles, ctx, models, identifier_cols, constant_cols, collinearity
    )

    feature_eng = recommend_feature_engineering(roles, ctx, pt, models)
    imbalance_handling = recommend_imbalance_handling(imbalance, ctx["n_rows"])

    has_dt = bool(roles["datetime"])
    split = recommend_split_strategy(pt, ctx["n_rows"], ctx["imbalanced"], has_dt)
    cv = recommend_cross_validation(pt, ctx["n_rows"], ctx["imbalanced"], has_dt)
    metrics = recommend_metrics(pt, ctx["imbalanced"], target_info)

    architecture = build_architecture(pt, models, preprocessing, feature_eng, imbalance_handling, cv)
    roadmap = build_roadmap(pt, models, readiness, leakage, preprocessing, imbalance_handling, metrics)

    reasoning = await generate_reasoning(problem, models, objective, ctx, imbalance)
    for m in models:
        m["reasoning"] = reasoning["model_reasons"].get(m["name"])

    return _sanitize({
        "valid": True,
        "objective": (objective or "").strip(),
        "problem": problem,
        "target_analysis": target_info,
        "feature_columns": feature_cols,
        "excluded_columns": {"identifiers": identifier_cols, "constants": constant_cols},
        "column_roles": roles,
        "readiness": readiness,
        "leakage": leakage,
        "multicollinearity": collinearity,
        "imbalance": imbalance,
        "models": models,
        "model_comparison": comparison,
        "preprocessing": preprocessing,
        "feature_engineering": feature_eng,
        "imbalance_handling": imbalance_handling,
        "split_strategy": split,
        "cross_validation": cv,
        "metrics": metrics,
        "architecture": architecture,
        "roadmap": roadmap,
        "summary": reasoning["summary"],
        "reasoning_source": reasoning["source"],
    })


# 19. Compact summary for the report

def ml_summary(strategy: dict) -> dict:
    if not strategy or not strategy.get("valid"):
        return {"available": False}

    return {
        "available": True,
        "problem_type": strategy["problem"]["problem_type"],
        "target": strategy["problem"].get("target"),
        "objective": strategy.get("objective"),
        "summary": strategy.get("summary"),
        "readiness": {
            "score": strategy["readiness"]["score"],
            "label": strategy["readiness"]["label"],
            "warnings": strategy["readiness"]["warnings"],
        },
        "imbalance": {
            "is_imbalanced": strategy["imbalance"].get("is_imbalanced", False),
            "severity": strategy["imbalance"].get("severity"),
        },
        "top_models": [
            {"name": m["name"], "score": m["score"], "suitability": m["suitability"]}
            for m in strategy["models"][:3]
        ],
        "primary_metric": strategy["metrics"]["primary"],
        "validation": strategy["cross_validation"]["method"],
        "roadmap": [r["title"] for r in strategy["roadmap"]],
    }




















