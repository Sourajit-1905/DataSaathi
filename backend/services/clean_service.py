"""
Clean Service
-------------
Implements 9 cleaning operations:

  C1 — Missing values (numerical)
  C2 — Missing values (categorical)
  C3 — Duplicate rows
  C4 — Outlier treatment
  C5 — Type conversion (object → numeric/datetime)
  C6 — String standardisation (trim, lowercase, title case)
  C7 — Constant / near-constant column removal
  C8 — Identifier column removal
  C9 — Negative value treatment

For each issue:
  - detect_*()     → finds affected rows/columns
  - recommend_*()  → suggests a strategy algorithmically
  - preview_*()    → returns before/after sample WITHOUT saving
  - apply_*()      → applies to working CSV (snapshot taken first)

Gemini adds a plain English "why" explanation to every recommendation.
Fallback: rule-based explanation if Gemini is unavailable.
"""
import json
from core.gemini_client import generate_json, render_prompt
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from core.cache import invalidate_profile
from core.config import settings
from core.exceptions import ColumnNotFoundError

logger = logging.getLogger(__name__)

# Gemini explanation helper
async def _get_gemini_explanation(issue_type: str, column: str, strategy: str, stats: dict) -> str | None:
    """
    Ask Gemini to explain why this strategy is recommended for this issue.
    Returns a plain English string or None on failure.
    """
    try:
        payload = {
            "issue_type": issue_type,
            "column": column,
            "strategy": strategy,
            "stats": stats,
        }
        prompt = render_prompt(
            "clean_recommend.txt",
            ISSUE_PAYLOAD=json.dumps(payload, indent=2, default=str),
        )
        if prompt is None:
            return None

        parsed = await generate_json(prompt)
        if not parsed:
            return None

        explanation = parsed.get("explanation", "")
        if isinstance(explanation, str) and len(explanation) > 10:
            return explanation.strip()

    except Exception as e:
        logger.warning(f"Gemini explanation failed for {issue_type}/{column}: {e}")

    return None

def _rule_based_explanation(issue_type: str, column: str, strategy: str, stats: dict) -> str:
    """Fallback plain English explanation when Gemini is unavailable."""
    explanations = {
        "missing_numerical": {
            "median": f"'{column}' has {stats.get('missing_pct', '?')}% missing values and is skewed (skewness={stats.get('skewness', '?')}), so median imputation is safer than mean as it is not affected by outliers.",
            "mean": f"'{column}' has {stats.get('missing_pct', '?')}% missing values and is symmetric, so mean imputation is appropriate.",
            "constant": f"'{column}' has {stats.get('missing_pct', '?')}% missing values — filling with a constant preserves the column structure.",
            "drop_rows": f"'{column}' has {stats.get('missing_pct', '?')}% missing values — dropping affected rows is appropriate when missing data is sparse.",
            "drop_column": f"'{column}' has {stats.get('missing_pct', '?')}% missing values — too many to impute reliably, so dropping the column is recommended.",
        },
        "missing_categorical": {
            "mode": f"'{column}' has {stats.get('missing_pct', '?')}% missing values — filling with the most frequent value ('{stats.get('mode', '?')}') is a reasonable default.",
            "constant": f"'{column}' has {stats.get('missing_pct', '?')}% missing values — filling with 'Unknown' preserves the rows without introducing bias.",
            "drop_rows": f"'{column}' has {stats.get('missing_pct', '?')}% missing values — dropping affected rows keeps the dataset clean.",
            "drop_column": f"'{column}' has {stats.get('missing_pct', '?')}% missing values — too many to impute reliably.",
        },
        "outliers": {
            "iqr_cap": f"'{column}' has {stats.get('outlier_count', '?')} outliers — capping at IQR bounds preserves the rows while limiting extreme influence.",
            "zscore_cap": f"'{column}' has {stats.get('outlier_count', '?')} outliers — z-score capping limits values beyond 3 standard deviations.",
            "drop_rows": f"'{column}' has {stats.get('outlier_count', '?')} outlier rows — dropping them removes extreme noise from the dataset.",
            "keep": f"'{column}' outliers appear to be legitimate values — keeping them is the safest choice.",
        },
        "duplicates": {
            "drop_duplicates": f"The dataset has {stats.get('duplicate_count', '?')} duplicate rows ({stats.get('duplicate_pct', '?')}%) — removing them prevents double-counting in analysis.",
        },
        "type_conversion": {
            "to_numeric": f"'{column}' is stored as text but contains numeric values — converting it enables mathematical operations.",
            "to_datetime": f"'{column}' is stored as text but contains dates — converting it enables time-series analysis.",
        },
        "string_standardisation": {
            "trim_whitespace": f"'{column}' has leading/trailing whitespace — trimming ensures consistent matching.",
            "lowercase": f"'{column}' has mixed case — converting to lowercase prevents duplicates like 'Male' and 'male'.",
            "titlecase": f"'{column}' has mixed case — title case standardises the values consistently.",
        },
        "constant_columns": {
            "drop_column": f"'{column}' has the same value in {stats.get('frequency_pct', '?')}% of rows — it carries no information and should be dropped.",
        },
        "identifier_columns": {
            "drop_column": f"'{column}' appears to be a unique identifier — it should be excluded from ML models as it adds no predictive value.",
        },
        "negative_values": {
            "set_zero": f"'{column}' has {stats.get('negative_count', '?')} negative values which are not valid — replacing with zero is the safest correction.",
            "drop_rows": f"'{column}' has {stats.get('negative_count', '?')} negative values — dropping the affected rows removes invalid data.",
            "abs_value": f"'{column}' has {stats.get('negative_count', '?')} negative values — taking the absolute value assumes the sign was entered incorrectly.",
        },
    }
    return explanations.get(issue_type, {}).get(strategy, f"Strategy '{strategy}' applied to '{column}'.")

# 1. Handling Missing Values

def detect_missing_numerical(df: pd.DataFrame) -> list[dict]:
    results = []

    for col in df.select_dtypes(include=[np.number]).columns:
        missing_count = int(df[col].isna().sum())

        if missing_count == 0:
            continue

        missing_pct = round(missing_count / len(df) * 100, 2)
        skewness = round(float(df[col].dropna().skew()), 4) if len(df[col].dropna()) > 3 else 0.0
        
        results.append({
            "column": col,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "skewness": skewness,
        })

    return results

async def recommend_missing_numerical(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    s = df[column].dropna()

    missing_count = int(df[column].isna().sum())
    missing_pct = round(missing_count / len(df) * 100, 2)
    skewness = round(float(s.skew()), 4) if len(s) > 3 else 0.0

    if missing_pct > 50: strategy = "drop_column"
    elif missing_pct > 30: strategy = "median" if abs(skewness) > 0.5 else "mean"
    elif abs(skewness) > 0.5 : strategy = "median"
    else: strategy = "mean"

    stats = {"missing_pct": missing_pct,
            "skewness": skewness,
            "missing_count": missing_count}
    
    explanation = await _get_gemini_explanation("missing_numerical", column, strategy, stats)

    if not explanation:
        explanation = _rule_based_explanation("missing_numerical", column, strategy, stats)

    return {
        "column": column,
        "issue_type": "missing_numerical",
        "recommended_strategy": strategy,
        "available_strategies": ["mean", "median", "constant", "drop_rows", "drop_column"],
        "stats": stats,
        "explanation": explanation,
    }

def _sanitize_for_json(records: list[dict]) -> list[dict]:
    """Replace NaN/Inf float values with None so the result is JSON serializable."""
    import math
    sanitized = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
        sanitized.append(clean_row)
    return sanitized


def preview_missing_numerical(df: pd.DataFrame, column: str, strategy: str, fill_value: float = 0.0) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    affected = df[df[column].isna()].head(5).copy()
    before_records = affected[[column]].to_dict(orient="records")

    if strategy in ("drop_rows", "drop_column"):
        return {
            "column": column,
            "strategy": strategy,
            "affected_rows": int(df[column].isna().sum()),
            "before": [],
            "after": [],
            "note": (
                f"This will drop {int(df[column].isna().sum())} rows."
                if strategy == "drop_rows"
                else "This will remove the column entirely."
            ),
        }

    s = df[column].dropna()
    if strategy == "mean":
        val = s.mean()
    elif strategy == "median":
        val = s.median()
    elif strategy == "constant":
        val = fill_value
    else:
        val = s.median()

    after_records = [{column: round(float(val), 4)} for _ in before_records]

    return {
        "column": column,
        "strategy": strategy,
        "fill_value": round(float(val), 4),
        "affected_rows": int(df[column].isna().sum()),
        "before": _sanitize_for_json(before_records),
        "after": _sanitize_for_json(after_records),
    }


def apply_missing_numerical(df: pd.DataFrame, column: str, strategy: str, fill_value: float = 0.0) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    df = df.copy()
    s = df[column].dropna()

    if strategy == "mean": df[column] = df[column].fillna(s.mean())

    elif strategy == "median": df[column] = df[column].fillna(s.median())

    elif strategy == "constant": df[column] = df[column].fillna(fill_value)
    
    elif strategy == "drop_rows" : df = df.dropna(subset=[column])
    
    elif strategy == "drop_column" : df = df.drop(columns=[column])

    return df


# 2.Missing Values (Categorical)

def detect_missing_categorical(df: pd.DataFrame) ->list[dict]:
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        missing_count = int(df[col].isna().sum())

        if missing_count == 0: continue

        missing_pct = round(missing_count/len(df)*100, 2)
        mode_val = str(df[col].mode().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"

        results.append({
            "column": col,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "mode": mode_val,
        })

    return results

async def recommend_missing_categorical(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    missing_count = int(df[column].isna().sum())
    missing_pct = round(missing_count / len(df) * 100, 2)
    mode_val = str(df[column].mode().iloc[0]) if len(df[column].dropna()) > 0 else "Unknown"

    if missing_pct > 50: strategy = "drop_column"

    elif missing_pct > 20:
        strategy = "constant"

    else:
        strategy = "mode"

    stats = {"missing_pct": missing_pct, "missing_count": missing_count, "mode": mode_val}
    
    explanation = await _get_gemini_explanation("missing_categorical", column, strategy, stats)
    
    if not explanation:
        explanation = _rule_based_explanation("missing_categorical", column, strategy, stats)

    return {
        "column": column,
        "issue_type": "missing_categorical",
        "recommended_strategy": strategy,
        "available_strategies": ["mode", "constant", "drop_rows", "drop_column"],
        "stats": stats,
        "explanation": explanation,
    }

def apply_missing_categorical(df: pd.DataFrame, column: str, strategy: str, fill_value: str = "Unknown") -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    df = df.copy()
    if strategy == "mode":
        mode_val = df[column].mode().iloc[0] if len(df[column].dropna()) > 0 else fill_value
        df[column] = df[column].fillna(mode_val)
    elif strategy == "constant":
        df[column] = df[column].fillna(fill_value)
    elif strategy == "drop_rows":
        df = df.dropna(subset=[column])
    elif strategy == "drop_column":
        df = df.drop(columns=[column])

    return df


# 3.Duplicate Rows

def detect_duplicates(df: pd.DataFrame) ->dict:
    dup_mask = df.duplicated()
    count = int(dup_mask.sum())
    pct = round(count/len(df)*100, 2) if len(df) > 0 else 0.0

    sample = df[dup_mask].head(5).where(pd.notnull(df[dup_mask].head(5)), None).to_dict(orient="records")

    return {
        "duplicate_count": count,
        "duplicate_pct": pct,
        "sample_rows": sample
    }

async def recommend_duplicates(df: pd.DataFrame) -> dict:
    info = detect_duplicates(df)
    strategy = "drop_duplicates" if info["duplicate_count"] > 0 else "none"
    stats = {"duplicate_count": info["duplicate_count"], "duplicate_pct": info["duplicate_pct"]}

    explanation = await _get_gemini_explanation("duplicates", "all_columns", strategy, stats)
    if not explanation:
        explanation = _rule_based_explanation("duplicates", "all_columns", strategy, stats)

    return {
        "issue_type": "duplicates",
        "recommended_strategy": strategy,
        "available_strategies": ["drop_duplicates", "none"],
        "stats": stats,
        "explanation": explanation,
    }

def apply_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


# 4. Outlier Treatment

def detect_outliers(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) < 4: continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (s < lower) | (s > upper)
        count = int(outlier_mask.sum())

        if count == 0: continue

        results.append({
            "column": col,
            "outlier_count": count,
            "outlier_pct": round(count / len(s) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        })

    return results


async def recommend_outliers(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    s = df[column].dropna()

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))

    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (s < lower) | (s > upper)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round(outlier_count / len(s) * 100, 2)

    if outlier_pct < 1: strategy = "keep"
    elif outlier_pct < 5: strategy = "iqr_cap"
    else: strategy = "iqr_cap"

    stats = {
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
    }

    explanation = await _get_gemini_explanation("outliers", column, strategy, stats)

    if not explanation:
        explanation = _rule_based_explanation("outliers", column, strategy, stats)

    return {
        "column": column,
        "issue_type": "outliers",
        "recommended_strategy": strategy,
        "available_strategies": ["iqr_cap", "zscore_cap", "drop_rows", "keep"],
        "stats": stats,
        "explanation": explanation,
    }


def apply_outliers(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    df = df.copy()
    s = df[column].dropna()

    if strategy == "iqr_cap":
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[column] = df[column].clip(lower=lower, upper=upper)

    elif strategy == "zscore_cap":
        mean = s.mean()
        std = s.std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        df[column] = df[column].clip(lower=lower, upper=upper)

    elif strategy == "drop_rows":
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[column] >= lower) & (df[column] <= upper)]

    return df


# 5.Type Conversion

def detect_type_inconsistencies(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        numeric_parsed = pd.to_numeric(s, errors="coerce")
        numeric_rate = numeric_parsed.notna().sum() / len(s)

        if numeric_rate >= 0.8:
            results.append({"column": col, "suggested_type": "numeric", "parse_rate": round(float(numeric_rate), 2)})
            continue

        try:
            dt_parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            dt_rate = dt_parsed.notna().sum() / len(s)

            if dt_rate >= 0.8:
                results.append({"column": col, "suggested_type": "datetime", "parse_rate": round(float(dt_rate), 2)})
        except Exception:
            pass

    return results


def apply_type_conversion(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    df = df.copy()
    if target_type == "to_numeric":
        converted = pd.to_numeric(df[column], errors="coerce")
        df[column] = converted.astype("float64")
    elif target_type == "to_datetime":
        df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")

    return df


# 6.String Standardisation

def detect_string_issues(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0 : continue

        whitespace_count = int(s.apply(lambda x: x != x.strip()).sum())
        mixed_case = int(s.nunique()) != int(s.str.lower().nunique())

        if whitespace_count > 0 or mixed_case:
            results.append({
                "column": col,
                "whitespace_count": whitespace_count,
                "mixed_case": mixed_case,
            })

    return results


def apply_string_standardisation(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    df = df.copy()
    if strategy == "trim_whitespace":
        df[column] = df[column].astype(str).str.strip()
    elif strategy == "lowercase":
        df[column] = df[column].astype(str).str.lower().str.strip()
    elif strategy == "titlecase":
        df[column] = df[column].astype(str).str.title().str.strip()

    return df


# 7. Constant Column Removal

def detect_constant_columns(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in df.columns:
        if len(df) == 0:
            continue
        top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.95:
            top_val = df[col].value_counts(dropna=False).index[0]
            results.append({
                "column": col,
                "dominant_value": str(top_val),
                "frequency_pct": round(float(top_freq) * 100, 2),
            })

    return results


def apply_drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    return df.drop(columns=[column])


# 8. Identifier Column Removal

from services.profile_service import identifier_detection

def detect_identifier_columns(df: pd.DataFrame) -> list[dict]:
    return identifier_detection(df)



# C9 — Negative Value Treatment


def _normalize_column_name(column: str) -> str:
    """Converts a column name into a normalized form for keyword inspection."""
    name = str(column).lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _contains_keyword(name: str, keywords: list[str]) -> bool:
    """Checks whether any keyword occurs as a token or substring in the normalized column name."""
    return any(keyword in name for keyword in keywords)


def _infer_negative_semantics(column: str, series: pd.Series) -> tuple[str, str, str]:
    
    name = _normalize_column_name(column)

    # 1. Category / label encoding (-1/0/1)
    unique_values = set(series.dropna().unique())
    if (
        len(unique_values) <= 3
        and -1 in unique_values
        and unique_values.issubset({-1, 0, 1})
    ):
        return (
            "encoded",
            "high",
            "Values resemble a categorical/binary encoding such as -1/0/1.",
        )

    # 2. Strongly signed concepts (negative values are meaningful)
    signed_keywords = [
        "temperature", "temp",
        "latitude", "longitude", "lat", "lon", "lng",
        "x_coordinate", "y_coordinate", "z_coordinate",
        "delta", "change", "difference", "diff", "variation", "deviation",
        "growth", "growth_rate", "return", "return_pct", "return_percent",
        "profit", "pnl", "net_profit", "net_income", "balance",
        "residual", "prediction_error", "error", "bias", "offset",
        "z_score", "zscore", "standardized", "standardised", "centered", "centred",
        "log", "ln",
        "embedding", "latent", "principal_component", "component", "pca",
        "velocity", "acceleration", "displacement", "signed_distance", "direction",
        "waveform", "signal", "amplitude",
    ]
    if _contains_keyword(name, signed_keywords):
        return (
            "signed",
            "high",
            "Negative values are likely semantically meaningful for this type of feature.",
        )

    # 3. Strongly non-negative concepts
    non_negative_keywords = [
        "price", "amount", "salary", "income", "revenue", "cost", "fee",
        "wage", "expense", "payment", "tax", "budget",
        "count", "number", "quantity", "qty", "total", "units", "items",
        "frequency_count", "occurrences",
        "age",
        "weight", "mass", "height", "length", "width", "breadth", "thickness",
        "area", "volume", "size", "diameter", "radius", "perimeter",
        "distance", "duration", "elapsed_time", "time_elapsed", "wait_time",
        "response_time", "latency", "delay",
        "wavelength", "frequency", "energy", "calories", "steps", "dosage", "concentration",
        "pixel", "intensity", "brightness",
        "probability", "prob", "confidence", "proportion",
        "inventory", "stock", "remaining", "available",
        "days", "hours", "minutes", "seconds",
        "rank", "position_index", "row_number",
    ]
    if _contains_keyword(name, non_negative_keywords):
        if "percent" in name or "percentage" in name or name.endswith("_pct"):
            return (
                "non_negative",
                "medium",
                "The column appears to represent a percentage/proportion.",
            )
        return (
            "non_negative",
            "high",
            "Negative values are generally not meaningful for this type of quantity.",
        )

    # 4. Ambiguous — cannot reliably infer from name alone
    return (
        "ambiguous",
        "low",
        "Column semantics cannot be reliably inferred from its name.",
    )


def _is_tiny_negative_noise(series: pd.Series) -> bool:
    """Detects negative values extremely close to zero — likely floating-point noise."""
    negative_values = series[series < 0]
    if negative_values.empty:
        return False
    scale = max(float(series.abs().median()), 1.0)
    tolerance = max(1e-12, scale * 1e-10)
    return bool((negative_values.abs() <= tolerance).all())


def detect_negative_values(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        negative_count = int((s < 0).sum())
        if negative_count == 0:
            continue
        negative_pct = round(negative_count / len(s) * 100, 2)
        semantic_class, confidence, reason = _infer_negative_semantics(col, s)
        tiny_negative_noise = _is_tiny_negative_noise(s)
        results.append({
            "column": col,
            "negative_count": negative_count,
            "negative_pct": negative_pct,
            "minimum_value": round(float(s.min()), 6),
            "maximum_value": round(float(s.max()), 6),
            "semantic_class": semantic_class,
            "semantic_confidence": confidence,
            "semantic_reason": reason,
            "tiny_negative_noise": tiny_negative_noise,
        })
    return results


async def recommend_negative_values(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Negative value treatment requires a numeric column: {column}")

    s = df[column].dropna()
    if len(s) == 0:
        return {
            "column": column,
            "issue_type": "negative_values",
            "recommended_strategy": "keep",
            "available_strategies": ["keep"],
            "stats": {"negative_count": 0, "negative_pct": 0.0},
            "explanation": "The column contains no non-null values.",
        }

    negative_count = int((s < 0).sum())
    negative_pct = round(negative_count / len(s) * 100, 2)
    semantic_class, confidence, reason = _infer_negative_semantics(column, s)
    tiny_negative_noise = _is_tiny_negative_noise(s)

    # Decision logic
    if negative_count == 0:
        strategy = "keep"
    elif tiny_negative_noise:
        strategy = "set_zero"
    elif semantic_class == "signed":
        strategy = "keep"
    elif semantic_class == "encoded":
        strategy = "flag_only"
    elif semantic_class == "non_negative":
        strategy = "set_nan"
    else:
        strategy = "flag_only"

    stats = {
        "negative_count": negative_count,
        "negative_pct": negative_pct,
        "minimum_value": round(float(s.min()), 6),
        "maximum_value": round(float(s.max()), 6),
        "semantic_class": semantic_class,
        "semantic_confidence": confidence,
        "semantic_reason": reason,
        "tiny_negative_noise": tiny_negative_noise,
    }

    explanation = await _get_gemini_explanation("negative_values", column, strategy, stats)
    if not explanation:
        explanation = _rule_based_explanation("negative_values", column, strategy, stats)

    return {
        "column": column,
        "issue_type": "negative_values",
        "recommended_strategy": strategy,
        "available_strategies": ["keep", "set_zero", "set_nan", "drop_rows", "abs_value", "flag_only"],
        "stats": stats,
        "explanation": explanation,
    }


def apply_negative_values(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    df = df.copy()

    if strategy == "keep":
        pass

    elif strategy == "set_zero":
        df[column] = df[column].clip(lower=0)

    elif strategy == "set_nan":
        df.loc[df[column] < 0, column] = np.nan

    elif strategy == "drop_rows":
        df = df[df[column] >= 0]

    elif strategy == "abs_value":
        df[column] = df[column].abs()

    elif strategy == "flag_only":
        flag_column = f"{column}_negative_flag"
        df[flag_column] = (df[column] < 0).astype(int)

    else:
        raise ValueError(f"Unknown negative-value strategy: {strategy}")

    return df



# Master — get all issues in one call


def get_all_issues(df: pd.DataFrame) -> dict:
   
    return {
        "missing_numerical": detect_missing_numerical(df),
        "missing_categorical": detect_missing_categorical(df),
        "duplicates": detect_duplicates(df),
        "outliers": detect_outliers(df),
        "type_inconsistencies": detect_type_inconsistencies(df),
        "string_issues": detect_string_issues(df),
        "constant_columns": detect_constant_columns(df),
        "identifier_columns": detect_identifier_columns(df),
        "negative_values": detect_negative_values(df),
    }