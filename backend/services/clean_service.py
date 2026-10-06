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
        from google import genai as google_genai

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            return None

        prompt_path = Path("prompts/clean_recommend.txt")
        if not prompt_path.exists():
            return None

        prompt_template = prompt_path.read_text(encoding="utf-8")
        payload = {
            "issue_type": issue_type,
            "column": column,
            "strategy": strategy,
            "stats": stats,
        }
        prompt = prompt_template.replace("{{ISSUE_PAYLOAD}}", json.dumps(payload, indent=2))

        client = google_genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        parsed = json.loads(raw)
        explanation = parsed.get("explanation", "")
        if explanation and len(explanation) > 10:
            return explanation

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

def preview_missing_numerical(df: pd.DataFrame, column: str, strategy: str, fill_value: float = 0.0) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    affected = df[df[column].isna()].head(5).copy()
    before = affected[[column]].copy()

    s = df[column].dropna()
    if strategy == "mean":
        val = s.mean()
    elif strategy == "median":
        val = s.median()
    elif strategy == "constant":
        val = fill_value
    elif strategy in ("drop_rows", "drop_column"):
        return {
            "column": column,
            "strategy": strategy,
            "affected_rows": int(df[column].isna().sum()),
            "before": [],
            "after": [],
            "note": f"This will {'drop ' + str(int(df[column].isna().sum())) + ' rows' if strategy == 'drop_rows' else 'remove the column entirely'}.",
        }
    else:
        val = s.median()

    after = before.copy()
    after[column] = round(float(val), 4)

    return {
        "column": column,
        "strategy": strategy,
        "fill_value": round(float(val), 4),
        "affected_rows": int(df[column].isna().sum()),
        "before": before.to_dict(orient="records"),
        "after": after.to_dict(orient="records"),
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
    if target_type == "numeric":
        df[column] = pd.to_numeric(df[column], errors="coerce")
    elif target_type == "datetime":
        df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")

    return df


