"""
Profile Service
---------------
Implements all 22 profiling functions.
Master function: run_full_profile(df) -> dict

All functions are READ-ONLY — no file is modified here.
Results are cached in core/cache.py after first run.

Gemini is called at the end to rewrite P22 insights as a narrative.
Fallback: if Gemini fails, rule-based templated insights are returned.
"""

import json
from core.gemini_client import generate_json, render_prompt

import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from core.cache import set_profile
from core.config import settings

logger = logging.getLogger(__name__)

current_year = datetime.now().year

# 1. Dataset Dimensions

def get_dimensions(df: pd.DataFrame) -> dict:
    """
    Determines the total number of rows and columns in the dataset. It also provides basic information such as dataset size and memory usage.
    """
    return {
        "rows" : len(df),
        "columns" : len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4),
        "total_cells" : len(df) * len(df.columns),
    }

# 2. Column Identification and Meaning

SEMANTIC_KEYWORDS = {
    "id": ["id", "key", "uuid", "serial", "code", "ref", "no", "num", "identifier"],
    "name": ["name", "firstname", "lastname", "fullname", "username"],
    "email": ["email", "mail"],
    "phone": ["phone", "mobile", "contact", "tel"],
    "date": ["date", "time", "created", "updated", "dob", "timestamp", "dt"],
    "age": ["age", "years"],
    "price": ["price", "cost", "amount", "fee", "salary", "revenue", "income", "wage"],
    "address": ["address", "city", "state", "country", "zip", "pincode", "region"],
    "gender": ["gender", "sex"],
    "status": ["status", "flag", "active", "is_", "has_"],
    "rating": ["rating", "score", "grade", "rank"],
    "percentage": ["pct", "percent", "ratio", "rate"],
}

def _infer_semantic_label(col: str) -> str:
    col_lower = col.lower().replace("_", " ").replace("-", " ")

    for label, keywords in SEMANTIC_KEYWORDS.items():
        if(any(key in col_lower for key in keywords)):
            return label

    return "general"

def identify_columns(df: pd.DataFrame) -> list[dict]:
    """
    Identifies each column and attempts to determine its semantic meaning based on the column name and its values.
    """
    results = []

    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        results.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "semantic_label": _infer_semantic_label(col),
            "sample_values": [str(v) for v in sample],
        })

    return results


# 3. Numerical Column Detection

def detect_numerical_columns(df: pd.DataFrame) -> list[dict]:
    """
    Identifies columns containing numerical data, such as integers, floating-point values, measurements, prices, ages, or quantities.
    """
    results = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].dtype == np.int64 or df[col].dtype == np.int32:
            subtype = "integer"

        elif df[col].dtype == np.float64 or df[col].dtype == np.float32:
            subtype = "float"

        else: subtype = "mixed"

        results.append({

            "column": col,
            "dtype": str(df[col].dtype),
            "sub_type": subtype,
        })

    return results


# 4. Categorical Column Detection

def detect_categorical_columns(df: pd.DataFrame) -> list[dict]:
    """
    Identifies columns containing categorical values, such as gender, city, department, product type, or other discrete categories.
    """

    results = []

    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "category":
            unique_cnt = df[col].nunique()
            top_values = df[col].value_counts().head(5).index.tolist()

            results.append(
                {
                    "column": col,
                    "unique_count": int(unique_cnt),
                    "top_values": [str(v) for v in top_values],
                }
            )

    return results

# 5. Date/Time Column 

def detect_datetime_columns(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in df.select_dtypes(include=["datetime64"]).columns:
        results.append({"column": col, "inferred_format": "datetime64", "already_parsed": True})

    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(50)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            success_rate = parsed.notna().sum() / max(len(sample), 1)
            if success_rate >= 0.8:
                results.append({
                    "column": col,
                    "inferred_format": "inferred",
                    "already_parsed": False,
                    "parse_success_rate": round(float(success_rate), 2),
                })
        except Exception:
            pass
        
    return results

# 6. Missing Value Analysis
def missing_value_analysis(df: pd.DataFrame) ->dict:
    """
    Identifies columns containing missing or null values and reports both the number and percentage of missing values.
    """
    results = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        percen = round(cnt / len(df) * 100, 2) if len(df) > 0 else 0.0

        results[col] = {
            "count": cnt,
            "pct":percen,
            "is_critical": percen > 30,
        }

    return results


# 7. Unique Value and Cardinality Analysis
def cardinality_analysis(df: pd.DataFrame)->dict:
    """
    Determines the number of unique values in each column and calculates its cardinality to identify low-, medium-, and high-cardinality features.
    """
    results = {}
    
    for col in df.columns:
        unique = int(df[col].nunique())

        if unique < 10: card_type = "low"
        elif unique <= 100 : card_type = "medium"
        else : card_type = "high"

        results[col] = {
            "unique_count": unique,
            "cardinality_type": card_type
        }

    return results

# 8. Duplicate Row Detection
def duplicate_row_detection(df: pd.DataFrame) -> dict:
    """
    Identifies completely duplicated rows in the dataset and reports their count and percentage.
    """
    dup = df.duplicated()

    count = int(dup.sum())
    percent = round(count / len(df) * 100, 2) if len(df) > 0 else 0.0

    sample_indices = df[dup].head(5).index.tolist()

    return {
        "count": count,
        "pct": percent,
        "sample_indices": [int(i) for i in sample_indices],
    }

# 9. Duplicate Column Detection
def duplicate_column_detection(df: pd.DataFrame) ->list:
    """
    Checks whether two or more columns contain identical data and identifies redundant columns.
    """
    pairs = []
    cols = df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                if df[cols[i]].equals(df[cols[j]]):
                    pairs.append({"col1": cols[i], "col2": cols[j]})
            except Exception:
                pass
    return pairs

# 10. Numerical Statistical Summary

def numerical_statistical_summary(df:pd.DataFrame) -> dict:
    """
    Provides basic statistical information for numerical columns, including minimum, maximum, mean, median, standard deviation, and range.
    """
    results = {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        curr = df[col].dropna()

        if(len(curr) == 0) : continue

        results[col] = {
            "min": round(float(curr.min()), 4),
            "max": round(float(curr.max()), 4),
            "mean": round(float(curr.mean()), 4),
            "median": round(float(curr.median()), 4),
            "std": round(float(curr.std()), 4),
            "range": round(float(curr.max() - curr.min()), 4),
            "q1": round(float(curr.quantile(0.25)), 4),
            "q3": round(float(curr.quantile(0.75)), 4),
        }

    return results

# 11. Suspicious and Invalid Value Detection
def suspicious_value_detection(df: pd.DataFrame) -> dict:
    """
    Identifies potentially invalid, impossible, or suspicious values, such as negative ages, percentages above 100%, or values outside expected domain ranges.
    """
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        col_lower = col.lower()
        flags = []

        # --- Rule 1: Age validation ---
        if "age" in col_lower:
            mask = (df[col] < 0) | (df[col] > 120)

            # Check if any rows violated the age condition
            if mask.sum() > 0:
                flags.append({
                    "rule": "Age must be between 0 and 120",
                    "flagged_count": int(mask.sum()),
                    "sample_values": df[col][mask].head(3).tolist(),
                })

        # --- Rule 2: Percentage / Ratio validation ---
        if any(k in col_lower for k in ["pct", "percent", "ratio", "percentage"]):
            mask = (df[col] < 0) | (df[col] > 100)

            if mask.sum() > 0:
                # Append rule violation details to the flags list
                flags.append({
                    "rule": "Percentage must be between 0 and 100",
                    "flagged_count": int(mask.sum()),
                    "sample_values": df[col][mask].head(3).tolist(),
                })

        # --- Rule 3: Year validation ---
        if "year" in col_lower:

            mask = (df[col] < 1900) | (df[col] > current_year + 1)

            if mask.sum() > 0:
                flags.append({
                    "rule": f"Year must be between 1900 and {current_year + 1}",
                    "flagged_count": int(mask.sum()),
                    "sample_values": df[col][mask].head(3).tolist(),
                })

        # --- Rule 4: Financial values non-negativity validation ---
        if any(
            k in col_lower
            for k in ["price", "amount", "revenue", "salary", "income", "cost", "fee", "wage"]
        ):
            mask = df[col] < 0

            if mask.sum() > 0:
                flags.append({
                    "rule": "Financial value must not be negative",
                    "flagged_count": int(mask.sum()),
                    "sample_values": df[col][mask].head(3).tolist(),
                })

        if flags:
            result[col] = flags

    return result


# 12. Data Type Consistency

def data_type_consistency(df: pd.DataFrame) -> list[dict]:
    """
    Checks whether the actual values in a column match its assigned data type and detects cases such as numerical values stored as text or inconsistent date formats.
    """
    inconsistent = []
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        # Try numeric conversion
        numeric_parsed = pd.to_numeric(s, errors="coerce")
        numeric_rate = numeric_parsed.notna().sum() / len(s)
        if numeric_rate >= 0.8:
            inconsistent.append({
                "column": col,
                "current_type": "object",
                "suggested_type": "numeric",
                "parse_rate": round(float(numeric_rate), 2),
            })
            continue

        # Try datetime conversion
        try:
            dt_parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            dt_rate = dt_parsed.notna().sum() / len(s)
            if dt_rate >= 0.8:
                inconsistent.append({
                    "column": col,
                    "current_type": "object",
                    "suggested_type": "datetime",
                    "parse_rate": round(float(dt_rate), 2),
                })
        except Exception:
            pass


    return inconsistent

# 13. Constant and Near-Constant Column Detection
def constant_column_detection(df: pd.DataFrame) -> list[dict]:
    """
    Identifies columns containing only one value or where one value dominates almost the entire column. Such columns may provide little analytical value.
    """
    result = []
    for col in df.columns:
        if len(df) == 0:
            continue
        top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.95:
            top_val = df[col].value_counts(dropna=False).index[0]
            result.append({
                "column": col,
                "dominant_value": str(top_val),
                "frequency_pct": round(float(top_freq) * 100, 2),
            })
    return result

# 14. Identifier Detection
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

ID_KEYWORDS = {"id", "key", "uuid", "code", "serial", "ref"}

def identifier_detection(df: pd.DataFrame) -> list[dict]:
    """
    Attempts to identify columns that are likely to represent IDs, such as customer IDs, transaction IDs, serial numbers, or UUIDs, especially when they contain mostly or entirely unique values.
    """
    result = []

    for col in df.columns:
        col_lower = col.lower().replace("_", "").replace("-", "")
        unique_ratio = df[col].nunique() / max(len(df), 1)

        name_match = any(kw in col_lower for kw in ID_KEYWORDS)

        sample = df[col].dropna().astype(str).head(20)

        uuid_match = (
            sample.apply(lambda v: bool(UUID_RE.match(v))).mean() >= 0.8
        )

        try:
            numeric = pd.to_numeric(df[col].dropna(), errors="coerce").dropna()

            sequential = (
                len(numeric) > 1
                and numeric.is_monotonic_increasing
                and (numeric.diff().dropna() == 1).all()
            )
        except Exception:
            sequential = False

        if unique_ratio > 0.9 and (name_match or uuid_match or sequential):
            result.append({
                "column": col,
                "unique_ratio": round(float(unique_ratio), 4),
                "reason": (
                    "name_keyword"
                    if name_match
                    else "uuid_pattern"
                    if uuid_match
                    else "sequential_int"
                ),
            })

    return result

# 15. Outlier Detection
def outlier_detection(df: pd.DataFrame) -> dict:
    """
    Identifies unusually distant observations in numerical columns using statistical methods such as the Interquartile Range (IQR) or standard deviation-based methods. Outliers are flagged as potential observations for investigation rather than automatically treated as errors.
    """

    result = {}
    for col in df.select_dtypes(include = [np.number]).columns:
        s = df[col].dropna()
        if(len(s) < 4) : continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))

        iqr = q3 - q1

        lower = q1 - (1.5*iqr)
        upper = q3 + (1.5*iqr)

        outliers = s[(s<lower) | (s>upper)]

        result[col] = {
            "outlier_count": int(len(outliers)),
            "pct": round(len(outliers) / len(s) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        }

    return result


# 16. Distribution Analysis

def distribution_analysis(df: pd.DataFrame) -> dict:
    """
    Analyzes the distribution of numerical variables using measures such as skewness and 
    basic distribution characteristics to identify symmetric, skewed, or highly concentrated data.
    """    
    result = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()

        if len(s) < 4:
            continue

        skew = float(s.skew())
        kurt = float(s.kurtosis())

        if abs(skew) < 0.5:
            label = "symmetric"

        elif abs(skew) <= 1.0:
            label = "moderately skewed"

        else:
            label = "highly skewed"

        result[col] = {
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "distribution_label": label,
        }

    return result

# 17. Categorical Distribution and Class Imbalance
def categorical_imbalance(df: pd.DataFrame) -> dict:
    """
    Analyzes the frequency distribution of categorical variables and detects cases where one or more categories dominate the dataset. This is particularly useful for identifying class imbalance before machine learning.
    """
    result = {}

    for col in df.select_dtypes(include=["object"]).columns:
        vc = df[col].value_counts(normalize=True, dropna=True)

        if len(vc) == 0:
            continue
        top_pct = round(float(vc.iloc[0]) * 100, 2)
        result[col] = {
            "top_category": str(vc.index[0]),
            "top_pct": top_pct,
            "is_imbalanced": top_pct > 80,
        }
    return result

# 18. Date/Time Coverage Analysis

def datetime_coverage(df: pd.DataFrame) -> dict:
    """
    For date/time columns, identifies the earliest and latest dates, total time coverage, and potential gaps or irregularities in the temporal data.
    """
    result = {}
    for col in df.select_dtypes(include=["datetime64"]).columns:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        s_sorted = s.sort_values()
        total_days = (s_sorted.iloc[-1] - s_sorted.iloc[0]).days
        diffs = s_sorted.diff().dropna()
        median_gap = diffs.median().days if len(diffs) > 0 else 0
        expected_count = total_days / max(median_gap, 1) if median_gap > 0 else len(s)
        gap_count = max(0, int(expected_count - len(s)))
        result[col] = {
            "min_date": str(s_sorted.iloc[0].date()),
            "max_date": str(s_sorted.iloc[-1].date()),
            "total_days": total_days,
            "estimated_gap_count": gap_count,
        }

    return result


# 19. Initial Relationship Screening

def initial_relationship_screening(df: pd.DataFrame) -> list[dict]:
    """
    Performs basic screening of relationships between numerical variables, such as correlation analysis, to identify potentially interesting relationships that can be investigated during Exploratory Data Analysis (EDA).
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return []
    # Cap at 20 columns for performance
    if num_df.shape[1] > 20:
        num_df = num_df.iloc[:, :20]

    corr = num_df.corr()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7 and not np.isnan(r):
                pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": round(float(r), 4),
                })
    return pairs



# 20. Potential Target Detection


def potential_target_detection(
    df: pd.DataFrame,
    identifier_cols: list[str],
    datetime_cols: list[str],
    constant_cols: list[str],
) -> list[dict]:
    """
    Identifies columns that could potentially serve as target variables for machine learning based on their data type, cardinality, and naming patterns. The system should recommend candidates rather than automatically selecting a target.
    """
    exclude = set(identifier_cols + datetime_cols + constant_cols)
    candidates = []

    for col in df.columns:
        if col in exclude:
            continue
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 30:
            continue

        unique = df[col].nunique()
        dtype = df[col].dtype

        if unique == 2:
            score, suitability, reasoning = 90, "binary_classification", "Binary column — ideal classification target"

        elif dtype == object and unique <= 10:
            score, suitability, reasoning = 75, "multi_class_classification", f"Categorical with {unique} classes — good classification target"

        elif dtype == object and unique <= 20:
            score, suitability, reasoning = 60, "multi_class_classification", f"Categorical with {unique} classes — possible classification target"

        elif np.issubdtype(dtype, np.number):
            score, suitability, reasoning = 70, "regression", "Continuous numerical — good regression target"
        else: continue

        candidates.append({
            "column": col,
            "score": score,
            "suitability": suitability,
            "reasoning": reasoning,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:3]


# 21. Data Quality Summary


def data_quality_summary(df: pd.DataFrame, all_results: dict) -> dict:
    """
    Generates an overall summary of the dataset's quality by combining detected issues such as missing values, duplicates, inconsistent data types, suspicious values, outliers, and class imbalance
    """
    score = 100
    breakdown = []

    missing = all_results.get("missing_values", {})
    critical_missing = [c for c, v in missing.items() if v["is_critical"]]
    if critical_missing:
        score -= 20
        breakdown.append(f"Critical missing (>30%): {critical_missing} (-20)")

    moderate_missing = [c for c, v in missing.items() if 5 < v["pct"] <= 30]
    deduction = min(len(moderate_missing) * 10, 30)
    if deduction > 0:
        score -= deduction
        breakdown.append(f"Moderate missing columns: {len(moderate_missing)} (-{deduction})")

    dups = all_results.get("duplicate_rows", {})
    if dups.get("pct", 0) > 5:
        score -= 10
        breakdown.append("Duplicate rows > 5% (-10)")

    suspicious = all_results.get("suspicious_values", {})
    if suspicious:
        score -= 10
        breakdown.append("Suspicious values detected (-10)")

    imbalance = all_results.get("categorical_imbalance", {})
    imbalanced = [c for c, v in imbalance.items() if v["is_imbalanced"]]
    if imbalanced:
        score -= 5
        breakdown.append(f"Imbalanced categories: {imbalanced} (-5)")

    score = max(0, score)
    if score >= 80:
        label = "excellent"
    elif score >= 60:
        label = "good"
    elif score >= 40:
        label = "fair"
    else:
        label = "poor"

    return {"score": score, "label": label, "breakdown": breakdown}



# 22. Insights Generation (rule-based fallback)

def _generate_rule_based_insights(all_results: dict) -> list[str]:
    """
    Converts profiling results into human-readable observations and warnings. For example, the system may report that a column has a high percentage of missing values, a dataset contains significant class imbalance, or a numerical variable is highly skewed
    """
    insights = []

    missing = all_results.get("missing_values", {})
    for col, v in missing.items():
        if v["pct"] > 5:
            strategy = "median imputation" if v["pct"] < 30 else "careful review or removal"
            insights.append(
                f"Column '{col}' has {v['pct']}% missing values — {strategy} is recommended."
            )

    dups = all_results.get("duplicate_rows", {})
    if dups.get("count", 0) > 0:
        insights.append(
            f"The dataset contains {dups['count']} duplicate rows ({dups['pct']}%) — consider removing them."
        )

    dist = all_results.get("distribution_analysis", {})
    for col, v in dist.items():
        if abs(v["skewness"]) > 1:
            direction = "right" if v["skewness"] > 0 else "left"
            insights.append(
                f"'{col}' is highly {direction}-skewed (skewness={v['skewness']}) — a log or power transform may help."
            )

    ids = all_results.get("identifier_columns", [])
    for item in ids:
        insights.append(
            f"'{item['column']}' appears to be an identifier column and should likely be excluded from ML models."
        )

    imbalance = all_results.get("categorical_imbalance", {})
    for col, v in imbalance.items():
        if v["is_imbalanced"]:
            insights.append(
                f"'{col}' shows class imbalance: '{v['top_category']}' accounts for {v['top_pct']}% of records."
            )

    corr_pairs = all_results.get("correlation_pairs", [])
    for pair in corr_pairs[:3]:
        insights.append(
            f"'{pair['col1']}' and '{pair['col2']}' are strongly correlated (r={pair['correlation']}) — consider dropping one for ML."
        )

    suspicious = all_results.get("suspicious_values", {})
    for col, flags in suspicious.items():
        for flag in flags[:1]:
            insights.append(
                f"'{col}' has {flag['flagged_count']} suspicious values: {flag['rule']}."
            )

    quality = all_results.get("quality_summary", {})
    if quality.get("label") in ["fair", "poor"]:
        insights.append(
            f"Overall data quality is {quality['label']} (score: {quality['score']}/100). "
            "Cleaning is strongly recommended before analysis."
        )

    return insights[:12] if insights else ["Dataset loaded. Run cleaning for a full quality assessment."]

async def _generate_gemini_insights(all_results: dict) -> list[str] | None:
    """
    Send the profiling summary to Gemini and get back a list of insight strings.
    Returns None on any failure so the caller can fall back to rule-based.
    Only summary statistics are sent, never raw rows.
    """
    try:
        summary = {
            "dimensions": all_results.get("dimensions"),
            "quality_summary": all_results.get("quality_summary"),
            "missing_values": {
                col: v for col, v in all_results.get("missing_values", {}).items()
                if v["pct"] > 0
            },
            "duplicate_rows": all_results.get("duplicate_rows"),
            "outlier_summary": {
                col: v for col, v in all_results.get("outlier_detection", {}).items()
                if v["outlier_count"] > 0
            },
            "distribution_analysis": all_results.get("distribution_analysis"),
            "correlation_pairs": all_results.get("correlation_pairs"),
            "identifier_columns": all_results.get("identifier_columns"),
            "categorical_imbalance": all_results.get("categorical_imbalance"),
            "suspicious_values": all_results.get("suspicious_values"),
            "potential_targets": all_results.get("potential_targets"),
        }

        prompt = render_prompt(
            "profiling_insights.txt",
            PROFILING_SUMMARY=json.dumps(summary, indent=2, default=str),
        )
        if prompt is None:
            return None

        parsed = await generate_json(prompt)
        if not parsed:
            return None

        insights = parsed.get("insights", [])
        if isinstance(insights, list) and len(insights) >= 3:
            return [str(i) for i in insights]

    except Exception as e:
        logger.warning(f"Gemini insights generation failed: {e}. Falling back to rule-based.")

    return None

async def run_full_profile(df: pd.DataFrame) -> dict:
    """
    Run all 22 profiling checks and return a combined result dict.
    Gemini is called at the end to generate narrative insights.
    Falls back to rule-based insights if Gemini is unavailable.
    """
    # Collect all algorithmic results first
    identifier_cols_raw = identifier_detection(df)
    identifier_col_names = [i["column"] for i in identifier_cols_raw]

    datetime_cols_raw = detect_datetime_columns(df)
    datetime_col_names = [d["column"] for d in datetime_cols_raw]

    constant_cols_raw = constant_column_detection(df)
    constant_col_names = [c["column"] for c in constant_cols_raw]

    missing = missing_value_analysis(df)
    dups = duplicate_row_detection(df)
    suspicious = suspicious_value_detection(df)
    imbalance = categorical_imbalance(df)
    dist = distribution_analysis(df)
    corr_pairs = initial_relationship_screening(df)
    quality = data_quality_summary(df, {
        "missing_values": missing,
        "duplicate_rows": dups,
        "suspicious_values": suspicious,
        "categorical_imbalance": imbalance,
    })

    all_results = {
        "dimensions": get_dimensions(df),
        "column_identification": identify_columns(df),
        "numerical_columns": detect_numerical_columns(df),
        "categorical_columns": detect_categorical_columns(df),
        "datetime_columns": datetime_cols_raw,
        "missing_values": missing,
        "cardinality": cardinality_analysis(df),
        "duplicate_rows": dups,
        "duplicate_columns": duplicate_column_detection(df),
        "numerical_summary": numerical_statistical_summary(df),
        "suspicious_values": suspicious,
        "type_inconsistencies": data_type_consistency(df),
        "constant_columns": constant_cols_raw,
        "identifier_columns": identifier_cols_raw,
        "outlier_detection": outlier_detection(df),
        "distribution_analysis": dist,
        "categorical_imbalance": imbalance,
        "datetime_coverage": datetime_coverage(df),
        "correlation_pairs": corr_pairs,
        "potential_targets": potential_target_detection(
            df, identifier_col_names, datetime_col_names, constant_col_names
        ),
        "quality_summary": quality,
    }

    gemini_insights = await _generate_gemini_insights(all_results)
    all_results["insights"] = gemini_insights if gemini_insights else _generate_rule_based_insights(all_results)
    all_results["insights_source"] = "gemini" if gemini_insights else "rule_based"

    return all_results