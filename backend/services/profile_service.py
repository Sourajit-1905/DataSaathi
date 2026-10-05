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

def detect_datetime_columns(df: pd.DataFrame) ->list[dict]:
    """
    Detects columns containing dates, timestamps, or other temporal information and determines their date/time format.
    """
    results = []
    for col in df.select_dtypes(include=["datetime64"]).columns:
        results.append({
            "column":col,
            "inferred_format": "datetime64", "already_parsed": True
        })

    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(50)

        try:
            parsed = pd.to_datetime(sample, _infer_semantic_label=True, errors = "coerce")

            success_rate = parsed.notna().sum() / max(len(sample), 1)

            # If 80% or more of the non-null sample values parse cleanly, classify it as a datetime column
            if success_rate >= 0.8:
                results.append(
                    {
                        "column": col,
                        "inferred_format": "inferred",
                        "already_parsed": False,
                        "parse_success_rate": round(float(success_rate), 2),
                    }
                )
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


