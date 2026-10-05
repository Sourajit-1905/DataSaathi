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