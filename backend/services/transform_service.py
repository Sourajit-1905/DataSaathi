import json
import logging
import re
from pathlib import Path
import json
from core.gemini_client import generate_json, render_prompt
import numpy as np
import pandas as pd

from core.config import settings
from core.exceptions import ColumnNotFoundError, InvalidFormulaError

logger = logging.getLogger(__name__)


# Gemini explanation helper

async def _get_gemini_explanation(
    transform_type: str,
    column: str,
    stats: dict,
) -> str | None:
    try:
        payload = {
            "transform_type": transform_type,
            "column": column,
            "stats": stats,
        }
        prompt = render_prompt(
            "transform_recommend.txt",
            TRANSFORM_PAYLOAD=json.dumps(payload, indent=2, default=str),
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
        logger.warning(f"Gemini explanation failed for {transform_type}/{column}: {e}")

    return None

def _rule_based_explanation(transform_type: str, column: str, stats: dict) -> str:
    skew = stats.get("skewness", 0)
    unique = stats.get("unique_count", "?")
    explanations = {
        "minmax": f"'{column}' will be scaled to [0, 1] — useful when features need to be on the same scale for distance-based models.",
        "zscore": f"'{column}' will be standardised to mean=0, std=1 — recommended for linear models and neural networks.",
        "robust": f"'{column}' will be scaled using median and IQR — robust to outliers unlike min-max or z-score.",
        "log": f"'{column}' is right-skewed (skewness={skew}) — log transform compresses the long tail and improves normality.",
        "power": f"'{column}' has skewness={skew} — power transform (Yeo-Johnson) makes the distribution more Gaussian.",
        "label_encode": f"'{column}' has {unique} categories — label encoding converts them to integers for tree-based models.",
        "onehot": f"'{column}' has {unique} categories — one-hot encoding creates binary columns for linear models.",
        "ordinal": f"'{column}' has {unique} ordered categories — ordinal encoding preserves the rank relationship.",
        "frequency": f"'{column}' has {unique} categories — frequency encoding replaces each category with its occurrence rate.",
        "target_encode": f"'{column}' has {unique} categories — target encoding replaces each category with the mean target value.",
        "equal_width": f"'{column}' will be binned into equal-width intervals — useful for converting continuous to categorical.",
        "equal_freq": f"'{column}' will be binned into equal-frequency quantiles — ensures balanced bin counts.",
        "custom_bins": f"'{column}' will be binned using custom thresholds — useful when domain knowledge defines meaningful ranges.",
        "polynomial": f"'{column}' squared term will be added — captures non-linear relationships with the target.",
        "interaction": f"Interaction term will capture the combined effect of two columns on the target.",
        "formula": f"Custom formula column will be derived from existing columns.",
        "lag": f"'{column}' lag feature will capture time-series dependency on previous values.",
        "rolling_mean": f"'{column}' rolling mean will smooth the time-series signal.",
        "extract_year": f"Year extracted from '{column}' — useful for trend analysis.",
        "extract_month": f"Month extracted from '{column}' — useful for seasonal analysis.",
        "extract_day": f"Day extracted from '{column}' — useful for daily pattern analysis.",
        "extract_dow": f"Day-of-week extracted from '{column}' — useful for weekly cycle analysis.",
        "extract_hour": f"Hour extracted from '{column}' — useful for intraday pattern analysis.",
        "pca": f"PCA will reduce the selected columns to 2 components — useful for visualisation and removing multicollinearity.",
        "exclude_ml": f"'{column}' will be flagged as excluded from ML features — identifier and constant columns should be excluded.",
    }

    return explanations.get(transform_type, f"Transform '{transform_type}' applied to '{column}'.")


# Helper — column stats for recommendation

def _get_column_stats(df: pd.DataFrame, column: str) -> dict:
    s = df[column].dropna()
    stats = {"column": column, "dtype": str(df[column].dtype)}
    if pd.api.types.is_numeric_dtype(df[column]):
        stats.update({
            "mean": round(float(s.mean()), 4) if len(s) > 0 else None,
            "std": round(float(s.std()), 4) if len(s) > 0 else None,
            "min": round(float(s.min()), 4) if len(s) > 0 else None,
            "max": round(float(s.max()), 4) if len(s) > 0 else None,
            "skewness": round(float(s.skew()), 4) if len(s) > 3 else 0.0,
            "has_negative": bool((s < 0).any()),
            "has_zero": bool((s == 0).any()),
        })
    else:
        stats.update({
            "unique_count": int(df[column].nunique()),
            "top_values": df[column].value_counts().head(5).index.tolist(),
        })
    return stats


# Recommendation — suggest best transform per column

async def recommend_transform(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    stats = _get_column_stats(df, column)
    dtype = df[column].dtype

    if pd.api.types.is_numeric_dtype(df[column]):
        skew = stats.get("skewness", 0)
        has_neg = stats.get("has_negative", False)
        has_zero = stats.get("has_zero", False)

        if abs(skew) > 1 and not has_neg and not has_zero:
            transform_type = "log"
        elif abs(skew) > 0.5:
            transform_type = "power"
        else:
            transform_type = "zscore"

        available = ["minmax", "zscore", "robust", "log", "power",
                     "polynomial", "lag", "rolling_mean"]

    elif dtype == object:
        unique_count = stats.get("unique_count", 0)
        if unique_count <= 2:
            transform_type = "label_encode"
        elif unique_count <= 10:
            transform_type = "onehot"
        else:
            transform_type = "frequency"

        available = ["label_encode", "onehot", "ordinal",
                     "frequency", "target_encode"]

    elif "datetime" in str(dtype):
        transform_type = "extract_month"
        available = ["extract_year", "extract_month", "extract_day",
                     "extract_dow", "extract_hour"]
    else:
        transform_type = "label_encode"
        available = ["label_encode"]

    explanation = await _get_gemini_explanation(transform_type, column, stats)
    if not explanation:
        explanation = _rule_based_explanation(transform_type, column, stats)

    return {
        "column": column,
        "recommended_transform": transform_type,
        "available_transforms": available,
        "stats": stats,
        "explanation": explanation,
    }


# Preview helper — before/after stats

def _before_after_stats(before: pd.Series, after: pd.Series) -> dict:
    def series_stats(s):
        s = s.dropna()
        if len(s) == 0:
            return {}
        if pd.api.types.is_numeric_dtype(s):
            return {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(float(s.skew()), 4),
            }
        return {"unique_count": int(s.nunique()), "sample": s.head(5).tolist()}

    return {
        "before": series_stats(before),
        "after": series_stats(after),
    }

from pandas import DataFrame


# 1. Min-Max Normalisation


def apply_minmax(df:DataFrame, column: str) -> DataFrame:
    """
    Scales numerical values to a common range, typically between 0 and 1. This is useful when numerical features have significantly different scales.
    """
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    df = df.copy()
    s = df[column]

    col_min = s.min()
    col_max = s.max()

    if col_max == col_min: df[column] = 0.0
    else : df[column] = (s-col_min) / (col_max - col_min)

    return df

def preview_minmax(df:DataFrame, column:str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    after_df = apply_minmax(df, column)

    return _before_after_stats(df[column], after_df[column])


# 2. Z-Score Standardisation

def apply_zscore(df:DataFrame, column: str) -> DataFrame:
    """
    Transforms numerical variables so that they have a mean close to 0 and a standard deviation close to 1.
    This is particularly useful for algorithms that are sensitive to feature scale, such as SVM, KNN, PCA, and certain optimization-based models

    """
    if column not in df.columns: raise ColumnNotFoundError(column)

    df = df.copy()
    s = df[column]

    mean_val = s.mean()
    std_val = s.std()

    if std_val == 0: df[column] = 0.0
    else: df[column] = (s - mean_val) / std_val

    return df

def preview_zscore(df:DataFrame, column:str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    after_df = apply_zscore(df, column)

    return _before_after_stats(df[column], after_df[column])


# 3. Robust Scaling

def apply_robust(df: DataFrame, column:str)->DataFrame:
    """
    Scales features using median and IQR, making it more resistant to outliers.
    """
    if column not in df.columns: raise ColumnNotFoundError(column)

    df = df.copy()
    s = df[column]

    median_val = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)

    if iqr == 0: df[column] = 0.0
    else: df[column] = (s - median_val) / iqr

    return df




# 4. Log Transform

def apply_log(df: DataFrame, column: str) -> DataFrame:
    if column not in df.columns: raise ColumnNotFoundError(column)

    df = df.copy()

    df[column] = np.log1p(df[column])

    return df

def preview_log(df: DataFrame, column: str) -> dict:
    if column not in df.columns: raise ColumnNotFoundError(column)
    
    after_df = apply_log(df, column)
    
    return _before_after_stats(df[column], after_df[column])


# 5. Power Transform (Yeo-Johnson)

def apply_power(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    from sklearn.preprocessing import PowerTransformer
    df = df.copy()
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    valid_mask = df[column].notna()
    
    df.loc[valid_mask, column] = pt.fit_transform(
        df.loc[valid_mask, column].values.reshape(-1, 1)
    ).flatten()
    
    return df

# 6 Label Encoding

def apply_label_encode(df:DataFrame, column: str) -> DataFrame:
    if column not in df.columns: raise ColumnNotFoundError(column)

    df = df.copy()
    categories = df[column].dropna().unique()
    mapping = {cat: idx for idx, cat in enumerate(sorted(categories, key=str))}

    df[column] = df[column].map(mapping)

    return df


# 7. One-Hot Encoding

def apply_onehot(df:DataFrame, column: str) -> DataFrame:
    if column not in df.columns: raise ColumnNotFoundError(column)

    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=False)

    df = df.drop(columns=[column])
    df = pd.concat([df, dummies], axis=1)

    return df

# 8 Ordinal Encoding

def apply_ordinal(df: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    mapping = {cat: idx for idx, cat in enumerate(order)}
    df[column] = df[column].map(mapping)
    return df


# 9  Frequency Encoding

def apply_frequency_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    freq_map = df[column].value_counts(normalize=True)
    df[column] = df[column].map(freq_map)
    return df


# 10 Target Encoding

def apply_target_encode(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    if target not in df.columns:
        raise ColumnNotFoundError(target)
    df = df.copy()
    target_mean = df.groupby(column)[target].mean()
    df[column] = df[column].map(target_mean)
    return df


# 11  Equal-Width Binning

def apply_equal_width_bin(df: pd.DataFrame, column: str, bins: int, new_column: str = None) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    out_col = new_column or f"{column}_bin"
    df[out_col] = pd.cut(df[column], bins=bins, labels=False)
    return df



# 12  Equal-Frequency Binning

def apply_equal_freq_bin(df: pd.DataFrame, column: str, bins: int, new_column: str = None) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    out_col = new_column or f"{column}_qbin"
    df[out_col] = pd.qcut(df[column], q=bins, labels=False, duplicates="drop")
    return df


# 13 Custom Threshold Binning

def apply_custom_bins(
    df: pd.DataFrame,
    column: str,
    thresholds: list[float],
    labels: list[str],
    new_column: str = None,
) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    out_col = new_column or f"{column}_custom_bin"
    bins = [-np.inf] + sorted(thresholds) + [np.inf]
    df[out_col] = pd.cut(df[column], bins=bins, labels=labels)
    return df


# 14 — Polynomial Feature (degree 2)

def apply_polynomial(df: DataFrame, column: str) -> DataFrame:
    if column not in df.columns: raise ColumnNotFoundError(column)

    df = df.copy()

    df[f"{column}_sq"] = df[column] ** 2

    return df


# 15  Interaction Term

def apply_interaction(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    if col_a not in df.columns:
        raise ColumnNotFoundError(col_a)
    if col_b not in df.columns:
        raise ColumnNotFoundError(col_b)
    df = df.copy()
    df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
    return df


# 16 Custom Formula Column

def apply_formula(df: pd.DataFrame, formula: str, new_column: str) -> pd.DataFrame:
    from core.security import validate_formula

    is_valid, reason = validate_formula(formula, df.columns.tolist())
    if not is_valid:
        raise InvalidFormulaError(reason)

    new_column = "derived" if new_column is None else str(new_column).strip()
    if not 1 <= len(new_column) <= 100 or any(ord(ch) < 32 for ch in new_column):
        raise InvalidFormulaError(
            "The new column name must be 1 to 100 characters long, with no control characters."
        )

    df = df.copy()
    try:
        df[new_column] = df.eval(formula, local_dict={}, global_dict={})
    except Exception:
        raise InvalidFormulaError(
            "The formula could not be evaluated. Check the column names, operators and brackets."
        )
    
    return df


# 17  Lag Feature
#

def apply_lag(df: pd.DataFrame, column: str, periods: int = 1) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    df[f"{column}_lag{periods}"] = df[column].shift(periods)
    return df



# T18 Rolling Mean


def apply_rolling_mean(df: pd.DataFrame, column: str, window: int = 3) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    df[f"{column}_roll{window}"] = df[column].rolling(window=window, min_periods=1).mean()
    return df


# T19-T23  Datetime Extractions


def _parse_datetime_col(df: pd.DataFrame, column: str) -> pd.Series:
    s = df[column]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce", format="mixed")
    return s


def apply_extract_year(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    df[f"{column}_year"] = _parse_datetime_col(df, column).dt.year
    return df


def apply_extract_month(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    df[f"{column}_month"] = _parse_datetime_col(df, column).dt.month
    return df


def apply_extract_day(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    df[f"{column}_day"] = _parse_datetime_col(df, column).dt.day
    return df


def apply_extract_dow(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    df[f"{column}_dow"] = _parse_datetime_col(df, column).dt.dayofweek
    return df


def apply_extract_hour(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ColumnNotFoundError(column)
    df = df.copy()
    df[f"{column}_hour"] = _parse_datetime_col(df, column).dt.hour
    return df


# T24  PCA (2 components)


def apply_pca(df: pd.DataFrame, columns: list[str], prefix: str = "pca") -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    for col in columns:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    df = df.copy()
    subset = df[columns].dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)

    df.loc[subset.index, f"{prefix}_1"] = components[:, 0]
    df.loc[subset.index, f"{prefix}_2"] = components[:, 1]

    return df


# 25  Flag column as excluded from ML

# Stored in session history — no df mutation needed
def flag_exclude_ml(column: str) -> dict:
    return {
        "transform_type": "exclude_ml",
        "column": column,
        "note": f"'{column}' flagged as excluded from ML features.",
    }


# Master apply dispatcher

def apply_transform(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """
    Single entry point for all transformations.
    payload keys: transform_type, column, [options...]
    """
    t = payload.get("transform_type")
    col = payload.get("column")

    if t == "minmax":
        return apply_minmax(df, col)
    elif t == "zscore":
        return apply_zscore(df, col)
    elif t == "robust":
        return apply_robust(df, col)
    elif t == "log":
        return apply_log(df, col)
    elif t == "power":
        return apply_power(df, col)
    elif t == "label_encode":
        return apply_label_encode(df, col)
    elif t == "onehot":
        return apply_onehot(df, col)
    elif t == "ordinal":
        return apply_ordinal(df, col, payload.get("order", []))
    elif t == "frequency":
        return apply_frequency_encode(df, col)
    elif t == "target_encode":
        return apply_target_encode(df, col, payload.get("target"))
    elif t == "equal_width":
        return apply_equal_width_bin(df, col, payload.get("bins", 5), payload.get("new_column"))
    elif t == "equal_freq":
        return apply_equal_freq_bin(df, col, payload.get("bins", 5), payload.get("new_column"))
    elif t == "custom_bins":
        return apply_custom_bins(
            df, col,
            payload.get("thresholds", []),
            payload.get("labels", []),
            payload.get("new_column"),
        )
    elif t == "polynomial":
        return apply_polynomial(df, col)
    elif t == "interaction":
        return apply_interaction(df, col, payload.get("col_b"))
    elif t == "formula":
        return apply_formula(df, payload.get("formula"), payload.get("new_column", "derived"))
    elif t == "lag":
        return apply_lag(df, col, payload.get("periods", 1))
    elif t == "rolling_mean":
        return apply_rolling_mean(df, col, payload.get("window", 3))
    elif t == "extract_year":
        return apply_extract_year(df, col)
    elif t == "extract_month":
        return apply_extract_month(df, col)
    elif t == "extract_day":
        return apply_extract_day(df, col)
    elif t == "extract_dow":
        return apply_extract_dow(df, col)
    elif t == "extract_hour":
        return apply_extract_hour(df, col)
    elif t == "pca":
        return apply_pca(df, payload.get("columns", [col]), payload.get("prefix", "pca"))
    elif t == "exclude_ml":
        return df  # No mutation — logged in history only
    else:
        raise ValueError(f"Unknown transform_type: '{t}'")













