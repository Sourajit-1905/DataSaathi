"""
EDA Service
-----------
Implements all 14 EDA sub-modules from the full spec:

  E1  Numerical Univariate
  E2  Categorical Univariate
  E3  Numerical vs Numerical (Bivariate)
  E4  Numerical vs Categorical
  E5  Categorical vs Categorical
  E6  Time-Based EDA
  E7  Multivariate Analysis
  E8  Outlier Investigation
  E9  Auto Insights (algorithmic + Gemini narrative)
  E10 Natural Language Question (Gemini-powered)
  E11 Chart Recommendation
  E12 Filter Engine (up to 5 filters, read-only)
  E13 EDA History
  E14 EDA Summary

All functions are READ-ONLY — working CSV is never modified.
Filters are applied in memory before every analysis.
"""


import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.config import settings

logger = logging.getLogger(__name__)


# 12 - Filter Engine

FILTER_OPERATORS = {
    "eq": lambda s, v: s == v,
    "neq": lambda s, v: s != v,
    "gt": lambda s, v: s > float(v),
    "gte": lambda s, v: s >= float(v),
    "lt": lambda s, v: s < float(v),
    "lte": lambda s, v: s <= float(v),
    "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
    "in": lambda s, v: s.isin(v if isinstance(v, list) else [v]),
    "notnull": lambda s, v: s.notna(),
    "isnull": lambda s, v: s.isna(),
}


def apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    """
    Apply up to 5 filters to a DataFrame.
    Each filter: { "column": str, "operator": str, "value": any }
    Returns filtered DataFrame — original is never modified.
    """
    if not filters:
        return df

    filtered = df.copy()
    for f in filters[:5]:
        col = f.get("column")
        op = f.get("operator", "eq")
        val = f.get("value")

        if col not in filtered.columns:
            logger.warning(f"Filter column '{col}' not found — skipping.")
            continue

        if op not in FILTER_OPERATORS:
            logger.warning(f"Unknown filter operator '{op}' — skipping.")
            continue

        try:
            mask = FILTER_OPERATORS[op](filtered[col], val)
            filtered = filtered[mask]
        except Exception as e:
            logger.warning(f"Filter failed for col={col} op={op}: {e}")

    return filtered

from pandas import DataFrame









# 1 - Numerical Univariate


def numerical_univariate(df: DataFrame, column: str) -> dict:
    """
    Full univariate analysis for a numerical column.
    Returns stats, percentiles, distribution label,
    histogram data, box plot stats, density estimate.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].dropna()
    if len(s) == 0:
        return {"error": f"Column '{column}' has no non-null values."}

    # Core Stats
    mean = float(s.mean())
    median = float(s.median())

    try: mode = float(s.mode().iloc[0])
    except Exception: mode = None

    variance = float(s.var())
    std = float(s.std())

    mini = float(s.min())
    maxi = float(s.max())

    val_range = maxi - mini

    skewness = float(s.skew())
    kurt = float(s.kurtosis())
    iqr = float(s.quantile(0.75) - s.quantile(0.25))

    
    # Percentiles
    percentiles = {
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
    }



    # Distribution label
    if abs(skewness) < 0.5:
        dist_label = "symmetric"
    elif abs(skewness) <= 1.0:
        dist_label = "moderately skewed"
    else:
        dist_label = "highly skewed"

    if kurt > 3:
        dist_label += " (heavy-tailed)"
    elif kurt < -1:
        dist_label += " (light-tailed)"


    # Histogram data (20 bins)
    counts, bin_edges = np.histogram(s.dropna(), bins=20)
    histogram = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 4) for e in bin_edges.tolist()],
        "bin_centers": [
            round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 4)
            for i in range(len(bin_edges) - 1)
        ],
    }

    # Box plot stats
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr_val = q3 - q1
    lower_fence = q1 - 1.5 * iqr_val
    upper_fence = q3 + 1.5 * iqr_val
    outliers = s[(s < lower_fence) | (s > upper_fence)].tolist()

    box_plot = {
        "min": mini,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": maxi,
        "lower_fence": round(lower_fence, 4),
        "upper_fence": round(upper_fence, 4),
        "outlier_count": len(outliers),
        "outlier_sample": [round(float(o), 4) for o in outliers[:10]],
    }


    # Density estimate (KDE) — 100 points
    try:
        kde = scipy_stats.gaussian_kde(s)
        x_range = np.linspace(mini, maxi, 100)
        density = {
            "x": [round(float(v), 4) for v in x_range.tolist()],
            "y": [round(float(v), 6) for v in kde(x_range).tolist()],
        }
    except Exception:
        density = None

    return {
        "column": column,
        "count": len(s),
        "missing": int(df[column].isna().sum()),
        "mean": round(mean, 4),
        "median": round(median, 4),
        "mode": round(mode, 4) if mode is not None else None,
        "variance": round(variance, 4),
        "std": round(std, 4),
        "min": round(mini, 4),
        "max": round(maxi, 4),
        "range": round(val_range, 4),
        "iqr": round(iqr, 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurt, 4),
        "distribution_label": dist_label,
        "percentiles": {k: round(v, 4) for k, v in percentiles.items()},
        "histogram": histogram,
        "box_plot": box_plot,
        "density": density,
    }

# 2. Categorical Univariate

def categorical_univariate(df: DataFrame, column: str) -> dict:
    """
    Full univariate analysis for a categorical column.
    Returns frequency table, proportions, pareto data,
    rare category detection, most/least common.
    """

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].dropna()
    if len(s) == 0:
        return {"error": f"Column '{column}' has no non-null values."}

    vc = s.value_counts()
    vc_pct = s.value_counts(normalize=True) * 100

    # Frequency table
    freq_table = [
        {
            "category": str(cat),
            "count": int(vc[cat]),
            "pct": round(float(vc_pct[cat]), 2),
        }
        for cat in vc.index
    ]

    # Pareto — cumulative %
    cumulative = 0.0
    pareto = []

    for row in freq_table:
        cumulative += row["pct"]

        pareto.append({
            "category": row["category"],
            "count": row["count"],
            "pct": row["pct"],
            "cumulative_pct": round(cumulative, 2),
        })



    # Rare categories (< 1% of total)
    rare_threshold = 1.0
    rare = [r for r in freq_table if r["pct"] < rare_threshold]



    # 80/20 analysis — how many categories make up 80%
    cats_for_80 = 0
    cum = 0.0

    for row in freq_table:
        cum += row["pct"]
        cats_for_80 += 1

        if cum >=80 : break

    return {
        "column": column,
        "count": len(s),
        "missing": int(df[column].isna().sum()),
        "unique_count": int(s.nunique()),
        "most_common": freq_table[:5],
        "least_common": freq_table[-5:],
        "frequency_table": freq_table,
        "pareto": pareto,
        "rare_categories": rare,
        "rare_threshold_pct": rare_threshold,
        "cats_for_80pct": cats_for_80,
    }



# 3. Numerical vs Numerical
def numerical_vs_numerical(df: DataFrame, col_x: str, col_y:str) ->dict:
    """
    Bivariate analysis between two numerical columns.
    Returns Pearson + Spearman correlations, scatter data,
    trend detection, regression line.
    """
    for col in [col_x, col_y]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found."}

    pair = df[[col_x, col_y]].dropna()

    if(len(pair)) < 3:
        return {"error": "Not enough non-null pairs for correlation."}

    x = pair[col_x]
    y = pair[col_y]


    # Pearson
    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)

    # Spearman
    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)

    # Relationship strength label
    abs_r = abs(pearson_r)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "positive" if pearson_r > 0 else "negative"


    # Regression line (OLS)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    x_line = [float(x.min()), float(x.max())]
    y_line = [round(slope * xi + intercept, 4) for xi in x_line]

    # Scatter data — sample up to 500 points for performance
    sample = pair.sample(min(500, len(pair)), random_state=42)
    scatter = {
        "x": sample[col_x].tolist(),
        "y": sample[col_y].tolist(),
    }

    # Trend detection
    if p_value < 0.05 and abs_r >= 0.4:
        trend = f"{strength.capitalize()} {direction} linear trend detected."
    elif spearman_p < 0.05 and abs(spearman_r) >= 0.4:
        trend = f"Monotonic {direction} relationship detected (non-linear)."
    else:
        trend = "No significant linear trend detected."

    return {
        "col_x": col_x,
        "col_y": col_y,
        "n_pairs": len(pair),
        "pearson": {
            "r": round(float(pearson_r), 4),
            "p_value": round(float(pearson_p), 6),
            "significant": bool(pearson_p < 0.05),
        },
        "spearman": {
            "r": round(float(spearman_r), 4),
            "p_value": round(float(spearman_p), 6),
            "significant": bool(spearman_p < 0.05),
        },
        "relationship": {
            "strength": strength,
            "direction": direction,
            "r_squared": round(float(r_value ** 2), 4),
        },
        "regression_line": {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "x": x_line,
            "y": y_line,
        },
        "scatter_sample": scatter,
        "trend_summary": trend,
    }




# 4. Numerical vs Categorical

def numerical_vs_categorical(df: DataFrame, num_col:str, cat_col: str) -> dict:
    """
    Compare a numerical variable across categories.
    Returns group-wise stats, box plot data, distribution comparison,
    group difference detection.
    """
    for col in [num_col, cat_col]:
        if col not in df.columns: return {"error": f"Column '{col}' not found."}

    pair = df[[num_col, cat_col]].dropna()
    groups = pair.groupby(cat_col)[num_col]

    group_stats = {}
    box_data = {}

    for name, g in groups:
        g = g.dropna()
        if len(g) == 0: continue

        q1 = float(g.quantile(0.25))
        q3 = float(g.quantile(0.75))

        iqr_val = q3 - q1

        group_stats[str(name)] = {
            "count": len(g),
            "mean": round(float(g.mean()), 4),
            "median": round(float(g.median()), 4),
            "std": round(float(g.std()), 4),
            "min": round(float(g.min()), 4),
            "max": round(float(g.max()), 4),
            "p25": round(q1, 4),
            "p75": round(q3, 4),
            "p90": round(float(g.quantile(0.90)), 4),
        }

        box_data[str(name)] = {
            "q1": q1,
            "median": round(float(g.median()), 4),
            "q3": q3,
            "lower_fence": round(q1 - 1.5 * iqr_val, 4),
            "upper_fence": round(q3 + 1.5 * iqr_val, 4),
            "outlier_count": int(((g < q1 - 1.5 * iqr_val) | (g > q3 + 1.5 * iqr_val)).sum()),
        }

    # Group difference detection
    means = {k: v["mean"] for k, v in group_stats.items()}
    if len(means) >= 2:
        max_group = max(means, key=means.get)
        min_group = min(means, key=means.get)
        diff = means[max_group] - means[min_group]
        overall_std = float(pair[num_col].std())
        effect = round(diff / overall_std, 4) if overall_std > 0 else 0
        difference_summary = (
            f"'{max_group}' has the highest mean {num_col} "
            f"({means[max_group]:.2f}) vs '{min_group}' "
            f"({means[min_group]:.2f}). Effect size: {effect:.2f}."
        )
    else:
        difference_summary = "Not enough groups to compare."


    
    # Violin data — distribution per group (sample)
    violin_data = {}
    for name, g in groups:
        g_clean = g.dropna()
        sample = g_clean.sample(min(200, len(g_clean)), random_state=42).tolist()
        violin_data[str(name)] = sample

    return {
        "num_col": num_col,
        "cat_col": cat_col,
        "group_stats": group_stats,
        "box_plot_by_category": box_data,
        "violin_data": violin_data,
        "difference_summary": difference_summary,
        "group_count": len(group_stats),
    }



# 5. Categorical vs Categorical

def categorical_vs_categorical(df:DataFrame, col_a:str, col_b:str) ->str:
    """
    Investigate relationship between two categorical variables.
    Returns contingency table, proportions, chi-square recommendation.
    """
    for col in [col_a, col_b]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found."}

    pair = df[[col_a, col_b]].dropna()

    # Contingency table
    ct = pd.crosstab(pair[col_a], pair[col_b])
    ct_dict = {
        str(idx): {str(col): int(val) for col, val in row.items()}
        for idx, row in ct.iterrows()
    }

    # Row proportions
    ct_pct = pd.crosstab(pair[col_a], pair[col_b], normalize="index") * 100
    ct_pct_dict = {
        str(idx): {str(col): round(float(val), 2) for col, val in row.items()}
        for idx, row in ct_pct.iterrows()
    }

    # Chi-square test
    try:
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(ct)
        chi2_result = {
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "dof": int(dof),
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"Significant association between '{col_a}' and '{col_b}' (p={p_value:.4f})."
                if p_value < 0.05
                else f"No significant association between '{col_a}' and '{col_b}' (p={p_value:.4f})."
            ),
        }
    except Exception as e:
        chi2_result = {"error": str(e)}

    # Grouped bar data
    grouped_bar = [
        {
            "category_a": str(idx),
            "counts": {str(col): int(val) for col, val in row.items()},
        }
        for idx, row in ct.iterrows()
    ]

    return {
        "col_a": col_a,
        "col_b": col_b,
        "n_pairs": len(pair),
        "contingency_table": ct_dict,
        "row_proportions": ct_pct_dict,
        "chi_square": chi2_result,
        "grouped_bar_data": grouped_bar,
        "unique_a": int(pair[col_a].nunique()),
        "unique_b": int(pair[col_b].nunique()),
    }





