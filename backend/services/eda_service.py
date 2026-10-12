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
from core.gemini_client import generate_json, render_prompt

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



# 6 Time-Based EDA

def time_based_eda(
    df: DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "M",
) -> dict:
    """
    Time-series EDA for a date + value column pair.
    freq: D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly
    Returns trend, aggregations, growth rate, rolling stats,
    peak/low detection, time gap detection.
    """
    for col in [date_col, value_col]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found."}

    df_t = df[[date_col, value_col]].copy()
    df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce", format="mixed")
    df_t = df_t.dropna(subset=[date_col, value_col])
    df_t = df_t.sort_values(date_col)

    if len(df_t) < 2:
        return {"error": "Not enough valid date-value pairs for time analysis."}

    # Aggregation
    freq_map = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}
    resample_freq = freq_map.get(freq.upper(), "ME")

    agg = df_t.set_index(date_col)[value_col].resample(resample_freq).agg(
        ["sum", "mean", "count"]
    ).reset_index()
    agg.columns = ["date", "sum", "mean", "count"]
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")

    trend_data = [
        {
            "date": row["date"],
            "sum": round(float(row["sum"]), 4),
            "mean": round(float(row["mean"]), 4),
            "count": int(row["count"]),
        }
        for _, row in agg.iterrows()
    ]

    # Growth rate (period-over-period)
    sums = agg["sum"].tolist()
    growth_rates = []
    for i in range(1, len(sums)):
        if sums[i - 1] != 0:
            gr = round((sums[i] - sums[i - 1]) / abs(sums[i - 1]) * 100, 2)
        else:
            gr = None
        growth_rates.append({"period": agg["date"].iloc[i], "growth_pct": gr})

    # Rolling mean (3-period)
    rolling = (
        df_t.set_index(date_col)[value_col]
        .resample(resample_freq)
        .mean()
        .rolling(3, min_periods=1)
        .mean()
        .reset_index()
    )
    rolling.columns = ["date", "rolling_mean"]
    rolling["date"] = rolling["date"].dt.strftime("%Y-%m-%d")
    rolling_data = [
        {"date": row["date"], "rolling_mean": round(float(row["rolling_mean"]), 4)}
        for _, row in rolling.iterrows()
    ]

    # Peak and low detection
    agg_sums = agg.set_index("date")["sum"]
    peak_period = str(agg_sums.idxmax())
    low_period = str(agg_sums.idxmin())

    # Time gap detection
    dates_sorted = df_t[date_col].sort_values()
    diffs = dates_sorted.diff().dropna()
    median_gap = diffs.median()
    large_gaps = diffs[diffs > median_gap * 3]
    gap_count = len(large_gaps)

    # Overall trend direction
    if len(sums) >= 2:
        first_half = np.mean(sums[: len(sums) // 2])
        second_half = np.mean(sums[len(sums) // 2 :])
        if second_half > first_half * 1.05:
            trend_direction = "upward"
        elif second_half < first_half * 0.95:
            trend_direction = "downward"
        else:
            trend_direction = "stable"
    else:
        trend_direction = "insufficient data"

    return {
        "date_col": date_col,
        "value_col": value_col,
        "freq": freq.upper(),
        "n_records": len(df_t),
        "date_range": {
            "min": str(df_t[date_col].min().date()),
            "max": str(df_t[date_col].max().date()),
            "total_days": int((df_t[date_col].max() - df_t[date_col].min()).days),
        },
        "trend_data": trend_data,
        "growth_rates": growth_rates,
        "rolling_mean": rolling_data,
        "peak_period": peak_period,
        "low_period": low_period,
        "trend_direction": trend_direction,
        "time_gaps": {
            "large_gap_count": gap_count,
            "median_gap_days": round(float(median_gap.days), 1) if hasattr(median_gap, "days") else None,
        },
    }

# 7 Multivariate Analysis

def multivariate_analysis(df: DataFrame, columns: list[str] = None) -> dict:
    """
    Correlation heatmap + pairwise relationships for numerical columns.
    Caps at 20 columns for performance.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        num_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    num_cols = num_cols[:20]

    if len(num_cols) < 2:
        return {"error": "Need at least 2 numerical columns for multivariate analysis."}

    sub = df[num_cols].dropna()

    # Correlation matrix
    corr = sub.corr()
    heatmap_data = []
    for i, row_col in enumerate(corr.columns):
        for j, col_col in enumerate(corr.columns):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                heatmap_data.append({
                    "row": row_col,
                    "col": col_col,
                    "value": round(float(val), 4),
                })

    # Strong pairs (|r| > 0.7, excluding diagonal)
    strong_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr.iloc[i, j]
            if not np.isnan(r) and abs(r) > 0.7:
                strong_pairs.append({
                    "col1": num_cols[i],
                    "col2": num_cols[j],
                    "correlation": round(float(r), 4),
                })

    # Pairwise summary
    pairwise = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr.iloc[i, j]
            if not np.isnan(r):
                abs_r = abs(r)
                strength = (
                    "strong" if abs_r >= 0.7
                    else "moderate" if abs_r >= 0.4
                    else "weak" if abs_r >= 0.2
                    else "negligible"
                )
                pairwise.append({
                    "col1": num_cols[i],
                    "col2": num_cols[j],
                    "r": round(float(r), 4),
                    "strength": strength,
                    "direction": "positive" if r > 0 else "negative",
                })

    return {
        "columns_analysed": num_cols,
        "n_columns": len(num_cols),
        "heatmap_data": heatmap_data,
        "strong_pairs": strong_pairs,
        "pairwise_summary": pairwise,
        "correlation_matrix": {
            col: {c: round(float(corr.loc[col, c]), 4) for c in num_cols}
            for col in num_cols
        },
    }




# 8 Outlier Investigation


def outlier_investigation(df:DataFrame, column: str)->dict:
    """
    Detailed outlier analysis for a numerical column.
    IQR + Z-score methods, extreme value investigation,
    outlier contribution to mean/sum.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].dropna()
    if len(s) < 4:
        return {"error": "Not enough values for outlier analysis."}


    # IQR method
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))

    iqr = q3 - q1

    lower_iqr = q1 - 1.5*iqr
    upper_iqr = q3 + 1.5*iqr

    iqr_outliers = s[(s < lower_iqr) | (s > upper_iqr)]

    # Z Sc0re Method
    z_scores = np.abs(scipy_stats.zscore(s))
    z_score_outlier_mask = z_scores > 3
    zscore_outliers = s[z_score_outlier_mask]

    # Extreme values (top/bottom 5)
    extreme_high = s.nlargest(5).tolist()
    extreme_low = s.nsmallest(5).tolist()

    # Outlier contribution
    total_sum = float(s.sum())
    outlier_sum = float(iqr_outliers.sum())
    contribution_pct = round(outlier_sum / total_sum * 100, 2) if total_sum != 0 else 0

    mean_with = round(float(s.mean()), 4)
    mean_without = round(float(s[~((s < lower_iqr) | (s > upper_iqr))].mean()), 4)
    mean_impact = round(abs(mean_with - mean_without), 4)

    return {
        "column": column,
        "total_values": len(s),
        "iqr_method": {
            "lower_bound": round(lower_iqr, 4),
            "upper_bound": round(upper_iqr, 4),
            "outlier_count": int(len(iqr_outliers)),
            "outlier_pct": round(len(iqr_outliers) / len(s) * 100, 2),
            "outlier_sample": [round(float(v), 4) for v in iqr_outliers.head(10).tolist()],
        },
        "zscore_method": {
            "threshold": 3.0,
            "outlier_count": int(len(zscore_outliers)),
            "outlier_pct": round(len(zscore_outliers) / len(s) * 100, 2),
            "outlier_sample": [round(float(v), 4) for v in zscore_outliers.head(10).tolist()],
        },
        "extreme_values": {
            "top_5": [round(float(v), 4) for v in extreme_high],
            "bottom_5": [round(float(v), 4) for v in extreme_low],
        },
        "outlier_contribution": {
            "sum_contribution_pct": contribution_pct,
            "mean_with_outliers": mean_with,
            "mean_without_outliers": mean_without,
            "mean_impact": mean_impact,
        },
    }



# 9 Auto Insights

def _generate_rule_based_eda_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Skewness insights
    for col in num_cols[:5]:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        skew = float(s.skew())
        if abs(skew) > 1:
            direction = "right" if skew > 0 else "left"
            insights.append(
                f"'{col}' is highly {direction}-skewed (skewness={skew:.2f}) — "
                f"consider a log or power transform before modelling."
            )

    # Categorical dominance
    for col in cat_cols[:3]:
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] >= 0.75:
            insights.append(
                f"'{vc.index[0]}' dominates '{col}' with "
                f"{vc.iloc[0]*100:.1f}% of records."
            )

    # Strong correlations
    if len(num_cols) >= 2:
        sub = df[num_cols[:10]].dropna()
        if len(sub) > 3:
            corr = sub.corr()
            for i in range(len(num_cols[:10])):
                for j in range(i + 1, len(num_cols[:10])):
                    r = corr.iloc[i, j]
                    if not np.isnan(r) and abs(r) > 0.7:
                        direction = "positive" if r > 0 else "negative"
                        insights.append(
                            f"'{num_cols[i]}' and '{num_cols[j]}' have a strong "
                            f"{direction} correlation (r={r:.2f})."
                        )

    # Missing value pattern
    missing = df.isnull().mean() * 100
    high_missing = missing[missing > 20]
    for col, pct in high_missing.items():
        insights.append(
            f"'{col}' has {pct:.1f}% missing values — "
            f"this may affect analysis reliability."
        )

    # Outlier summary
    for col in num_cols[:3]:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr_val = q3 - q1
        outlier_pct = ((s < q1 - 1.5*iqr_val) | (s > q3 + 1.5*iqr_val)).mean() * 100
        if outlier_pct > 5:
            insights.append(
                f"'{col}' has {outlier_pct:.1f}% outliers by IQR — "
                f"investigate before modelling."
            )

    return insights[:12] if insights else [
        "Dataset loaded successfully. Run individual analyses for detailed insights."
    ]

async def generate_eda_insights(df: pd.DataFrame) -> dict:
    """
    Generate EDA insights — Gemini narrative if available,
    rule-based fallback otherwise.
    """
    rule_insights = _generate_rule_based_eda_insights(df)

    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        summary = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "numerical_columns": num_cols[:10],
            "categorical_columns": cat_cols[:10],
            "missing_pcts": {
                col: round(float(df[col].isna().mean() * 100), 2)
                for col in df.columns
                if df[col].isna().mean() > 0
            },
            "rule_based_insights": rule_insights,
        }

        prompt = render_prompt(
            "eda_question.txt",
            EDA_SUMMARY=json.dumps(summary, indent=2, default=str),
            USER_QUESTION="Generate insights from this dataset summary.",
        )
        if prompt is not None:
            parsed = await generate_json(prompt)
            insights = parsed.get("insights", []) if parsed else []
            if isinstance(insights, list) and len(insights) >= 3:
                return {"insights": [str(i) for i in insights], "source": "gemini"}

    except Exception as e:
        logger.warning(f"Gemini EDA insights failed: {e}")

    return {"insights": rule_insights, "source": "rule_based"}


async def answer_question(df: pd.DataFrame, question: str) -> dict:
    """
    Answer a natural language question about the dataset.
    Gemini understands the question, picks the right analysis,
    and returns a plain English answer + supporting data.
    Fallback: keyword-based routing.
    """
    question = (question or "").strip()[:1000]
    question_lower = question.lower()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Try Gemini first
    try:
        summary = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "numerical_columns": num_cols[:15],
            "categorical_columns": cat_cols[:15],
            "datetime_columns": dt_cols,
            "sample_stats": {
                col: {
                    "mean": round(float(df[col].dropna().mean()), 4),
                    "std": round(float(df[col].dropna().std()), 4),
                }
                for col in num_cols[:5]
                if len(df[col].dropna()) > 0
            },
        }

        prompt = render_prompt(
            "eda_question.txt",
            EDA_SUMMARY=json.dumps(summary, indent=2, default=str),
            USER_QUESTION=question,
        )
        if prompt is not None:
            parsed = await generate_json(prompt)
            if parsed and parsed.get("answer"):
                return {
                    "question": question,
                    "answer": str(parsed["answer"]),
                    "suggested_analysis": parsed.get("suggested_analysis"),
                    "source": "gemini",
                }

    except Exception as e:
        logger.warning(f"Gemini question answering failed: {e}")

    # Fallback — keyword routing
    if any(kw in question_lower for kw in ["correlat", "related", "relationship"]):
        result = multivariate_analysis(df)
        answer = f"Found {len(result.get('strong_pairs', []))} strong correlations."
        suggested = "multivariate"

    elif any(kw in question_lower for kw in ["outlier", "unusual", "extreme"]):
        col = num_cols[0] if num_cols else None
        answer = f"Use the Outlier Investigation tab to explore unusual values in {col}." if col else "No numerical columns found."
        suggested = "outlier"

    elif any(kw in question_lower for kw in ["distribut", "spread", "range", "skew"]):
        col = num_cols[0] if num_cols else None
        answer = f"Use the Univariate Analysis tab to explore the distribution of {col}." if col else "No numerical columns found."
        suggested = "univariate_numerical"

    elif any(kw in question_lower for kw in ["categor", "frequenc", "common", "rare"]):
        col = cat_cols[0] if cat_cols else None
        answer = f"Use the Univariate Analysis tab to explore the categories of {col}." if col else "No categorical columns found."
        suggested = "univariate_categorical"

    elif any(kw in question_lower for kw in ["trend", "over time", "time series", "change"]):
        answer = "Use the Time-Series tab to explore trends over time."
        suggested = "time_series"

    else:
        answer = "Please use the analysis tabs to explore your data, or refine your question with specific column names."
        suggested = "general"

    return {
        "question": question,
        "answer": answer,
        "suggested_analysis": suggested,
        "source": "rule_based",
    }

# 11 Chart Recommendation

def recommend_chart(
    df: pd.DataFrame,
    col_x: str,
    col_y: str = None,
) -> dict:
    """
    Recommend the best chart type based on column dtypes.
    """
    if col_x not in df.columns:
        return {"error": f"Column '{col_x}' not found."}

    x_is_num = pd.api.types.is_numeric_dtype(df[col_x])
    x_is_dt = pd.api.types.is_datetime64_any_dtype(df[col_x])
    x_is_cat = not x_is_num and not x_is_dt

    if col_y is None:
        if x_is_num:
            chart = "histogram"
            reason = f"'{col_x}' is numerical — histogram shows its distribution."
        elif x_is_dt:
            chart = "line_chart"
            reason = f"'{col_x}' is a datetime — line chart shows trends over time."
        else:
            chart = "bar_chart"
            reason = f"'{col_x}' is categorical — bar chart shows category frequencies."
        return {"col_x": col_x, "col_y": None, "recommended_chart": chart, "reason": reason}

    if col_y not in df.columns:
        return {"error": f"Column '{col_y}' not found."}

    y_is_num = pd.api.types.is_numeric_dtype(df[col_y])
    y_is_dt = pd.api.types.is_datetime64_any_dtype(df[col_y])
    y_is_cat = not y_is_num and not y_is_dt

    if x_is_num and y_is_num:
        chart = "scatter_plot"
        reason = "Both columns are numerical — scatter plot shows their relationship."
    elif x_is_dt and y_is_num:
        chart = "line_chart"
        reason = f"'{col_x}' is datetime and '{col_y}' is numerical — line chart shows the trend."
    elif x_is_cat and y_is_num:
        chart = "box_plot"
        reason = f"'{col_x}' is categorical and '{col_y}' is numerical — box plot compares distributions across groups."
    elif x_is_num and y_is_cat:
        chart = "box_plot"
        reason = f"'{col_y}' is categorical and '{col_x}' is numerical — box plot compares distributions across groups."
    elif x_is_cat and y_is_cat:
        chart = "grouped_bar_chart"
        reason = "Both columns are categorical — grouped bar chart shows category combinations."
    else:
        chart = "bar_chart"
        reason = "Default recommendation — adjust based on your analysis goal."

    alternatives = {
        "scatter_plot": ["line_chart", "hexbin"],
        "box_plot": ["violin_plot", "bar_chart"],
        "line_chart": ["area_chart", "scatter_plot"],
        "bar_chart": ["pareto_chart", "pie_chart"],
        "grouped_bar_chart": ["stacked_bar_chart", "heatmap"],
        "histogram": ["density_plot", "box_plot"],
    }

    return {
        "col_x": col_x,
        "col_y": col_y,
        "recommended_chart": chart,
        "reason": reason,
        "alternatives": alternatives.get(chart, []),
    }


# 14 EDA Summary

def generate_eda_summary(df: pd.DataFrame, history: list[dict]) -> dict:
    """
    Generate an end-of-stage EDA summary.
    Combines auto-insights with history of analyses run.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Correlation summary
    strong_correlations = []
    if len(num_cols) >= 2:
        sub = df[num_cols[:10]].dropna()
        if len(sub) > 3:
            corr = sub.corr()
            for i in range(len(num_cols[:10])):
                for j in range(i + 1, len(num_cols[:10])):
                    r = corr.iloc[i, j]
                    if not np.isnan(r) and abs(r) > 0.7:
                        strong_correlations.append({
                            "col1": num_cols[i],
                            "col2": num_cols[j],
                            "r": round(float(r), 4),
                        })

    # Distribution summary
    skewed_cols = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) > 3 and abs(float(s.skew())) > 1:
            skewed_cols.append(col)

    # Missing value summary
    missing = {
        col: round(float(df[col].isna().mean() * 100), 2)
        for col in df.columns
        if df[col].isna().mean() > 0.05
    }

    # History summary
    analyses_run = list({h.get("analysis_type") for h in history if h.get("analysis_type")})
    questions_asked = [h.get("question") for h in history if h.get("question")]

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "column_types": {
            "numerical": len(num_cols),
            "categorical": len(cat_cols),
            "datetime": len(dt_cols),
        },
        "strong_correlations": strong_correlations,
        "skewed_columns": skewed_cols,
        "high_missing_columns": missing,
        "analyses_run": analyses_run,
        "questions_asked": questions_asked,
        "total_analyses": len(history),
    }



