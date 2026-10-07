"""
Stats Service
-------------
Implements 21 statistical test functions:

  PARAMETRIC       S1  One-Sample t-test
                   S2  Independent Samples t-test
                   S3  Paired Samples t-test
                   S4  One-Way ANOVA
                   S5  Two-Way ANOVA
                   S6  Repeated Measures ANOVA (approximation)
                   S7  Pearson Correlation
                   S8  Linear Regression (simple)
                   S9  Multiple Regression

  NON-PARAMETRIC   S10 Mann-Whitney U Test
                   S11 Wilcoxon Signed-Rank Test
                   S12 Kruskal-Wallis Test
                   S13 Spearman Correlation
                   S14 Kendall's Tau

  CATEGORICAL      S15 Chi-Square Test of Independence
                   S16 Chi-Square Goodness of Fit
                   S17 Fisher's Exact Test

  DISTRIBUTION     S18 Shapiro-Wilk Normality Test
                   S19 Kolmogorov-Smirnov Test
                   S20 Levene's Test (Homogeneity of Variance)
                   S21 Bartlett's Test

Each test returns: statistic, p_value, effect_size, interpretation,
assumption_checks, and plain_english (Gemini or rule-based).
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.config import settings
from core.exceptions import ColumnNotFoundError

logger = logging.getLogger(__name__)


# Gemini plain English explanation

async def _get_gemini_explanation(test_name: str, result: dict) -> str | None:
    try:
        from google import genai as google_genai

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            return None

        prompt_path = Path("prompts/stats_explain.txt")
        if not prompt_path.exists():
            return None

        prompt_template = prompt_path.read_text(encoding="utf-8")
        payload = {
            "test_name": test_name,
            "statistic": result.get("statistic"),
            "p_value": result.get("p_value"),
            "effect_size": result.get("effect_size"),
            "interpretation": result.get("interpretation"),
            "assumption_checks": result.get("assumption_checks"),
        }
        prompt = prompt_template.replace(
            "{{TEST_RESULT}}", json.dumps(payload, indent=2)
        )

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
        explanation = parsed.get("plain_english", "")
        if explanation and len(explanation) > 10:
            return explanation

    except Exception as e:
        logger.warning(f"Gemini stats explanation failed for {test_name}: {e}")

    return None


def _rule_based_explanation(test_name: str, result: dict) -> str:
    p = result.get("p_value", 1.0)
    stat = result.get("statistic", 0)
    effect = result.get("effect_size", {})
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    effect_label = effect.get("label", "unknown") if isinstance(effect, dict) else "unknown"

    explanations = {
        "one_sample_t": f"The one-sample t-test result is {sig} (t={stat:.3f}, p={p:.4f}). The effect size is {effect_label}.",
        "independent_t": f"The difference between groups is {sig} (t={stat:.3f}, p={p:.4f}). The effect size is {effect_label}, suggesting a {effect_label} practical difference.",
        "paired_t": f"The paired comparison is {sig} (t={stat:.3f}, p={p:.4f}). The effect size is {effect_label}.",
        "one_way_anova": f"The one-way ANOVA is {sig} (F={stat:.3f}, p={p:.4f}). The effect size (η²) is {effect_label}.",
        "pearson": f"The Pearson correlation is {sig} (r={stat:.3f}, p={p:.4f}). The relationship strength is {effect_label}.",
        "linear_regression": f"The linear regression is {sig} (F={stat:.3f}, p={p:.4f}). The model explains {effect.get('r_squared', 0)*100:.1f}% of variance." if isinstance(effect, dict) else f"Linear regression result: {sig}.",
        "mann_whitney": f"The Mann-Whitney U test is {sig} (U={stat:.3f}, p={p:.4f}). The effect size (r) is {effect_label}.",
        "wilcoxon": f"The Wilcoxon signed-rank test is {sig} (W={stat:.3f}, p={p:.4f}).",
        "kruskal_wallis": f"The Kruskal-Wallis test is {sig} (H={stat:.3f}, p={p:.4f}).",
        "spearman": f"The Spearman correlation is {sig} (ρ={stat:.3f}, p={p:.4f}). The relationship is {effect_label}.",
        "chi_square": f"The chi-square test is {sig} (χ²={stat:.3f}, p={p:.4f}). The effect size (Cramér's V) is {effect_label}.",
        "shapiro_wilk": f"The Shapiro-Wilk normality test is {sig} (W={stat:.3f}, p={p:.4f}). {'The data appears normally distributed.' if p >= 0.05 else 'The data deviates significantly from normality.'}",
        "levene": f"Levene's test for equal variances is {sig} (F={stat:.3f}, p={p:.4f}). {'Variances are equal across groups.' if p >= 0.05 else 'Variances differ significantly across groups.'}",
    }
    return explanations.get(
        test_name,
        f"The {test_name} test result is {sig} (statistic={stat:.3f}, p={p:.4f})."
    )


# Assumption Checkers

def check_normality(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 3:
        return {"passed": False, "reason": "Too few observations for normality test."}
    if len(s) > 5000:
        s = s.sample(5000, random_state=42)
    stat, p = scipy_stats.shapiro(s)
    passed = p >= 0.05
    return {
        "test": "shapiro_wilk",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "passed": passed,
        "reason": "Normal" if passed else f"Non-normal (p={p:.4f})",
    }


def check_equal_variance(*groups: pd.Series) -> dict:
    clean_groups = [g.dropna() for g in groups if len(g.dropna()) >= 2]
    if len(clean_groups) < 2:
        return {"passed": False, "reason": "Need at least 2 groups."}
    stat, p = scipy_stats.levene(*clean_groups)
    passed = p >= 0.05
    return {
        "test": "levene",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "passed": passed,
        "reason": "Equal variances" if passed else f"Unequal variances (p={p:.4f})",
    }


def check_sample_size(series: pd.Series, min_n: int = 30) -> dict:
    n = len(series.dropna())
    passed = n >= min_n
    return {
        "n": n,
        "min_required": min_n,
        "passed": passed,
        "reason": f"n={n} {'≥' if passed else '<'} {min_n}",
    }


def _effect_size_label(value: float, thresholds: tuple) -> str:
    small, medium = thresholds
    abs_val = abs(value)
    if abs_val < small:
        return "negligible"
    elif abs_val < medium:
        return "small"
    elif abs_val < 0.8:
        return "medium"
    else:
        return "large"


# Test Recommender

def recommend_test(
    df: pd.DataFrame,
    question: str,
    columns: list[str],
) -> dict:
    """
    Recommend the appropriate statistical test based on:
    - Number of columns
    - Column dtypes
    - Sample sizes
    - Normality checks
    - Research question keywords
    """
    if not columns:
        return {"error": "No columns provided for test recommendation."}

    col_types = {}
    for col in columns:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found."}
        col_types[col] = "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

    num_cols = [c for c in columns if col_types[c] == "numerical"]
    cat_cols = [c for c in columns if col_types[c] == "categorical"]

    question_lower = question.lower()

    # Normality checks for numerical columns
    normality = {}
    for col in num_cols[:2]:
        normality[col] = check_normality(df[col])

    all_normal = all(v["passed"] for v in normality.values())

    # Recommendation logic
    if len(num_cols) == 1 and len(cat_cols) == 0:
        # One numerical — compare to known value or test normality
        if any(kw in question_lower for kw in ["normal", "distribut"]):
            recommended = "shapiro_wilk"
            rationale = "Testing whether a single numerical variable follows a normal distribution."
        else:
            recommended = "one_sample_t"
            rationale = "Comparing a single numerical variable to a hypothesised mean."

    elif len(num_cols) == 2 and len(cat_cols) == 0:
        # Two numerical — correlation or regression
        if any(kw in question_lower for kw in ["correlat", "related", "relationship"]):
            if all_normal:
                recommended = "pearson"
                rationale = "Both variables are numerical and approximately normal — Pearson correlation is appropriate."
            else:
                recommended = "spearman"
                rationale = "Data is non-normal — Spearman correlation is more robust."
        else:
            recommended = "linear_regression"
            rationale = "Predicting one numerical variable from another."

    elif len(num_cols) == 1 and len(cat_cols) == 1:
        # One numerical, one categorical — group comparison
        n_groups = df[cat_cols[0]].nunique()
        if n_groups == 2:
            if all_normal:
                recommended = "independent_t"
                rationale = "Comparing means of two independent groups with normal data."
            else:
                recommended = "mann_whitney"
                rationale = "Non-normal data — Mann-Whitney U is the non-parametric alternative."
        else:
            if all_normal:
                recommended = "one_way_anova"
                rationale = f"Comparing means across {n_groups} groups with normal data."
            else:
                recommended = "kruskal_wallis"
                rationale = f"Non-normal data — Kruskal-Wallis is the non-parametric alternative for {n_groups} groups."

    elif len(num_cols) == 0 and len(cat_cols) == 2:
        # Two categorical — chi-square
        min_group = min(df[cat_cols[0]].nunique(), df[cat_cols[1]].nunique())
        if min_group == 2 and df[cat_cols[0]].nunique() == 2:
            recommended = "fisher_exact"
            rationale = "Both variables are binary — Fisher's Exact Test is appropriate for small samples."
        else:
            recommended = "chi_square_independence"
            rationale = "Testing association between two categorical variables."

    elif len(num_cols) >= 2 and len(cat_cols) == 0:
        recommended = "multiple_regression"
        rationale = "Multiple numerical predictors — multiple regression is appropriate."

    else:
        recommended = "pearson"
        rationale = "Default recommendation — review column types and refine the question."

    # Assumption summary
    assumptions = {
        "normality": normality,
        "sample_sizes": {
            col: check_sample_size(df[col]) for col in num_cols[:2]
        },
    }

    return {
        "recommended_test": recommended,
        "rationale": rationale,
        "column_types": col_types,
        "assumptions": assumptions,
        "available_tests": [
            "one_sample_t", "independent_t", "paired_t",
            "one_way_anova", "two_way_anova",
            "pearson", "spearman", "kendall",
            "linear_regression", "multiple_regression",
            "mann_whitney", "wilcoxon", "kruskal_wallis",
            "chi_square_independence", "chi_square_goodness",
            "fisher_exact", "shapiro_wilk", "ks_test",
            "levene", "bartlett",
        ],
    }


# S1: One-Sample t-test

async def one_sample_t(
    df: pd.DataFrame, column: str, pop_mean: float = 0.0
) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    s = df[column].dropna()
    stat, p = scipy_stats.ttest_1samp(s, pop_mean)
    d = (s.mean() - pop_mean) / s.std()

    result = {
        "test": "one_sample_t",
        "column": column,
        "n": len(s),
        "sample_mean": round(float(s.mean()), 4),
        "hypothesised_mean": pop_mean,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "cohen_d": round(float(d), 4),
            "label": _effect_size_label(d, (0.2, 0.5)),
        },
        "interpretation": (
            f"The sample mean ({s.mean():.4f}) is significantly different from {pop_mean}."
            if p < 0.05
            else f"No significant difference from {pop_mean} (p={p:.4f})."
        ),
        "assumption_checks": {
            "normality": check_normality(s),
            "sample_size": check_sample_size(s),
        },
    }

    explanation = await _get_gemini_explanation("one_sample_t", result)
    result["plain_english"] = explanation or _rule_based_explanation("one_sample_t", result)
    return result


# 2  Independent Samples t-test

async def independent_t(
    df: pd.DataFrame, num_col: str, cat_col: str
) -> dict:
    if num_col not in df.columns:
        raise ColumnNotFoundError(num_col)
    if cat_col not in df.columns:
        raise ColumnNotFoundError(cat_col)

    groups = [
        df[df[cat_col] == g][num_col].dropna()
        for g in df[cat_col].dropna().unique()[:2]
    ]
    if len(groups) < 2:
        return {"error": "Need exactly 2 groups for independent t-test."}

    g1, g2 = groups[0], groups[1]
    stat, p = scipy_stats.ttest_ind(g1, g2, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0

    group_names = df[cat_col].dropna().unique()[:2].tolist()

    result = {
        "test": "independent_t",
        "num_col": num_col,
        "cat_col": cat_col,
        "groups": {
            str(group_names[0]): {"n": len(g1), "mean": round(float(g1.mean()), 4), "std": round(float(g1.std()), 4)},
            str(group_names[1]): {"n": len(g2), "mean": round(float(g2.mean()), 4), "std": round(float(g2.std()), 4)},
        },
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "cohen_d": round(float(d), 4),
            "label": _effect_size_label(d, (0.2, 0.5)),
        },
        "interpretation": (
            f"Significant difference in {num_col} between groups (p={p:.4f})."
            if p < 0.05
            else f"No significant difference in {num_col} between groups (p={p:.4f})."
        ),
        "assumption_checks": {
            "normality_g1": check_normality(g1),
            "normality_g2": check_normality(g2),
            "equal_variance": check_equal_variance(g1, g2),
        },
    }

    explanation = await _get_gemini_explanation("independent_t", result)
    result["plain_english"] = explanation or _rule_based_explanation("independent_t", result)
    return result


# 3 Paired Samples t-test

async def paired_t(
    df: pd.DataFrame, col_before: str, col_after: str
) -> dict:
    for col in [col_before, col_after]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[col_before, col_after]].dropna()
    a = pair[col_before]
    b = pair[col_after]
    stat, p = scipy_stats.ttest_rel(a, b)
    diff = b - a
    d = diff.mean() / diff.std() if diff.std() > 0 else 0

    result = {
        "test": "paired_t",
        "col_before": col_before,
        "col_after": col_after,
        "n_pairs": len(pair),
        "mean_before": round(float(a.mean()), 4),
        "mean_after": round(float(b.mean()), 4),
        "mean_difference": round(float(diff.mean()), 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "cohen_d": round(float(d), 4),
            "label": _effect_size_label(d, (0.2, 0.5)),
        },
        "interpretation": (
            f"Significant change from {col_before} to {col_after} (p={p:.4f})."
            if p < 0.05
            else f"No significant change (p={p:.4f})."
        ),
        "assumption_checks": {
            "normality_diff": check_normality(diff),
        },
    }

    explanation = await _get_gemini_explanation("paired_t", result)
    result["plain_english"] = explanation or _rule_based_explanation("paired_t", result)
    return result


# 4 One-Way ANOVA

async def one_way_anova(
    df: pd.DataFrame, num_col: str, cat_col: str
) -> dict:
    if num_col not in df.columns:
        raise ColumnNotFoundError(num_col)
    if cat_col not in df.columns:
        raise ColumnNotFoundError(cat_col)

    groups = [
        g.dropna()
        for _, g in df.groupby(cat_col)[num_col]
        if len(g.dropna()) >= 2
    ]
    if len(groups) < 2:
        return {"error": "Need at least 2 groups with ≥2 observations each."}

    stat, p = scipy_stats.f_oneway(*groups)

    # Eta-squared
    grand_mean = df[num_col].dropna().mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((v - grand_mean)**2 for g in groups for v in g)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    # Post-hoc Tukey HSD
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(
            df[num_col].dropna(),
            df.loc[df[num_col].notna(), cat_col],
        )
        posthoc = [
            {
                "group1": str(r[0]),
                "group2": str(r[1]),
                "mean_diff": round(float(r[2]), 4),
                "p_adj": round(float(r[3]), 6),
                "significant": bool(r[6]),
            }
            for r in tukey.summary().data[1:]
        ]
    except Exception:
        posthoc = []

    group_stats = {
        str(name): {
            "n": len(g),
            "mean": round(float(g.mean()), 4),
            "std": round(float(g.std()), 4),
        }
        for name, g in df.groupby(cat_col)[num_col]
        if len(g.dropna()) >= 2
    }

    result = {
        "test": "one_way_anova",
        "num_col": num_col,
        "cat_col": cat_col,
        "n_groups": len(groups),
        "group_stats": group_stats,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "eta_squared": round(float(eta_sq), 4),
            "label": _effect_size_label(eta_sq, (0.01, 0.06)),
        },
        "posthoc_tukey": posthoc,
        "interpretation": (
            f"Significant difference in {num_col} across {len(groups)} groups (p={p:.4f})."
            if p < 0.05
            else f"No significant difference across groups (p={p:.4f})."
        ),
        "assumption_checks": {
            "equal_variance": check_equal_variance(*groups),
            "normality_overall": check_normality(df[num_col].dropna()),
        },
    }

    explanation = await _get_gemini_explanation("one_way_anova", result)
    result["plain_english"] = explanation or _rule_based_explanation("one_way_anova", result)
    return result


# S5 Two-Way ANOVA

async def two_way_anova(
    df: pd.DataFrame, num_col: str, cat_col1: str, cat_col2: str
) -> dict:
    try:
        import statsmodels.formula.api as smf

        for col in [num_col, cat_col1, cat_col2]:
            if col not in df.columns:
                raise ColumnNotFoundError(col)

        clean = df[[num_col, cat_col1, cat_col2]].dropna()
        formula = f"Q('{num_col}') ~ C(Q('{cat_col1}')) + C(Q('{cat_col2}')) + C(Q('{cat_col1}')):C(Q('{cat_col2}'))"
        model = smf.ols(formula, data=clean).fit()

        import statsmodels.api as sm
        anova_table = sm.stats.anova_lm(model, typ=2)

        effects = {}
        for idx in anova_table.index:
            row = anova_table.loc[idx]
            effects[str(idx)] = {
                "F": round(float(row.get("F", 0)), 4),
                "p_value": round(float(row.get("PR(>F)", 1)), 6),
                "significant": bool(row.get("PR(>F)", 1) < 0.05),
            }

        result = {
            "test": "two_way_anova",
            "num_col": num_col,
            "cat_col1": cat_col1,
            "cat_col2": cat_col2,
            "effects": effects,
            "statistic": round(float(anova_table["F"].max()), 4),
            "p_value": round(float(anova_table["PR(>F)"].min()), 6),
            "significant": bool(anova_table["PR(>F)"].min() < 0.05),
            "effect_size": {"label": "see_effects_table"},
            "interpretation": "See effects table for individual factor and interaction results.",
            "assumption_checks": {
                "normality": check_normality(clean[num_col]),
            },
        }

        explanation = await _get_gemini_explanation("two_way_anova", result)
        result["plain_english"] = explanation or _rule_based_explanation("two_way_anova", result)
        return result

    except Exception as e:
        return {"error": f"Two-way ANOVA failed: {str(e)}"}


# S6  Repeated Measures ANOVA (approximation via paired t-tests)

async def repeated_measures_anova(
    df: pd.DataFrame, columns: list[str]
) -> dict:
    for col in columns:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    clean = df[columns].dropna()
    if len(clean) < 3:
        return {"error": "Not enough observations for repeated measures ANOVA."}

    # Friedman test (non-parametric RM ANOVA)
    stat, p = scipy_stats.friedmanchisquare(*[clean[c] for c in columns])

    group_means = {col: round(float(clean[col].mean()), 4) for col in columns}

    result = {
        "test": "repeated_measures_anova",
        "columns": columns,
        "n": len(clean),
        "group_means": group_means,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {"label": "see_group_means"},
        "interpretation": (
            f"Significant difference across {len(columns)} repeated measures (p={p:.4f})."
            if p < 0.05
            else f"No significant difference across repeated measures (p={p:.4f})."
        ),
        "assumption_checks": {
            "normality": check_normality(clean[columns[0]]),
        },
    }

    explanation = await _get_gemini_explanation("repeated_measures_anova", result)
    result["plain_english"] = explanation or _rule_based_explanation("repeated_measures_anova", result)
    return result


# S7  Pearson Correlation

async def pearson_correlation(
    df: pd.DataFrame, col_x: str, col_y: str
) -> dict:
    for col in [col_x, col_y]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[col_x, col_y]].dropna()
    r, p = scipy_stats.pearsonr(pair[col_x], pair[col_y])

    result = {
        "test": "pearson",
        "col_x": col_x,
        "col_y": col_y,
        "n": len(pair),
        "statistic": round(float(r), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "r": round(float(r), 4),
            "r_squared": round(float(r**2), 4),
            "label": _effect_size_label(r, (0.1, 0.3)),
        },
        "interpretation": (
            f"Significant {'positive' if r > 0 else 'negative'} correlation between {col_x} and {col_y} (r={r:.4f}, p={p:.4f})."
            if p < 0.05
            else f"No significant correlation (r={r:.4f}, p={p:.4f})."
        ),
        "assumption_checks": {
            "normality_x": check_normality(pair[col_x]),
            "normality_y": check_normality(pair[col_y]),
        },
    }

    explanation = await _get_gemini_explanation("pearson", result)
    result["plain_english"] = explanation or _rule_based_explanation("pearson", result)
    return result


# S8 Simple Linear Regression

async def linear_regression(
    df: pd.DataFrame, predictor: str, outcome: str
) -> dict:
    for col in [predictor, outcome]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[predictor, outcome]].dropna()
    x = pair[predictor]
    y = pair[outcome]

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

    result = {
        "test": "linear_regression",
        "predictor": predictor,
        "outcome": outcome,
        "n": len(pair),
        "statistic": round(float((slope / std_err) ** 2), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "coefficients": {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "std_error": round(float(std_err), 6),
        },
        "effect_size": {
            "r_squared": round(float(r_value**2), 4),
            "r": round(float(r_value), 4),
            "label": _effect_size_label(r_value, (0.1, 0.3)),
        },
        "interpretation": (
            f"'{predictor}' significantly predicts '{outcome}' (p={p_value:.4f}). "
            f"R² = {r_value**2:.4f} — explains {r_value**2*100:.1f}% of variance."
            if p_value < 0.05
            else f"'{predictor}' does not significantly predict '{outcome}' (p={p_value:.4f})."
        ),
        "assumption_checks": {
            "normality_outcome": check_normality(y),
            "sample_size": check_sample_size(pair[predictor]),
        },
    }

    explanation = await _get_gemini_explanation("linear_regression", result)
    result["plain_english"] = explanation or _rule_based_explanation("linear_regression", result)
    return result


# S9 Multiple Regression

async def multiple_regression(
    df: pd.DataFrame, predictors: list[str], outcome: str
) -> dict:
    try:
        import statsmodels.api as sm

        for col in predictors + [outcome]:
            if col not in df.columns:
                raise ColumnNotFoundError(col)

        clean = df[predictors + [outcome]].dropna()
        X = sm.add_constant(clean[predictors])
        y = clean[outcome]
        model = sm.OLS(y, X).fit()

        coefficients = {
            str(name): {
                "coef": round(float(val), 6),
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05),
            }
            for name, val, p in zip(
                model.params.index,
                model.params.values,
                model.pvalues.values,
            )
        }

        result = {
            "test": "multiple_regression",
            "predictors": predictors,
            "outcome": outcome,
            "n": len(clean),
            "statistic": round(float(model.fvalue), 4),
            "p_value": round(float(model.f_pvalue), 6),
            "significant": bool(model.f_pvalue < 0.05),
            "coefficients": coefficients,
            "effect_size": {
                "r_squared": round(float(model.rsquared), 4),
                "adj_r_squared": round(float(model.rsquared_adj), 4),
                "label": _effect_size_label(model.rsquared, (0.02, 0.13)),
            },
            "interpretation": (
                f"The model significantly predicts '{outcome}' (F={model.fvalue:.3f}, p={model.f_pvalue:.4f}). "
                f"R² = {model.rsquared:.4f}."
                if model.f_pvalue < 0.05
                else f"The model does not significantly predict '{outcome}' (p={model.f_pvalue:.4f})."
            ),
            "assumption_checks": {
                "normality_outcome": check_normality(y),
                "sample_size": check_sample_size(clean[predictors[0]]),
            },
        }

        explanation = await _get_gemini_explanation("multiple_regression", result)
        result["plain_english"] = explanation or _rule_based_explanation("multiple_regression", result)
        return result

    except Exception as e:
        return {"error": f"Multiple regression failed: {str(e)}"}


# S10 Mann-Whitney U Test

async def mann_whitney(
    df: pd.DataFrame, num_col: str, cat_col: str
) -> dict:
    if num_col not in df.columns:
        raise ColumnNotFoundError(num_col)
    if cat_col not in df.columns:
        raise ColumnNotFoundError(cat_col)

    groups = [
        g.dropna()
        for _, g in df.groupby(cat_col)[num_col]
    ]
    if len(groups) < 2:
        return {"error": "Need at least 2 groups."}

    g1, g2 = groups[0], groups[1]
    stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")

    # Effect size r
    n = len(g1) + len(g2)
    z = (stat - len(g1) * len(g2) / 2) / np.sqrt(len(g1) * len(g2) * (n + 1) / 12)
    r = abs(z) / np.sqrt(n)

    group_names = df[cat_col].dropna().unique()[:2].tolist()

    result = {
        "test": "mann_whitney",
        "num_col": num_col,
        "cat_col": cat_col,
        "groups": {
            str(group_names[0]): {"n": len(g1), "median": round(float(g1.median()), 4)},
            str(group_names[1]): {"n": len(g2), "median": round(float(g2.median()), 4)},
        },
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "r": round(float(r), 4),
            "label": _effect_size_label(r, (0.1, 0.3)),
        },
        "interpretation": (
            f"Significant difference in {num_col} between groups (p={p:.4f})."
            if p < 0.05
            else f"No significant difference (p={p:.4f})."
        ),
        "assumption_checks": {
            "sample_size_g1": check_sample_size(g1, min_n=5),
            "sample_size_g2": check_sample_size(g2, min_n=5),
        },
    }

    explanation = await _get_gemini_explanation("mann_whitney", result)
    result["plain_english"] = explanation or _rule_based_explanation("mann_whitney", result)
    return result


# S11  Wilcoxon Signed-Rank Test

async def wilcoxon_test(
    df: pd.DataFrame, col_before: str, col_after: str
) -> dict:
    for col in [col_before, col_after]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[col_before, col_after]].dropna()
    stat, p = scipy_stats.wilcoxon(pair[col_before], pair[col_after])

    result = {
        "test": "wilcoxon",
        "col_before": col_before,
        "col_after": col_after,
        "n_pairs": len(pair),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {"label": "not_applicable"},
        "interpretation": (
            f"Significant change from {col_before} to {col_after} (p={p:.4f})."
            if p < 0.05
            else f"No significant change (p={p:.4f})."
        ),
        "assumption_checks": {
            "sample_size": check_sample_size(pair[col_before], min_n=10),
        },
    }

    explanation = await _get_gemini_explanation("wilcoxon", result)
    result["plain_english"] = explanation or _rule_based_explanation("wilcoxon", result)
    return result


# S12 Kruskal-Wallis Test

async def kruskal_wallis(
    df: pd.DataFrame, num_col: str, cat_col: str
) -> dict:
    if num_col not in df.columns:
        raise ColumnNotFoundError(num_col)
    if cat_col not in df.columns:
        raise ColumnNotFoundError(cat_col)

    groups = [g.dropna() for _, g in df.groupby(cat_col)[num_col] if len(g.dropna()) >= 2]
    if len(groups) < 2:
        return {"error": "Need at least 2 groups."}

    stat, p = scipy_stats.kruskal(*groups)

    group_stats = {
        str(name): {"n": len(g), "median": round(float(g.median()), 4)}
        for name, g in df.groupby(cat_col)[num_col]
        if len(g.dropna()) >= 2
    }

    result = {
        "test": "kruskal_wallis",
        "num_col": num_col,
        "cat_col": cat_col,
        "n_groups": len(groups),
        "group_stats": group_stats,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {"label": "see_group_medians"},
        "interpretation": (
            f"Significant difference in {num_col} across groups (p={p:.4f})."
            if p < 0.05
            else f"No significant difference (p={p:.4f})."
        ),
        "assumption_checks": {
            "min_group_size": {"passed": all(len(g) >= 5 for g in groups)},
        },
    }

    explanation = await _get_gemini_explanation("kruskal_wallis", result)
    result["plain_english"] = explanation or _rule_based_explanation("kruskal_wallis", result)
    return result


# S13  Spearman Correlation

async def spearman_correlation(
    df: pd.DataFrame, col_x: str, col_y: str
) -> dict:
    for col in [col_x, col_y]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[col_x, col_y]].dropna()
    r, p = scipy_stats.spearmanr(pair[col_x], pair[col_y])

    result = {
        "test": "spearman",
        "col_x": col_x,
        "col_y": col_y,
        "n": len(pair),
        "statistic": round(float(r), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "rho": round(float(r), 4),
            "label": _effect_size_label(r, (0.1, 0.3)),
        },
        "interpretation": (
            f"Significant monotonic {'positive' if r > 0 else 'negative'} relationship "
            f"between {col_x} and {col_y} (ρ={r:.4f}, p={p:.4f})."
            if p < 0.05
            else f"No significant monotonic relationship (ρ={r:.4f}, p={p:.4f})."
        ),
        "assumption_checks": {
            "sample_size": check_sample_size(pair[col_x]),
        },
    }

    explanation = await _get_gemini_explanation("spearman", result)
    result["plain_english"] = explanation or _rule_based_explanation("spearman", result)
    return result


# S14  Kendall's Tau

async def kendall_tau(
    df: pd.DataFrame, col_x: str, col_y: str
) -> dict:
    for col in [col_x, col_y]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[col_x, col_y]].dropna()
    tau, p = scipy_stats.kendalltau(pair[col_x], pair[col_y])

    result = {
        "test": "kendall",
        "col_x": col_x,
        "col_y": col_y,
        "n": len(pair),
        "statistic": round(float(tau), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "tau": round(float(tau), 4),
            "label": _effect_size_label(tau, (0.1, 0.3)),
        },
        "interpretation": (
            f"Significant ordinal association between {col_x} and {col_y} (τ={tau:.4f}, p={p:.4f})."
            if p < 0.05
            else f"No significant ordinal association (τ={tau:.4f}, p={p:.4f})."
        ),
        "assumption_checks": {
            "sample_size": check_sample_size(pair[col_x]),
        },
    }

    explanation = await _get_gemini_explanation("kendall", result)
    result["plain_english"] = explanation or _rule_based_explanation("kendall", result)
    return result


# S15  Chi-Square Test of Independence

async def chi_square_independence(
    df: pd.DataFrame, col_a: str, col_b: str
) -> dict:
    for col in [col_a, col_b]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[col_a, col_b]].dropna()
    ct = pd.crosstab(pair[col_a], pair[col_b])
    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)

    # Cramér's V
    n = len(pair)
    min_dim = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0

    result = {
        "test": "chi_square_independence",
        "col_a": col_a,
        "col_b": col_b,
        "n": n,
        "dof": int(dof),
        "statistic": round(float(chi2), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "cramers_v": round(float(v), 4),
            "label": _effect_size_label(v, (0.1, 0.3)),
        },
        "interpretation": (
            f"Significant association between '{col_a}' and '{col_b}' (χ²={chi2:.4f}, p={p:.4f})."
            if p < 0.05
            else f"No significant association (χ²={chi2:.4f}, p={p:.4f})."
        ),
        "assumption_checks": {
            "min_expected_freq": {
                "passed": bool((expected >= 5).all()),
                "reason": "All expected frequencies ≥ 5" if (expected >= 5).all() else "Some expected frequencies < 5 — consider Fisher's Exact Test.",
            },
        },
    }

    explanation = await _get_gemini_explanation("chi_square", result)
    result["plain_english"] = explanation or _rule_based_explanation("chi_square", result)
    return result


# S16  Chi-Square Goodness of Fit

async def chi_square_goodness(
    df: pd.DataFrame, column: str, expected_proportions: dict = None
) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    observed = df[column].value_counts()
    n = len(df[column].dropna())

    if expected_proportions:
        expected = np.array([expected_proportions.get(str(k), 1/len(observed)) * n for k in observed.index])
    else:
        expected = np.full(len(observed), n / len(observed))

    stat, p = scipy_stats.chisquare(observed.values, expected)

    result = {
        "test": "chi_square_goodness",
        "column": column,
        "n": n,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {"label": "see_observed_vs_expected"},
        "observed": {str(k): int(v) for k, v in observed.items()},
        "expected": {str(k): round(float(e), 2) for k, e in zip(observed.index, expected)},
        "interpretation": (
            f"Observed distribution of '{column}' significantly differs from expected (p={p:.4f})."
            if p < 0.05
            else f"Observed distribution does not significantly differ from expected (p={p:.4f})."
        ),
        "assumption_checks": {
            "min_expected_freq": {
                "passed": bool((expected >= 5).all()),
                "reason": "All expected frequencies ≥ 5",
            },
        },
    }

    explanation = await _get_gemini_explanation("chi_square_goodness", result)
    result["plain_english"] = explanation or _rule_based_explanation("chi_square_goodness", result)
    return result


# S17  Fisher's Exact Test

async def fisher_exact(
    df: pd.DataFrame, col_a: str, col_b: str
) -> dict:
    for col in [col_a, col_b]:
        if col not in df.columns:
            raise ColumnNotFoundError(col)

    pair = df[[col_a, col_b]].dropna()
    ct = pd.crosstab(pair[col_a], pair[col_b])

    if ct.shape != (2, 2):
        return {"error": "Fisher's Exact Test requires a 2x2 contingency table."}

    odds_ratio, p = scipy_stats.fisher_exact(ct.values)

    result = {
        "test": "fisher_exact",
        "col_a": col_a,
        "col_b": col_b,
        "n": len(pair),
        "statistic": round(float(odds_ratio), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size": {
            "odds_ratio": round(float(odds_ratio), 4),
            "label": "see_odds_ratio",
        },
        "interpretation": (
            f"Significant association between '{col_a}' and '{col_b}' (OR={odds_ratio:.4f}, p={p:.4f})."
            if p < 0.05
            else f"No significant association (OR={odds_ratio:.4f}, p={p:.4f})."
        ),
        "assumption_checks": {
            "table_shape": {"passed": True, "reason": "2x2 contingency table confirmed."},
        },
    }

    explanation = await _get_gemini_explanation("fisher_exact", result)
    result["plain_english"] = explanation or _rule_based_explanation("fisher_exact", result)
    return result


# S18 — Shapiro-Wilk Normality Test

async def shapiro_wilk(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    s = df[column].dropna()
    if len(s) > 5000:
        s = s.sample(5000, random_state=42)

    stat, p = scipy_stats.shapiro(s)

    result = {
        "test": "shapiro_wilk",
        "column": column,
        "n": len(s),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "is_normal": bool(p >= 0.05),
        "effect_size": {"label": "not_applicable"},
        "interpretation": (
            f"'{column}' appears normally distributed (W={stat:.4f}, p={p:.4f})."
            if p >= 0.05
            else f"'{column}' deviates significantly from normality (W={stat:.4f}, p={p:.4f})."
        ),
        "assumption_checks": {
            "sample_size": check_sample_size(s, min_n=3),
        },
    }

    explanation = await _get_gemini_explanation("shapiro_wilk", result)
    result["plain_english"] = explanation or _rule_based_explanation("shapiro_wilk", result)
    return result


# S19  Kolmogorov-Smirnov Test

async def ks_test(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ColumnNotFoundError(column)

    s = df[column].dropna()
    stat, p = scipy_stats.kstest(
        (s - s.mean()) / s.std(),
        "norm",
    )

    result = {
        "test": "ks_test",
        "column": column,
        "n": len(s),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "is_normal": bool(p >= 0.05),
        "effect_size": {"label": "not_applicable"},
        "interpretation": (
            f"'{column}' is consistent with a normal distribution (D={stat:.4f}, p={p:.4f})."
            if p >= 0.05
            else f"'{column}' is not consistent with a normal distribution (D={stat:.4f}, p={p:.4f})."
        ),
        "assumption_checks": {
            "sample_size": check_sample_size(s),
        },
    }

    explanation = await _get_gemini_explanation("ks_test", result)
    result["plain_english"] = explanation or _rule_based_explanation("ks_test", result)
    return result


# S20 Levene's Test

async def levene_test(
    df: pd.DataFrame, num_col: str, cat_col: str
) -> dict:
    if num_col not in df.columns:
        raise ColumnNotFoundError(num_col)
    if cat_col not in df.columns:
        raise ColumnNotFoundError(cat_col)

    groups = [g.dropna() for _, g in df.groupby(cat_col)[num_col] if len(g.dropna()) >= 2]
    if len(groups) < 2:
        return {"error": "Need at least 2 groups."}

    stat, p = scipy_stats.levene(*groups)

    result = {
        "test": "levene",
        "num_col": num_col,
        "cat_col": cat_col,
        "n_groups": len(groups),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "equal_variances": bool(p >= 0.05),
        "effect_size": {"label": "not_applicable"},
        "interpretation": (
            f"Variances are equal across groups (p={p:.4f}) — parametric tests are appropriate."
            if p >= 0.05
            else f"Variances differ significantly (p={p:.4f}) — use Welch's t-test or non-parametric alternatives."
        ),
        "assumption_checks": {
            "min_group_size": {"passed": all(len(g) >= 2 for g in groups)},
        },
    }

    explanation = await _get_gemini_explanation("levene", result)
    result["plain_english"] = explanation or _rule_based_explanation("levene", result)
    return result


# 21 Bartlett's Test

async def bartlett_test(
    df: pd.DataFrame, num_col: str, cat_col: str
) -> dict:
    if num_col not in df.columns:
        raise ColumnNotFoundError(num_col)
    if cat_col not in df.columns:
        raise ColumnNotFoundError(cat_col)

    groups = [g.dropna() for _, g in df.groupby(cat_col)[num_col] if len(g.dropna()) >= 2]
    if len(groups) < 2:
        return {"error": "Need at least 2 groups."}

    stat, p = scipy_stats.bartlett(*groups)

    result = {
        "test": "bartlett",
        "num_col": num_col,
        "cat_col": cat_col,
        "n_groups": len(groups),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "equal_variances": bool(p >= 0.05),
        "effect_size": {"label": "not_applicable"},
        "interpretation": (
            f"Variances are equal across groups — Bartlett (p={p:.4f})."
            if p >= 0.05
            else f"Variances differ significantly — Bartlett (p={p:.4f}). Note: Bartlett is sensitive to non-normality."
        ),
        "assumption_checks": {
            "normality": check_normality(df[num_col].dropna()),
        },
    }

    explanation = await _get_gemini_explanation("bartlett", result)
    result["plain_english"] = explanation or _rule_based_explanation("bartlett", result)
    return result


# Master dispatcher

async def run_test(df: pd.DataFrame, payload: dict) -> dict:
    """
    Single entry point for all 21 statistical tests.
    payload keys: test_name + test-specific column args
    """
    t = payload.get("test_name")

    if t == "one_sample_t":
        return await one_sample_t(df, payload["column"], payload.get("pop_mean", 0.0))
    elif t == "independent_t":
        return await independent_t(df, payload["num_col"], payload["cat_col"])
    elif t == "paired_t":
        return await paired_t(df, payload["col_before"], payload["col_after"])
    elif t == "one_way_anova":
        return await one_way_anova(df, payload["num_col"], payload["cat_col"])
    elif t == "two_way_anova":
        return await two_way_anova(df, payload["num_col"], payload["cat_col1"], payload["cat_col2"])
    elif t == "repeated_measures_anova":
        return await repeated_measures_anova(df, payload["columns"])
    elif t == "pearson":
        return await pearson_correlation(df, payload["col_x"], payload["col_y"])
    elif t == "linear_regression":
        return await linear_regression(df, payload["predictor"], payload["outcome"])
    elif t == "multiple_regression":
        return await multiple_regression(df, payload["predictors"], payload["outcome"])
    elif t == "mann_whitney":
        return await mann_whitney(df, payload["num_col"], payload["cat_col"])
    elif t == "wilcoxon":
        return await wilcoxon_test(df, payload["col_before"], payload["col_after"])
    elif t == "kruskal_wallis":
        return await kruskal_wallis(df, payload["num_col"], payload["cat_col"])
    elif t == "spearman":
        return await spearman_correlation(df, payload["col_x"], payload["col_y"])
    elif t == "kendall":
        return await kendall_tau(df, payload["col_x"], payload["col_y"])
    elif t == "chi_square_independence":
        return await chi_square_independence(df, payload["col_a"], payload["col_b"])
    elif t == "chi_square_goodness":
        return await chi_square_goodness(df, payload["column"], payload.get("expected_proportions"))
    elif t == "fisher_exact":
        return await fisher_exact(df, payload["col_a"], payload["col_b"])
    elif t == "shapiro_wilk":
        return await shapiro_wilk(df, payload["column"])
    elif t == "ks_test":
        return await ks_test(df, payload["column"])
    elif t == "levene":
        return await levene_test(df, payload["num_col"], payload["cat_col"])
    elif t == "bartlett":
        return await bartlett_test(df, payload["num_col"], payload["cat_col"])
    else:
        return {"error": f"Unknown test_name: '{t}'"}
    














    