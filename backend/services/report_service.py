

import asyncio
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.cache import (
    get_eda_history,
    get_ml_result,
    get_nlp_history,
    get_profile,
    get_stats_history,
    set_profile,
)
from core.config import settings
from services import session_service

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
TEMPLATE_NAME = "report_template.html"

SECTION_DEFS = [
    ("executive_summary", "Executive Summary"),
    ("profiling", "Data Profiling Summary"),
    ("cleaning", "Data Cleaning Summary"),
    ("transformation", "Data Transformation Summary"),
    ("eda", "Exploratory Data Analysis Summary"),
    ("statistics", "Statistical Analysis Summary"),
    ("nlp", "NLP Analysis Summary"),
    ("ml_strategy", "ML Strategy Summary"),
    ("history", "Full Operation History"),
    ("appendix", "Appendix"),
]
SECTION_KEYS = [k for k, _ in SECTION_DEFS]
SECTION_TITLES = dict(SECTION_DEFS)
VALID_FORMATS = {"pdf", "html"}

NOT_PERFORMED_NOTES = {
    "profiling": "Profiling could not be generated for the current dataset.",
    "cleaning": "No cleaning operations were applied.",
    "transformation": "No transformations were applied.",
    "eda": "No exploratory analyses were run.",
    "statistics": "No statistical tests were run.",
    "nlp": "No NLP analyses were run.",
    "ml_strategy": "No ML strategy was generated.",
    "history": "No cleaning or transformation operations were recorded.",
}

# Transform groupings used for the transformation summary
_SCALED = {"minmax", "zscore", "robust", "log", "power"}
_ENCODED = {"label_encode", "onehot", "ordinal", "frequency", "target_encode"}
_BINNED = {"equal_width", "equal_freq", "custom_bins"}
_DATETIME = {"extract_year", "extract_month", "extract_day", "extract_dow", "extract_hour"}
_ENGINEERED = {"polynomial", "interaction", "formula", "lag", "rolling_mean"}

MAX_OPS_IN_SECTION = 100
MAX_APPENDIX_COLUMNS = 100
MAX_NLP_COLUMNS = 2


class PDFEngineUnavailable(Exception):
    """Raised when WeasyPrint (or its system libraries) cannot be loaded."""


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


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n} {word}"
    if word.endswith("analysis"):
        return f"{n} {word[:-2]}es"   # analysis -> analyses
    return f"{n} {word}s"


def _options_text(options) -> str:
    if not options:
        return ""
    try:
        text = json.dumps(options, default=str)
    except Exception:
        text = str(options)
    return text[:150]


def _section(key: str, performed: bool, data: dict | None = None, note: str | None = None) -> dict:
    return {
        "key": key,
        "title": SECTION_TITLES[key],
        "number": None,  # assigned once the final section list is known
        "performed": performed,
        "note": note if note is not None else (None if performed else NOT_PERFORMED_NOTES.get(key)),
        "data": data or {},
    }


def _not_performed(key: str, note: str | None = None) -> dict:
    return _section(key, False, None, note)


def _normalise_sections(sections: list[str] | None) -> list[str]:
    """Validate requested sections; return them in canonical order. Empty = all."""
    if not sections:
        return list(SECTION_KEYS)
    unknown = [s for s in sections if s not in SECTION_KEYS]
    if unknown:
        raise ValueError(f"Unknown section(s): {unknown}. Valid sections: {SECTION_KEYS}")
    wanted = set(sections)
    return [k for k in SECTION_KEYS if k in wanted]


# Section availability (drives the frontend checklist badges)

def get_section_availability(session_id: str) -> list[dict]:
    history = session_service.get_history(session_id)
    n_clean = sum(1 for o in history if o.get("stage") == "clean")
    n_transform = sum(1 for o in history if o.get("stage") == "transform")
    n_eda = len(get_eda_history(session_id))
    n_stats = len(get_stats_history(session_id))
    n_nlp = len(get_nlp_history(session_id))
    ml = get_ml_result(session_id)
    ml_ok = bool(ml and ml.get("valid"))
    profile_cached = get_profile(session_id) is not None

    info = {
        "executive_summary": (True, "Generated from the sections included in the report."),
        "profiling": (True, "Cached." if profile_cached else "Will be computed when the report is generated."),
        "cleaning": (n_clean > 0, f"{_plural(n_clean, 'operation')} applied." if n_clean else NOT_PERFORMED_NOTES["cleaning"]),
        "transformation": (n_transform > 0, f"{_plural(n_transform, 'operation')} applied." if n_transform else NOT_PERFORMED_NOTES["transformation"]),
        "eda": (n_eda > 0, f"{_plural(n_eda, 'analysis')} run." if n_eda else NOT_PERFORMED_NOTES["eda"]),
        "statistics": (n_stats > 0, f"{_plural(n_stats, 'test')} run." if n_stats else NOT_PERFORMED_NOTES["statistics"]),
        "nlp": (n_nlp > 0, f"{_plural(n_nlp, 'analysis')} run." if n_nlp else NOT_PERFORMED_NOTES["nlp"]),
        "ml_strategy": (ml_ok, "Strategy generated." if ml_ok else NOT_PERFORMED_NOTES["ml_strategy"]),
        "history": (len(history) > 0, f"{_plural(len(history), 'operation')} recorded." if history else NOT_PERFORMED_NOTES["history"]),
        "appendix": (True, "Column map and numeric statistics."),
    }
    return [
        {"key": k, "title": t, "completed": info[k][0], "detail": info[k][1]}
        for k, t in SECTION_DEFS
    ]


# Section 2 — Profiling

def _profile_data(p: dict) -> dict:
    missing = [
        {"column": c, "count": _int(v.get("count")), "pct": v.get("pct"), "is_critical": bool(v.get("is_critical"))}
        for c, v in p.get("missing_values", {}).items()
        if _int(v.get("count")) > 0
    ]
    missing.sort(key=lambda x: x["pct"] or 0, reverse=True)

    outliers = [
        {"column": c, "outlier_count": _int(v.get("outlier_count")), "pct": v.get("pct")}
        for c, v in p.get("outlier_detection", {}).items()
        if _int(v.get("outlier_count")) > 0
    ]
    outliers.sort(key=lambda x: x["pct"] or 0, reverse=True)

    skewed = [
        {"column": c, "skewness": v.get("skewness"), "label": v.get("distribution_label")}
        for c, v in p.get("distribution_analysis", {}).items()
        if abs(v.get("skewness") or 0) > 1
    ]
    skewed.sort(key=lambda x: abs(x["skewness"] or 0), reverse=True)

    suspicious = []
    for col, flags in p.get("suspicious_values", {}).items():
        for f in flags:
            suspicious.append({"column": col, "rule": f.get("rule"), "count": _int(f.get("flagged_count"))})

    dup = p.get("duplicate_rows", {})
    return {
        "dimensions": p.get("dimensions", {}),
        "quality": p.get("quality_summary", {}),
        "insights": list(p.get("insights", [])),
        "insights_source": p.get("insights_source", "rule_based"),
        "missing_columns": missing[:15],
        "duplicate_rows": {"count": _int(dup.get("count")), "pct": dup.get("pct", 0)},
        "outlier_columns": outliers[:10],
        "skewed_columns": skewed[:10],
        "identifier_columns": [i.get("column") for i in p.get("identifier_columns", [])],
        "constant_columns": [c.get("column") for c in p.get("constant_columns", [])],
        "correlated_pairs": list(p.get("correlation_pairs", []))[:10],
        "potential_targets": [
            {"column": t.get("column"), "suitability": t.get("suitability"), "reasoning": t.get("reasoning")}
            for t in p.get("potential_targets", [])
        ],
        "type_issues": [
            {"column": t.get("column"), "suggested_type": t.get("suggested_type")}
            for t in p.get("type_inconsistencies", [])
        ],
        "suspicious": suspicious[:15],
    }


async def collect_profiling(session_id: str, df: pd.DataFrame) -> dict:
    # Every clean/transform apply clears the profile cache, so recompute when missing
    profile = get_profile(session_id)
    if profile is None:
        from services.profile_service import run_full_profile
        profile = await run_full_profile(df)
        set_profile(session_id, profile)
    return _section("profiling", True, _profile_data(profile))


# Sections 3 & 4 — Cleaning / Transformation (from session history)

def collect_cleaning(session_id: str) -> dict:
    ops = [o for o in session_service.get_history(session_id) if o.get("stage") == "clean"]
    if not ops:
        return _not_performed("cleaning")

    rows = [
        {
            "step": i,
            "issue_type": o.get("issue_type"),
            "column": o.get("column"),
            "strategy": o.get("strategy"),
            "rows_before": _int(o.get("rows_before")),
            "rows_after": _int(o.get("rows_after")),
            "cols_before": _int(o.get("cols_before")),
            "cols_after": _int(o.get("cols_after")),
            "timestamp": o.get("timestamp"),
        }
        for i, o in enumerate(ops, start=1)
    ]
    return _section("cleaning", True, {
        "total_operations": len(ops),
        "net_rows_change": sum(r["rows_after"] - r["rows_before"] for r in rows),
        "net_columns_change": sum(r["cols_after"] - r["cols_before"] for r in rows),
        "by_issue": [
            {"issue_type": k, "count": v}
            for k, v in Counter(r["issue_type"] for r in rows).most_common()
        ],
        "operations": rows[:MAX_OPS_IN_SECTION],
        "truncated": len(rows) > MAX_OPS_IN_SECTION,
    })


def collect_transformation(session_id: str) -> dict:
    ops = [o for o in session_service.get_history(session_id) if o.get("stage") == "transform"]
    if not ops:
        return _not_performed("transformation")

    rows, new_features, columns = [], [], []
    counts = {"scaled": 0, "encoded": 0, "binned": 0, "datetime_extracted": 0,
              "engineered": 0, "dim_reduction": 0, "excluded_from_ml": 0}

    for i, o in enumerate(ops, start=1):
        t = o.get("transform_type")
        if t in _SCALED:
            counts["scaled"] += 1
        elif t in _ENCODED:
            counts["encoded"] += 1
        elif t in _BINNED:
            counts["binned"] += 1
        elif t in _DATETIME:
            counts["datetime_extracted"] += 1
        elif t in _ENGINEERED:
            counts["engineered"] += 1
        elif t == "pca":
            counts["dim_reduction"] += 1
        elif t == "exclude_ml":
            counts["excluded_from_ml"] += 1

        col = o.get("column")
        if col and col not in columns:
            columns.append(col)
        new_cols = list(o.get("new_columns") or [])
        for c in new_cols:
            if c not in new_features:
                new_features.append(c)

        rows.append({
            "step": i,
            "transform_type": t,
            "column": col,
            "new_columns": new_cols,
            "options": _options_text(o.get("options")),
            "rows_before": _int(o.get("rows_before")),
            "rows_after": _int(o.get("rows_after")),
            "cols_before": _int(o.get("cols_before")),
            "cols_after": _int(o.get("cols_after")),
            "timestamp": o.get("timestamp"),
        })

    return _section("transformation", True, {
        "total_operations": len(ops),
        "columns_transformed": columns,
        "new_features": new_features,
        "counts": counts,
        "by_type": [
            {"transform_type": k, "count": v}
            for k, v in Counter(r["transform_type"] for r in rows).most_common()
        ],
        "operations": rows[:MAX_OPS_IN_SECTION],
        "truncated": len(rows) > MAX_OPS_IN_SECTION,
    })


# Section 5 — EDA

def collect_eda(session_id: str, df: pd.DataFrame) -> dict:
    from services import eda_service

    history = get_eda_history(session_id)
    if not history:
        return _not_performed("eda")

    summary = eda_service.generate_eda_summary(df, history)
    try:
        observations = eda_service._generate_rule_based_eda_insights(df)
    except Exception:
        observations = []

    analyses = []
    for h in history[:MAX_OPS_IN_SECTION]:
        if h.get("analysis_type") == "question":
            continue
        cols = [str(h[k]) for k in ("column", "col_x", "col_y") if h.get(k)]
        analyses.append({
            "analysis_type": h.get("analysis_type"),
            "columns": ", ".join(cols),
            "filters": _int(h.get("filter_count")),
            "rows_analysed": _int(h.get("rows_analysed")),
        })

    return _section("eda", True, {
        "total_analyses": len(history),
        "analyses": analyses,
        "questions": list(summary.get("questions_asked", [])),
        "shape": summary.get("shape", {}),
        "column_types": summary.get("column_types", {}),
        "strong_correlations": list(summary.get("strong_correlations", [])),
        "skewed_columns": list(summary.get("skewed_columns", [])),
        "high_missing_columns": [
            {"column": c, "pct": p} for c, p in summary.get("high_missing_columns", {}).items()
        ],
        "observations": observations,
    })


# Section 6 — Statistics

def collect_statistics(session_id: str) -> dict:
    history = get_stats_history(session_id)
    if not history:
        return _not_performed("statistics")

    tests = [
        {
            "test_name": h.get("test_name"),
            "options": _options_text(h.get("options")),
            "p_value": h.get("p_value"),
            "significant": h.get("significant"),
            "plain_english": h.get("plain_english"),
        }
        for h in history[:MAX_OPS_IN_SECTION]
    ]
    return _section("statistics", True, {
        "total_tests": len(history),
        "significant_count": sum(1 for h in history if h.get("significant") is True),
        "tests": tests,
    })


# Section 7 — NLP (compact summary for the columns already analysed)

def _nlp_column_summary(df: pd.DataFrame, col: str) -> dict:
    from services import nlp_service

    if col not in df.columns:
        return {"column": col, "error": "This column no longer exists in the working dataset."}
    try:
        sentiment = nlp_service.sentiment_aggregation(df, col)
        keywords = nlp_service.tfidf_keywords(df, col, 10)
        topics = nlp_service.lda_topics(df, col, 3)
        quality = nlp_service.text_quality_assessment(df, col)
        stats = nlp_service.text_statistics(df, col)

        if "error" in sentiment:
            sentiment = {}
        if keywords and "error" in keywords[0]:
            keywords = []
        if topics and "error" in topics[0]:
            topics = []
        if "error" in quality:
            quality = {}
        if "error" in stats:
            stats = {}

        return nlp_service.nlp_summary(col, sentiment, keywords, topics, quality, stats)
    except Exception as e:
        logger.warning(f"NLP report summary failed for column '{col}': {e}")
        return {"column": col, "error": "The NLP summary could not be completed for this column."}


async def collect_nlp(session_id: str, df: pd.DataFrame) -> dict:
    history = get_nlp_history(session_id)
    if not history:
        return _not_performed("nlp")

    columns: list[str] = []
    for h in history:
        c = h.get("column")
        if c and c not in columns:
            columns.append(c)

    # Heavy (VADER, LDA) — keep it off the event loop
    summaries = []
    for col in columns[:MAX_NLP_COLUMNS]:
        summaries.append(await asyncio.to_thread(_nlp_column_summary, df, col))

    return _section("nlp", True, {
        "total_analyses": len(history),
        "analysis_counts": [
            {"analysis_type": k, "count": v}
            for k, v in Counter(h.get("analysis_type") for h in history).most_common()
        ],
        "columns": summaries,
        "columns_not_summarised": max(0, len(columns) - MAX_NLP_COLUMNS),
    })


# Section 8 — ML Strategy

def collect_ml(session_id: str) -> dict:
    strategy = get_ml_result(session_id)
    if not strategy or not strategy.get("valid"):
        return _not_performed("ml_strategy")

    problem = strategy.get("problem", {})
    readiness = strategy.get("readiness", {})
    imbalance = strategy.get("imbalance", {})
    metrics = strategy.get("metrics", {})

    return _section("ml_strategy", True, {
        "problem_type": problem.get("problem_type"),
        "target": problem.get("target"),
        "objective": strategy.get("objective") or "",
        "summary": strategy.get("summary"),
        "reasoning_source": strategy.get("reasoning_source"),
        "readiness": {
            "score": readiness.get("score"),
            "label": readiness.get("label"),
            "warnings": list(readiness.get("warnings", [])),
        },
        "imbalance": {
            "is_imbalanced": bool(imbalance.get("is_imbalanced")),
            "severity": imbalance.get("severity"),
        },
        "top_models": [
            {
                "name": m.get("name"),
                "score": m.get("score"),
                "suitability": m.get("suitability"),
                "reasoning": m.get("reasoning"),
            }
            for m in strategy.get("models", [])[:3]
        ],
        "metrics": {
            "primary": metrics.get("primary"),
            "secondary": list(metrics.get("secondary", [])),
            "reason": metrics.get("reason"),
        },
        "validation": (strategy.get("cross_validation") or {}).get("method"),
        "split": (strategy.get("split_strategy") or {}).get("strategy"),
        "preprocessing": [p.get("technique") for p in strategy.get("preprocessing", [])[:6]],
        "roadmap": [
            {"step_number": r.get("step_number"), "title": r.get("title"), "description": r.get("description")}
            for r in strategy.get("roadmap", [])
        ],
    })


# Section 9 — Full operation history

def collect_history(session_id: str) -> dict:
    history = session_service.get_history(session_id)
    if not history:
        return _not_performed("history")

    rows = []
    for i, o in enumerate(history, start=1):
        stage = o.get("stage")
        if stage == "clean":
            action = f"{o.get('issue_type')} → {o.get('strategy')}"
        elif stage == "transform":
            action = f"{o.get('transform_type')}"
        else:
            action = str(o.get("action") or o.get("transform_type") or o.get("issue_type") or "")
        rows.append({
            "step": i,
            "stage": stage,
            "action": action,
            "column": o.get("column"),
            "rows_before": o.get("rows_before"),
            "rows_after": o.get("rows_after"),
            "cols_before": o.get("cols_before"),
            "cols_after": o.get("cols_after"),
            "timestamp": o.get("timestamp"),
        })

    return _section("history", True, {
        "total_operations": len(rows),
        "operations": rows,
        "analysis_counts": {
            "eda": len(get_eda_history(session_id)),
            "statistics": len(get_stats_history(session_id)),
            "nlp": len(get_nlp_history(session_id)),
        },
    })


# Section 10 — Appendix

def collect_appendix(session_id: str, df: pd.DataFrame) -> dict:
    profile = get_profile(session_id)
    roles = {}
    if profile:
        roles = {c.get("name"): c.get("semantic_label") for c in profile.get("column_identification", [])}

    n_rows = max(len(df), 1)
    column_map = [
        {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_pct": round(float(df[col].isna().sum() / n_rows * 100), 2),
            "unique_count": int(df[col].nunique()),
            "role": roles.get(col),
        }
        for col in list(df.columns)[:MAX_APPENDIX_COLUMNS]
    ]

    numeric_summary = []
    for col in df.select_dtypes(include=[np.number]).columns[:30]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        numeric_summary.append({
            "column": col,
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4) if len(s) > 1 else None,
            "min": round(float(s.min()), 4),
            "median": round(float(s.median()), 4),
            "max": round(float(s.max()), 4),
        })

    return _section("appendix", True, {
        "total_columns": len(df.columns),
        "column_map": column_map,
        "truncated": len(df.columns) > MAX_APPENDIX_COLUMNS,
        "numeric_summary": numeric_summary,
    })


# Section 1 — Executive summary (Gemini + rule-based fallback)

def _build_facts(df: pd.DataFrame, collected: dict) -> dict:
    """Compact, privacy-safe facts drawn only from sections in this report."""
    facts: dict = {"rows": len(df), "columns": len(df.columns)}

    def done(key):
        s = collected.get(key)
        return s["data"] if s and s["performed"] else None

    if (p := done("profiling")):
        facts["quality"] = {"score": p["quality"].get("score"), "label": p["quality"].get("label")}
        facts["profile_insights"] = p["insights"][:4]

    if (c := done("cleaning")):
        facts["cleaning_operations"] = c["total_operations"]
        facts["net_rows_change"] = c["net_rows_change"]

    if (t := done("transformation")):
        facts["transformation_operations"] = t["total_operations"]
        facts["new_features"] = len(t["new_features"])

    if (e := done("eda")):
        facts["eda_analyses"] = e["total_analyses"]
        facts["strong_correlations"] = e["strong_correlations"][:3]

    if (s := done("statistics")):
        facts["stats_tests"] = s["total_tests"]
        facts["significant_tests"] = s["significant_count"]
        facts["significant_test_names"] = [x["test_name"] for x in s["tests"] if x["significant"]][:3]

    if (n := done("nlp")):
        facts["nlp"] = [
            {"column": x.get("column"), "overall_sentiment": (x.get("sentiment") or {}).get("overall")}
            for x in n["columns"] if "error" not in x
        ]

    if (m := done("ml_strategy")):
        facts["ml"] = {
            "problem_type": m["problem_type"],
            "target": m["target"],
            "top_model": m["top_models"][0]["name"] if m["top_models"] else None,
            "readiness": m["readiness"]["label"],
            "readiness_warnings": m["readiness"]["warnings"][:2],
        }

    facts["not_performed"] = [
        s["title"] for k, s in collected.items() if k != "executive_summary" and not s["performed"]
    ]
    return facts


def _rule_based_summary(facts: dict) -> dict:
    parts = [
        f"This report summarises an analysis of a dataset with {facts['rows']:,} rows "
        f"and {facts['columns']} columns."
    ]
    if "quality" in facts:
        q = facts["quality"]
        parts.append(f"Data quality was assessed as {q['label']} ({q['score']}/100).")

    prep = []
    if facts.get("cleaning_operations"):
        prep.append(_plural(facts["cleaning_operations"], "cleaning operation"))

    if facts.get("transformation_operations"):
        prep.append(_plural(facts["transformation_operations"], "transformation"))

    if prep:
        parts.append("The data was prepared with " + " and ".join(prep) + ".")

    if facts.get("eda_analyses"):
        n_eda = facts["eda_analyses"]
        parts.append(
            f"{_plural(n_eda, 'exploratory analysis')} {'was' if n_eda == 1 else 'were'} recorded."
        )

    if facts.get("stats_tests"):
        parts.append(
            f"{facts['significant_tests']} of {_plural(facts['stats_tests'], 'statistical test')} "
            f"were significant at the 5% level."
            if facts["stats_tests"] != 1 else
            f"The statistical test was {'significant' if facts['significant_tests'] else 'not significant'} at the 5% level."
        )

    if "ml" in facts:
        ml = facts["ml"]
        text = f"The modelling task was framed as {str(ml['problem_type']).replace('_', ' ')}"
        if ml.get("target"):
            text += f" with '{ml['target']}' as the target"
        if ml.get("top_model"):
            text += f"; the highest-ranked starting model is {ml['top_model']}"
        parts.append(text + ".")
        
    if facts.get("not_performed"):
        parts.append("Not performed: " + ", ".join(facts["not_performed"]) + ".")

    highlights: list[str] = []
    highlights.extend(facts.get("profile_insights", [])[:2])
    for c in facts.get("strong_correlations", [])[:2]:
        highlights.append(f"'{c.get('col1')}' and '{c.get('col2')}' are strongly correlated (r={c.get('r')}).")
    for name in facts.get("significant_test_names", [])[:2]:
        highlights.append(f"The {name} test produced a statistically significant result.")
    if "ml" in facts and facts["ml"].get("readiness_warnings"):
        highlights.append(facts["ml"]["readiness_warnings"][0])
    for item in facts.get("nlp", [])[:1]:
        if item.get("overall_sentiment"):
            highlights.append(f"Overall sentiment in '{item['column']}' is {item['overall_sentiment']}.")

    return {"summary": " ".join(parts), "highlights": highlights[:6], "source": "rule_based"}


def _gemini_summary(facts: dict) -> dict | None:
    try:
        from google import genai as google_genai

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            return None

        prompt_path = Path("prompts/report_summary.txt")
        if not prompt_path.exists():
            return None

        prompt = prompt_path.read_text(encoding="utf-8").replace(
            "{{REPORT_FACTS}}", json.dumps(_sanitize(facts), indent=2)
        )

        client = google_genai.Client(api_key=api_key)
        response = client.models.generate_content(model=settings.GEMINI_MODEL, contents=prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        parsed = json.loads(raw)
        summary = parsed.get("summary", "")
        highlights = parsed.get("highlights", [])
        if isinstance(summary, str) and len(summary) > 30:
            if not isinstance(highlights, list):
                highlights = []
            return {
                "summary": summary.strip(),
                "highlights": [str(h) for h in highlights][:6],
                "source": "gemini",
            }
    except Exception as e:
        logger.warning(f"Gemini executive summary failed: {e}")
    return None


async def collect_executive_summary(df: pd.DataFrame, collected: dict) -> dict:
    facts = _build_facts(df, collected)
    fallback = _rule_based_summary(facts)
    result = _gemini_summary(facts)
    if result is None:
        result = fallback
    elif not result["highlights"]:
        result["highlights"] = fallback["highlights"]
    return _section("executive_summary", True, result)


# Report assembly

async def _collect(key: str, session_id: str, df: pd.DataFrame) -> dict:
    if key == "profiling":
        return await collect_profiling(session_id, df)
    if key == "cleaning":
        return collect_cleaning(session_id)
    if key == "transformation":
        return collect_transformation(session_id)
    if key == "eda":
        return collect_eda(session_id, df)
    if key == "statistics":
        return collect_statistics(session_id)
    if key == "nlp":
        return await collect_nlp(session_id, df)
    if key == "ml_strategy":
        return collect_ml(session_id)
    if key == "history":
        return collect_history(session_id)
    if key == "appendix":
        return collect_appendix(session_id, df)
    raise ValueError(f"No collector for section '{key}'")


async def build_report(session_id: str, df: pd.DataFrame, sections: list[str] | None = None) -> dict:
    keys = _normalise_sections(sections)
    collected: dict = {}

    for key in keys:
        if key == "executive_summary":
            continue
        try:
            collected[key] = await _collect(key, session_id, df)
        except Exception as e:
            logger.exception(f"Report section '{key}' failed for session_id={session_id}: {e}")
            collected[key] = _not_performed(key, "This section could not be generated.")

    if "executive_summary" in keys:
        try:
            collected["executive_summary"] = await collect_executive_summary(df, collected)
        except Exception as e:
            logger.exception(f"Executive summary failed for session_id={session_id}: {e}")
            collected["executive_summary"] = _not_performed(
                "executive_summary", "The executive summary could not be generated."
            )

    ordered = [collected[k] for k in SECTION_KEYS if k in collected]
    for i, s in enumerate(ordered, start=1):
        s["number"] = i

    return _sanitize({
        "title": "DataSaathi Analysis Report",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": {"rows": len(df), "columns": len(df.columns)},
        "sections": ordered,
        "not_performed": [s["title"] for s in ordered if not s["performed"]],
    })


# Rendering & export

def render_report_html(report: dict) -> str:
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,  # column names / text values must never inject markup
    )
    return env.get_template(TEMPLATE_NAME).render(report=report)


_pdf_checked: bool | None = None


def pdf_available() -> bool:
    """
    True if WeasyPrint and its system libraries load. Result is cached
    """
    global _pdf_checked
    if _pdf_checked is None:
        try:
            import weasyprint  # noqa: F401
            _pdf_checked = True
        except Exception as e:  # ImportError, or OSError when GTK/Pango is missing
            logger.warning(f"PDF engine unavailable: {e}")
            _pdf_checked = False
    return _pdf_checked


def _block_fetch(*args, **kwargs):
    raise ValueError("External resources are blocked in reports.")


def _write_pdf(html: str, path: Path) -> None:
    from weasyprint import HTML

    HTML(string=html, url_fetcher=_block_fetch).write_pdf(str(path))


def _report_path(session_id: str, fmt: str) -> Path:
    return settings.reports_path / f"{session_id}_report.{fmt}"


def get_report_file(session_id: str, fmt: str) -> Path | None:
    """
    Return the generated file's path if it exists. The path is never exposed to clients
    """
    if fmt not in VALID_FORMATS:
        return None
    path = _report_path(session_id, fmt)
    return path if path.exists() else None


async def generate_report(session_id: str, sections: list[str] | None, fmt: str) -> dict:
    """
    Build, render and write the report.
    HTML is always written (the preview needs it, and PDF is rendered from it).
    A stale file of the other format is removed so it can't be downloaded by mistake.
    """
    fmt = (fmt or "html").lower()
    if fmt not in VALID_FORMATS:
        raise ValueError(f"Invalid format '{fmt}'. Valid formats: {sorted(VALID_FORMATS)}")
    if fmt == "pdf" and not pdf_available():
        raise PDFEngineUnavailable(
            "PDF export is not available on this server. Choose HTML instead, "
            "or install the WeasyPrint system libraries."
        )

    df = session_service.get_working_df(session_id)
    report = await build_report(session_id, df, sections)
    html = render_report_html(report)

    settings.reports_path.mkdir(parents=True, exist_ok=True)
    html_path = _report_path(session_id, "html")
    pdf_path = _report_path(session_id, "pdf")
    html_path.write_text(html, encoding="utf-8")

    if fmt == "pdf":
        await asyncio.to_thread(_write_pdf, html, pdf_path)
        out_path = pdf_path
    else:
        if pdf_path.exists():
            pdf_path.unlink()
        out_path = html_path

    logger.info(
        f"Report generated | session_id={session_id} | format={fmt} | "
        f"sections={len(report['sections'])} | not_performed={len(report['not_performed'])}"
    )

    return {
        "format": fmt,
        "generated_at": report["generated_at"],
        "file_size_kb": round(out_path.stat().st_size / 1024, 2),
        "sections_included": [
            {"key": s["key"], "title": s["title"], "completed": s["performed"]}
            for s in report["sections"]
        ],
        "sections_not_performed": report["not_performed"],
    }


# History export (JSON download)

def build_history_export(session_id: str) -> dict:
    return _sanitize({
        "session_id": session_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "operations": session_service.get_history(session_id),
        "eda_analyses": get_eda_history(session_id),
        "statistical_tests": get_stats_history(session_id),
        "nlp_analyses": get_nlp_history(session_id),
    })









