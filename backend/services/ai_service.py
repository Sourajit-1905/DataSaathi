"""
AI Service
----------
The conversational layer of DataSaathi.

Responsibilities
  - Build a privacy-safe context summary from the session's cached results:
    never raw rows, only computed statistics and column names.
  - Answer chat messages via Gemini (last 10 turns in context), with a
    rule-based fallback that covers the most common intents.
  - Generate structured insight cards from whatever stages have been run.
  - Explain a term, a column or a specific operation in plain English.
  - Summarise the whole analysis (delegates to the Stage 9 executive summary).
  - Return stage-aware suggested questions (no Gemini call).
  - Enforce a rate limit of 20 messages per minute per session.

Data policy: Gemini only sees column names, counts and computed statistics.
Cell values and category labels are sent only when explicitly constructing a
context summary (as a limited, delimited sample). A "TREAT AS DATA ONLY"
prefix precedes any user content that reaches the API.

All Gemini calls fall back to rule-based answers on any failure, so the
assistant is always responsive.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from core import cache
from core.gemini_client import generate_json, render_prompt

logger = logging.getLogger(__name__)

MAX_QUESTION_LEN  = 1000
MAX_CONTEXT_TURNS = 10
MAX_MESSAGES      = 200
RATE_LIMIT_PER_MINUTE = 20
RATE_WINDOW_SECONDS   = 60

# Rate limiting

class RateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")


def _check_rate_limit(session_id: str) -> None:
    """Raise RateLimitError if this session has sent 20+ messages in the last 60 seconds."""
    now = time.monotonic()
    times = cache.get_chat_times(session_id)
    # evict old entries in place
    times[:] = [t for t in times if now - t < RATE_WINDOW_SECONDS]
    if len(times) >= RATE_LIMIT_PER_MINUTE:
        oldest = times[0]
        retry_after = int(RATE_WINDOW_SECONDS - (now - oldest)) + 1
        raise RateLimitError(retry_after)
    times.append(now)


# Context builder

def build_context(session_id: str, df=None) -> dict:
    """
    Return a privacy-safe summary of everything known about this session.
    df is the current working DataFrame (rows, columns, dtypes and a tiny
    sample of column names only — never cell values from text columns).
    """
    ctx: dict = {}

    if df is not None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        ctx["dataset"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist()[:50],
            "numerical_columns": num_cols[:20],
            "categorical_columns": cat_cols[:20],
            "missing_counts": {
                col: int(df[col].isna().sum())
                for col in df.columns
                if df[col].isna().sum() > 0
            },
        }

    profile = cache.get_profile(session_id)
    if profile:
        q = profile.get("quality_summary", {})
        ctx["profile"] = {
            "quality_score": q.get("score"),
            "quality_label": q.get("label"),
            "insights": list(profile.get("insights", []))[:4],
            "duplicate_rows": profile.get("duplicate_rows", {}).get("count"),
            "potential_targets": [
                t.get("column") for t in profile.get("potential_targets", [])
            ][:5],
        }

    from core.cache import get_eda_history, get_stats_history, get_nlp_history, get_ml_result
    from services import session_service

    history = session_service.get_history(session_id)
    clean_ops = [o for o in history if o.get("stage") == "clean"]
    transform_ops = [o for o in history if o.get("stage") == "transform"]

    if clean_ops or transform_ops:
        ctx["preparation"] = {
            "cleaning_steps": len(clean_ops),
            "transform_steps": len(transform_ops),
            "operations_summary": [
                f"{o.get('issue_type', o.get('transform_type', '?'))} on {o.get('column', 'dataset')}"
                for o in history[:10]
            ],
        }

    eda_history = get_eda_history(session_id)
    if eda_history:
        ctx["eda"] = {
            "analyses_run": len(eda_history),
            "analysis_types": list({h.get("analysis_type") for h in eda_history}),
        }

    stats_history = get_stats_history(session_id)
    if stats_history:
        ctx["statistics"] = {
            "tests_run": len(stats_history),
            "significant_tests": [
                h.get("test_name") for h in stats_history if h.get("significant")
            ][:5],
        }

    nlp_history = get_nlp_history(session_id)
    if nlp_history:
        ctx["nlp"] = {
            "analyses_run": len(nlp_history),
            "columns_analysed": list({h.get("column") for h in nlp_history if h.get("column")}),
        }

    ml = get_ml_result(session_id)
    if ml and ml.get("valid"):
        problem = ml.get("problem", {})
        ctx["ml_strategy"] = {
            "problem_type": problem.get("problem_type"),
            "target": problem.get("target"),
            "top_model": (ml.get("models") or [{}])[0].get("name"),
            "primary_metric": ml.get("metrics", {}).get("primary"),
            "readiness": ml.get("readiness", {}).get("label"),
        }

    return ctx


def _context_fingerprint(session_id: str) -> str:
    """A short hash that changes whenever any analysis result changes."""
    from core.cache import get_eda_history, get_stats_history, get_nlp_history, get_ml_result
    from services import session_service
    parts = [
        len(session_service.get_history(session_id)),
        len(get_eda_history(session_id)),
        len(get_stats_history(session_id)),
        len(get_nlp_history(session_id)),
        bool(cache.get_profile(session_id)),
        bool(get_ml_result(session_id)),
    ]
    return hashlib.md5(json.dumps(parts).encode()).hexdigest()[:12]


# Rule-based intent detection and answers

def _intent(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in ["how many row", "how many column", "shape", "size", "big is"]):
        return "dataset_size"
    if any(kw in t for kw in ["what column", "which column", "list column", "columns are"]):
        return "columns"
    if any(kw in t for kw in ["missing", "null", "nan", "empty"]):
        return "missing"
    if any(kw in t for kw in ["quality", "score", "issue"]):
        return "quality"
    if any(kw in t for kw in ["duplicate", "repeated", "same row"]):
        return "duplicates"
    if any(kw in t for kw in ["what have", "what was done", "what did", "history", "applied", "cleaned", "transformed"]):
        return "history"
    if any(kw in t for kw in ["what should", "recommend", "suggest", "next step", "what to do"]):
        return "next_step"
    if any(kw in t for kw in ["target", "predict", "ml", "model", "machine learning"]):
        return "ml_target"
    if any(kw in t for kw in ["eda", "explor", "analysis", "analys"]):
        return "eda_summary"
    if any(kw in t for kw in ["statistic", "test", "significant", "p-value", "p value"]):
        return "stats_summary"
    return "unknown"


def _rule_based_reply(question: str, ctx: dict) -> str:
    intent = _intent(question)
    ds = ctx.get("dataset", {})
    prep = ctx.get("preparation", {})

    if intent == "dataset_size" and ds:
        return f"Your dataset has {ds['rows']:,} rows and {ds['columns']} columns."

    if intent == "columns" and ds:
        cols = ds.get("column_names", [])
        return (f"Your dataset has {len(cols)} columns: {', '.join(cols[:20])}"
                + (" (and more)" if len(cols) > 20 else "") + ".")

    if intent == "missing":
        missing = ds.get("missing_counts", {})
        if not missing:
            return "No missing values were found in the current working dataset."
        lines = [f"  • {col}: {count:,}" for col, count in list(missing.items())[:10]]
        return "Missing values:\n" + "\n".join(lines)

    if intent == "quality":
        p = ctx.get("profile", {})
        if p:
            return (f"Data quality is {p.get('quality_label', 'unknown')} "
                    f"(score {p.get('quality_score', '?')}/100).")
        return "Run the Profiling stage to see a quality score."

    if intent == "duplicates":
        p = ctx.get("profile", {})
        dups = p.get("duplicate_rows")
        if dups is not None:
            return (f"There are {dups:,} duplicate rows in the original dataset."
                    if dups else "No duplicate rows were found.")
        return "Run the Profiling stage to check for duplicates."

    if intent == "history" and prep:
        parts = []
        if prep.get("cleaning_steps"):
            parts.append(f"{prep['cleaning_steps']} cleaning step(s)")
        if prep.get("transform_steps"):
            parts.append(f"{prep['transform_steps']} transformation(s)")
        summary = " and ".join(parts) + " have been applied."
        ops = prep.get("operations_summary", [])
        if ops:
            summary += " Including: " + "; ".join(ops[:5]) + "."
        return summary
    if intent == "history":
        return "No cleaning or transformations have been applied yet."

    if intent == "ml_target":
        ml = ctx.get("ml_strategy", {})
        if ml:
            return (f"The ML task is framed as {ml['problem_type']} "
                    f"with '{ml['target']}' as the target. "
                    f"The top recommended starting model is {ml['top_model']}.")
        return "No ML strategy has been generated yet. Use the ML Strategy tab to get recommendations."

    if intent == "eda_summary":
        eda = ctx.get("eda", {})
        if eda:
            return (f"{eda['analyses_run']} EDA analyses have been run, "
                    f"covering: {', '.join(eda.get('analysis_types', []))}.")
        return "No EDA analyses have been run yet. Use the Explore tab to start."

    if intent == "stats_summary":
        stats = ctx.get("statistics", {})
        if stats:
            sig = stats.get("significant_tests", [])
            return (f"{stats['tests_run']} statistical test(s) run. "
                    + (f"Significant: {', '.join(sig)}." if sig else "None were significant at 5%."))
        return "No statistical tests have been run yet. Use the Statistics tab."

    if intent == "next_step":
        suggestions = []
        if not ctx.get("profile"):
            suggestions.append("Run Profiling to understand your data quality.")
        elif not ctx.get("preparation"):
            suggestions.append("Review the Cleaning suggestions based on the profile.")
        if not ctx.get("eda"):
            suggestions.append("Explore distributions and correlations in the EDA tab.")
        if not ctx.get("statistics"):
            suggestions.append("Test your hypotheses in the Statistics tab.")
        if not ctx.get("ml_strategy"):
            suggestions.append("Generate an ML Strategy when you are ready to model.")
        if not suggestions:
            suggestions.append("Generate a Report to package your findings.")
        return "Suggested next steps:\n" + "\n".join(f"  • {s}" for s in suggestions)

    return ("I can answer questions about your dataset size, columns, missing values, quality, "
            "duplicates, what's been done, next steps, and the ML strategy — "
            "either with AI or from the cached results. "
            "Try: \"How many rows are there?\", \"What columns are missing data?\", "
            "or \"What should I do next?\"")


# Chat

async def chat(
    session_id: str,
    question: str,
    df=None,
) -> dict:
    """
    Append the user's message, call Gemini with up to the last 10 turns, store
    the reply and return it. Falls back to rule-based on any Gemini failure.
    """
    from services import session_service

    question = (question or "").strip()[:MAX_QUESTION_LEN]
    _check_rate_limit(session_id)

    ctx = build_context(session_id, df)
    rule_answer = _rule_based_reply(question, ctx)

    user_msg = {
        "role": "user",
        "content": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    thread = session_service.append_chat(session_id, [user_msg], MAX_MESSAGES)

    answer, source = rule_answer, "rule_based"

    # Build conversation for Gemini (last MAX_CONTEXT_TURNS turns before this question)
    prior = thread[-(MAX_CONTEXT_TURNS * 2 + 1):-1]   # exclude the message just added
    conversation = [{"role": m["role"], "content": m["content"]} for m in prior]

    prompt = render_prompt(
        "chat.txt",
        CONTEXT=json.dumps(ctx, indent=2, default=str),
        CONVERSATION=json.dumps(conversation, indent=2),
        QUESTION=question,
    )
    parsed = None
    if prompt:
        try:
            parsed = await generate_json(prompt)
        except Exception as _e:
            logger.warning(f"Gemini chat call failed: {_e}")
        if parsed and isinstance(parsed.get("answer"), str) and len(parsed["answer"]) > 10:
            answer = parsed["answer"].strip()
            source = "gemini"

    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    thread = session_service.append_chat(session_id, [assistant_msg], MAX_MESSAGES)

    return {
        "answer": answer,
        "source": source,
        "suggested_tab": parsed.get("suggested_tab") if source == "gemini" and parsed else _suggest_tab(ctx, question),
        "thread_length": len(thread),
    }


def _suggest_tab(ctx: dict, question: str) -> str | None:
    """Very cheap tab suggestion from the same intent classifier, for the rule-based path."""
    intent = _intent(question)
    mapping = {
        "missing": "clean",
        "quality": "profile",
        "duplicates": "clean",
        "eda_summary": "eda",
        "stats_summary": "stats",
        "ml_target": "ml",
        "next_step": None,
    }
    return mapping.get(intent)


# Insights

def _build_insight_cards(session_id: str, ctx: dict) -> list[dict]:
    """Rule-based insight cards: one per notable finding, with a stage tag."""
    cards = []

    p = ctx.get("profile", {})
    if p.get("quality_score") is not None and p["quality_score"] < 70:
        cards.append({"kind": "warning", "stage": "profile", "tab": "profile",
                      "title": "Low data quality",
                      "body": f"Quality score is {p['quality_score']}/100 ({p.get('quality_label', '?')}). "
                              f"Check the profile for issues before proceeding."})

    ds = ctx.get("dataset", {})
    missing = ds.get("missing_counts", {})
    rows = max(ds.get("rows", 1), 1)
    heavy = [(c, n) for c, n in missing.items() if n / rows > 0.2]
    for col, count in heavy[:3]:
        cards.append({"kind": "warning", "stage": "clean", "tab": "clean",
                      "title": f"Heavy missing values in '{col}'",
                      "body": f"'{col}' is missing {count:,} values "
                              f"({count / rows:.0%}). "
                              f"Consider imputing or dropping it."})

    stats = ctx.get("statistics", {})
    for test in stats.get("significant_tests", [])[:3]:
        cards.append({"kind": "insight", "stage": "stats", "tab": "stats",
                      "title": f"Significant result: {test}",
                      "body": f"The {test} test produced a statistically significant result. "
                              f"See the Statistics tab for details."})

    ml = ctx.get("ml_strategy", {})
    if ml.get("readiness") and ml["readiness"] not in ("ready", "good"):
        cards.append({"kind": "warning", "stage": "ml", "tab": "ml",
                      "title": "Dataset readiness concerns",
                      "body": f"Readiness is '{ml['readiness']}'. "
                              f"Resolve the warnings in the ML Strategy tab before modelling."})

    prep = ctx.get("preparation", {})
    if not prep and ctx.get("profile"):
        cards.append({"kind": "tip", "stage": "clean", "tab": "clean",
                      "title": "No cleaning applied yet",
                      "body": "Profiling is done but no cleaning steps have been applied. "
                              "Review the recommendations in the Clean tab."})

    # Show the empty-state card whenever nothing meaningful was found
    meaningful = [c for c in cards if c.get("kind") != "tip" or c.get("stage") != "clean"]
    if not meaningful:
        cards = [{"kind": "tip", "stage": "overview", "tab": None,
                  "title": "Nothing to report yet",
                  "body": "Run Profiling, EDA or Statistics to generate insight cards."}]

    return cards


async def get_insights(session_id: str, df=None, refresh: bool = False) -> dict:
    """
    Return structured insight cards. Results are cached by a fingerprint of the
    analysis results; refresh=True forces regeneration.
    """
    fingerprint = _context_fingerprint(session_id)
    cached = cache.get_ai_insights(session_id)
    if not refresh and cached and cached.get("fingerprint") == fingerprint:
        return cached

    ctx = build_context(session_id, df)
    rule_cards = _build_insight_cards(session_id, ctx)
    cards, source = rule_cards, "rule_based"

    prompt = render_prompt(
        "ai_insights.txt",
        CONTEXT=json.dumps(ctx, indent=2, default=str),
    )
    if prompt:
        parsed = await generate_json(prompt)
        if parsed and isinstance(parsed.get("cards"), list) and len(parsed["cards"]) >= 1:
            raw_cards = parsed["cards"][:10]
            validated = []
            for c in raw_cards:
                if isinstance(c.get("title"), str) and isinstance(c.get("body"), str):
                    validated.append({
                        "kind": c.get("kind", "insight"),
                        "stage": c.get("stage", "overview"),
                        "tab": c.get("tab"),
                        "title": c["title"][:120],
                        "body": c["body"][:500],
                    })
            if validated:
                cards, source = validated, "gemini"

    result = {
        "fingerprint": fingerprint,
        "cards": cards,
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    cache.set_ai_insights(session_id, result)
    return result


# Explain

TERM_GLOSSARY = {
    "skewness":    "A measure of how lopsided a distribution is. 0 = symmetric. Positive = long right tail. Negative = long left tail.",
    "kurtosis":    "How heavy the tails of a distribution are. High kurtosis means more extreme outliers than a normal distribution.",
    "outlier":     "A data point far from the rest of the data. Typically defined as more than 1.5 × IQR from the nearest quartile.",
    "iqr":         "The interquartile range: the distance from the 25th to the 75th percentile. A robust measure of spread.",
    "p-value":     "The probability of seeing a result at least this extreme if the null hypothesis were true. Lower = more surprising.",
    "correlation": "How closely two variables move together. Ranges from −1 (perfect negative) to 1 (perfect positive). 0 = no linear relationship.",
    "missing":     "A cell with no value (NaN / None). Missing values must be handled before many analyses and almost all ML models.",
    "imputation":  "Filling in missing values with a substitute such as the mean, median, mode, or a model prediction.",
    "encoding":    "Converting a categorical column (text) to numbers so an ML model can use it.",
    "normalisation": "Scaling a column so all values fall in the same range, usually 0–1 (min-max) or mean 0, std 1 (z-score).",
    "roc-auc":     "A metric for binary classification. 1.0 = perfect, 0.5 = random guessing. AUC is the area under the ROC curve.",
    "precision":   "Of all the rows predicted as positive, what fraction actually are positive. Trade-off with recall.",
    "recall":      "Of all actual positives, what fraction were predicted as positive. Trade-off with precision.",
    "f1":          "The harmonic mean of precision and recall. Useful when both matter and the classes are imbalanced.",
    "rmse":        "Root-mean-squared error for regression. Same units as the target. Penalises large errors heavily.",
    "mae":         "Mean absolute error for regression. Same units as the target. Treats all errors equally.",
    "overfit":     "When a model learns the training data too closely and performs poorly on new data.",
    "underfit":    "When a model is too simple to capture the patterns in the data.",
    "gradient boosting": "An ensemble of decision trees built sequentially, each one correcting the errors of the last.",
    "random forest":     "An ensemble of decision trees trained on random subsets of the data. Averages their predictions.",
    "logistic regression": "A linear model for binary or multiclass classification. Outputs probabilities.",
    "cross-validation":  "Splitting the data into k folds and training k models, each on a different k−1 folds. Gives a more reliable estimate of generalisation.",
    "smote":       "Synthetic Minority Oversampling: creates new synthetic examples of the minority class to help with class imbalance.",
    "lda":         "Latent Dirichlet Allocation — a topic model that finds groups of co-occurring words (topics) in a text column.",
    "tfidf":       "Term frequency–inverse document frequency. A measure of how important a word is to one document relative to the whole collection.",
    "sentiment":   "An assessment of whether text is positive, negative or neutral, typically using VADER or a transformer model.",
    "embeddings":  "Dense numerical vectors that represent text so that semantically similar texts are close in the vector space.",
}


def _glossary_lookup(term: str) -> str | None:
    t = term.lower().strip()
    if t in TERM_GLOSSARY:
        return TERM_GLOSSARY[t]
    for key, val in TERM_GLOSSARY.items():
        if key in t or t in key:
            return val
    return None


def _rule_based_explain(kind: str, target: str, ctx: dict) -> str:
    if kind == "term":
        hit = _glossary_lookup(target)
        return hit or f"'{target}' is a technical term I don't have a built-in explanation for. Enable AI to get a plain-English definition."

    if kind == "column":
        ds = ctx.get("dataset", {})
        cols = ds.get("column_names", [])
        if target not in cols:
            return f"Column '{target}' is not in the current working dataset."
        miss = ds.get("missing_counts", {}).get(target, 0)
        num_cols = ds.get("numerical_columns", [])
        cat_cols = ds.get("categorical_columns", [])
        dtype = "numerical" if target in num_cols else ("categorical" if target in cat_cols else "unknown type")
        return (f"'{target}' is a {dtype} column with {miss:,} missing value(s). "
                f"Use the EDA tab to explore its distribution.")

    if kind == "operation":
        return (f"The operation '{target}' modifies the working dataset. "
                f"See the History section or enable AI for a detailed explanation.")

    return f"I don't have a built-in explanation for '{target}'. Enable AI for a detailed plain-English explanation."


async def explain(
    session_id: str,
    kind: str,
    target: str,
    extra: dict | None = None,
    df=None,
) -> dict:
    """
    Explain a term, column or operation in plain English.
    kind: 'term' | 'column' | 'operation'
    target: what to explain (column name, term text, or operation type)
    extra: optional extra payload (e.g. stats from a recent test result)
    """
    kind = (kind or "term").lower().strip()
    target = (target or "").strip()[:200]

    ctx = build_context(session_id, df)
    rule_text = _rule_based_explain(kind, target, ctx)

    prompt = render_prompt(
        "ai_explain.txt",
        KIND=kind,
        TARGET=target,
        EXTRA=json.dumps(extra or {}, indent=2, default=str),
        CONTEXT=json.dumps(ctx, indent=2, default=str),
    )
    if prompt:
        try:
            parsed = await generate_json(prompt)
        except Exception as _e:
            logger.warning(f"Gemini explain call failed: {_e}")
            parsed = None
        if parsed and isinstance(parsed.get("explanation"), str) and len(parsed["explanation"]) > 20:
            return {"explanation": parsed["explanation"].strip(), "source": "gemini"}

    return {"explanation": rule_text, "source": "rule_based"}


# Summarize (delegates to Stage 9)

async def summarize(session_id: str, df=None) -> dict:
    """
    Return a short narrative summary of the analysis.
    Reuses the Stage 9 executive-summary machinery: only the 'executive_summary'
    section is collected, so nothing heavy is recomputed.
    """
    try:
        from services import report_service
        report = await report_service.build_report(session_id, df, ["executive_summary"])
        sec = report["sections"][0]
        return {
            "summary": sec["data"].get("summary", ""),
            "highlights": sec["data"].get("highlights", []),
            "source": sec["data"].get("source", "rule_based"),
        }
    except Exception as e:
        logger.warning(f"Summarize failed for session_id={session_id}: {e}")
        ctx = build_context(session_id, df)
        return {
            "summary": _rule_based_reply("what have you done", ctx),
            "highlights": [],
            "source": "rule_based",
        }


# Suggestions (no Gemini call: instant, stage-aware)

def get_suggestions(session_id: str) -> list[dict]:
    """
    Return up to 6 starter questions. They are chosen based on what has and
    hasn't been done yet, and they never call Gemini.
    """
    from core.cache import get_eda_history, get_stats_history, get_ml_result
    from services import session_service

    history = session_service.get_history(session_id)
    has_profile = cache.get_profile(session_id) is not None
    has_prep = bool(history)
    has_eda = bool(get_eda_history(session_id))
    has_stats = bool(get_stats_history(session_id))
    has_ml = bool(get_ml_result(session_id))

    chips = []

    chips.append({"text": "How many rows and columns does my dataset have?", "intent": "dataset_size"})
    chips.append({"text": "Which columns have missing values?", "intent": "missing"})

    if not has_profile:
        chips.append({"text": "What is the data quality like?", "intent": "quality"})
    else:
        chips.append({"text": "Are there any duplicates?", "intent": "duplicates"})

    if has_prep:
        chips.append({"text": "What cleaning and transforms have been applied?", "intent": "history"})
    else:
        chips.append({"text": "What should I do next?", "intent": "next_step"})

    if has_eda:
        chips.append({"text": "Summarise the EDA findings so far.", "intent": "eda_summary"})
    if has_stats:
        chips.append({"text": "Which statistical tests were significant?", "intent": "stats_summary"})
    if has_ml:
        chips.append({"text": "What ML model is recommended?", "intent": "ml_target"})
    if not chips[-1:] or chips[-1].get("intent") not in ("next_step",):
        chips.append({"text": "What should I do next?", "intent": "next_step"})

    return chips[:6]


# Status (for the badge)

def get_status() -> dict:
    """Safe-to-expose AI status. The key is never included."""
    from core.gemini_client import gemini_status
    status = gemini_status()
    return {
        "ai_enabled": status["configured"] and status["sdk_installed"] and not status["cooling_down"],
        "configured": status["configured"],
        "sdk_installed": status["sdk_installed"],
        "cooling_down": status["cooling_down"],
        "retry_after_seconds": status["retry_after_seconds"],
        "model": status["model"],
        "mode": (
            "cooling_down" if status["cooling_down"]
            else ("ai" if status["configured"] and status["sdk_installed"] else "rule_based")
        ),
    }