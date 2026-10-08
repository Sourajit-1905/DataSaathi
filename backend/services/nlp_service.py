"""
NLP Service
-----------
Implements 27 NLP functions across 7 categories:

  QUALITY          N1  Text column detection
                   N2  Text quality assessment
                   N3  Language detection

  SENTIMENT        N4  VADER sentiment scoring (per row)
                   N5  Sentiment aggregation (pos/neg/neu counts)
                   N6  Sentiment distribution

  KEYWORDS         N7  TF-IDF keyword extraction
                   N8  Word frequency count
                   N9  Word cloud data
                   N10 N-gram extraction (bigrams/trigrams)

  TOPICS           N11 LDA topic modelling
                   N12 Topic-keyword mapping
                   N13 Document-topic assignment

  SIMILARITY       N14 Sentence embedding (sampled)
                   N15 Semantic similarity search
                   N16 Most similar pairs

  CLUSTERING       N17 K-Means clustering on embeddings
                   N18 Cluster keyword summary
                   N19 Cluster size distribution

  INSIGHTS         N20 Rule-based NLP insights
                   N21 Gemini NLP narrative
                   N22 Text statistics (avg length, vocab size)
                   N23 Readability estimate
                   N24 Named entity pattern detection
                   N25 Duplicate text detection
                   N26 Empty/noise detection
                   N27 NLP summary for report
"""

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import settings

logger = logging.getLogger(__name__)

# Sampling cap for embedding operations
EMBEDDING_SAMPLE_CAP = 2000


# Lazy imports — heavy libraries loaded only when needed

def _get_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _get_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except Exception:
            return set()


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# Gemini insights helper

async def _get_gemini_nlp_insights(summary: dict) -> list[str] | None:
    try:
        from google import genai as google_genai

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            return None

        prompt_path = Path("prompts/nlp_insights.txt")
        if not prompt_path.exists():
            return None

        prompt_template = prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.replace(
            "{{NLP_SUMMARY}}", json.dumps(summary, indent=2)
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
        insights = parsed.get("insights", [])
        if isinstance(insights, list) and len(insights) >= 2:
            return insights

    except Exception as e:
        logger.warning(f"Gemini NLP insights failed: {e}")

    return None




from pandas import DataFrame

#   1. Text Column Detection
def detect_text_columns(df:DataFrame) -> list[dict]:
    """
    Identify columns that contain free-form text
    (avg word count > 3 and unique ratio > 0.3).
    """
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].dropna().astype(str)
        if(len(s) == 0): continue

        avg_words = s.str.split().str.len().mean()
        unique_ratio = s.nunique()/len(s)

        avg_length = s.str.len().mean()

        is_text = avg_words > 3 and unique_ratio > 0.3

        results.append({
            "column": col,
            "is_text_column": bool(is_text),
            "avg_word_count": round(float(avg_words), 2),
            "avg_length": round(float(avg_length), 2),
            "unique_ratio": round(float(unique_ratio), 4),
            "sample": s.head(2).tolist(),
        })

    return results

# 2. Text Quality Assessment

def text_quality_assessment(df: DataFrame, column: str) -> dict:
    """
    Assess text quality: empty, duplicate, very short, noise.
    Returns a quality score out of 100.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].astype(str)
    total = len(s)

    empty_mask = df[column].isna() | (s.str.strip() == "") | (s == "nan")
    empty_count = int(empty_mask.sum())

    duplicate_count = int(s[~empty_mask].duplicated().sum())
    very_short = int((s[~empty_mask].str.split().str.len() <= 1).sum())

    noise_patterns = r"^[\W\d\s]+$"
    noise_count = int(s[~empty_mask].str.match(noise_patterns).sum())

    quality_score = 100
    quality_score -= min(30, int(empty_count / total * 100))
    quality_score -= min(20, int(duplicate_count / total * 100))
    quality_score -= min(15, int(very_short / total * 100))
    quality_score -= min(15, int(noise_count / total * 100))
    quality_score = max(0, quality_score)

    return {
        "column": column,
        "total": total,
        "empty_count": empty_count,
        "empty_pct": round(empty_count / total * 100, 2),
        "duplicate_count": duplicate_count,
        "duplicate_pct": round(duplicate_count / total * 100, 2),
        "very_short_count": very_short,
        "noise_count": noise_count,
        "quality_score": quality_score,
        "quality_label": (
            "excellent" if quality_score >= 80
            else "good" if quality_score >= 60
            else "fair" if quality_score >= 40
            else "poor"
        ),
    }

# 3. Language Detection

def detect_language(df: DataFrame, column: str)->dict:
    """Detect the dominant language in a text column."""
    if column not in df.columns: return {"error":f"Column '{column}' not found."}

    try:
        from langdetect import detect, LangDetectException

        s = df[column].dropna().astype(str)

        sample = s.sample(min(100, len(s)), random_state=42)

        lang_counts = {}

        for text in sample:
            try:
                lang = detect(text)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            except LangDetectException: pass

        if not lang_counts: return {"column": column, "dominant_language": "unknown", "language_counts": {}}

        dominant = max(lang_counts, key=lang_counts.get)
        
        return {
            "column": column,
            "dominant_language": dominant,
            "language_counts": lang_counts,
            "sample_size": len(sample),
        }

    
    except ImportError:
        return {"column": column, "dominant_language": "unknown", "error": "langdetect not installed."}


# Text cleaning helper

def _clean_text(text: str, stopwords: set) -> list[str]:
    """Lowercase, remove punctuation, remove stopwords, return token list."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in stopwords and len(t) > 2]


# 4. VADER Sentiment Scoring

def vader_sentiment_scores(df: DataFrame, column: str)-> list[dict]:
    """
    Score each row with VADER. Returns per-row scores.
    """
    if column not in df.columns: return [{"error": f"Column '{column}' not found."}]

    analyzer = _get_vader()
    results = []

    for idx, text in df[column].fillna("").astype(str).items():
        scores = analyzer.polarity_scores(text)
        label = ("positive" if scores["compound"] >= 0.05
                 else "negative" if scores["compound"] <= -0.05
                 else "neutral")

        results.append({
            "index": int(idx),
            "text_preview": text[:100],
            "compound": round(float(scores["compound"]), 4),
            "positive": round(float(scores["pos"]), 4),
            "negative": round(float(scores["neg"]), 4),
            "neutral": round(float(scores["neu"]), 4),
            "label": label,
        })

    return results
    

# 5. Sentiment Aggregation

def sentiment_aggregation(df: pd.DataFrame, column: str) -> dict:
    """
    Aggregate sentiment across all rows.
    """

    if column not in df.columns: return {"error":f"Column '{column}' not found."}

    scores = vader_sentiment_scores(df, column)
    if scores and "error" in scores[0]: return scores[0]

    total =len(scores)

    positive = sum(1 for s in scores if s["label"] == "positive")
    negative = sum(1 for s in scores if s["label"] == "negative")
    neutral = sum(1 for s in scores if s["label"] == "neutral")

    avg_compound = sum(s["compound"] for s in scores) / total if total > 0 else 0

    overall = (
        "positive" if avg_compound >= 0.05
        else "negative" if avg_compound <= -0.05
        else "neutral"
    )

    return {
        "column": column,
        "total": total,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "positive_pct": round(positive / total * 100, 2) if total > 0 else 0,
        "negative_pct": round(negative / total * 100, 2) if total > 0 else 0,
        "neutral_pct": round(neutral / total * 100, 2) if total > 0 else 0,
        "avg_compound": round(float(avg_compound), 4),
        "overall_sentiment": overall,
    }


# 6. Sentiment Distribution

def sentiment_distribution(df: DataFrame, column: str)->dict:
    """
    Histogram of compound scores in 10 bins from -1 to 1.
    """

    if column not in df.columns: return {"error": f"Column '{column}' not found."}

    analyzer = _get_vader()
    compounds = []

    for text in df[column].fillna("").astype(str):
        scores = analyzer.polarity_scores(text)
        compounds.append(scores["compound"])

    bins = np.linspace(-1, 1, 11)

    counts, edges = np.histogram(compounds, bins= bins)

    return {
        "column": column,
        "histogram": {
            "counts": counts.tolist(),
            "bin_edges": [round(float(e), 2) for e in edges.tolist()],
        },
        "mean_compound": round(float(np.mean(compounds)), 4),
        "std_compound": round(float(np.std(compounds)), 4),
    }


# 7. TF-IDF Keyword Extraction