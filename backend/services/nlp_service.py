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

def tfidf_keywords(df: DataFrame, column: str, top_n: int = 20) -> list[dict]:
    """
    Extract top N keywords using TF-IDF.
    """
    if column not in df.columns: return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = df[column].fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            min_df=2,
            ngram_range=(1, 1),
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        top_indices = mean_scores.argsort()[::-1][:top_n]
        return [
            {
                "keyword": feature_names[i],
                "score": round(float(mean_scores[i]), 6),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
        ]

    except Exception as e:
        logger.warning(f"TF-IDF failed: {e}")
        return [{"error": str(e)}]


    
# 8. Word Frequency Count

def word_frequency(df: DataFrame, column:str, top_n: int = 30) -> list[dict]:


    if column not in df.columns: return [{"error": f"Column '{column}' not found."}]

    stopwords = _get_stopwords()
    all_tokens = []
    for text in df[column].fillna("").astype(str):
        all_tokens.extend(_clean_text(text, stopwords))

    counter = Counter(all_tokens)
    return [
        {"word": word, "count": count, "rank": rank + 1}
        for rank, (word, count) in enumerate(counter.most_common(top_n))
    ]

# 9. Word Cloud Data

def word_cloud_data(df: DataFrame, column: str, top_n:int = 50) -> list[dict]:

    freq = word_frequency(df, column, top_n=top_n)

    if freq and "error" in freq[0]: return freq

    max_count = max(item["count"] for item in freq) if freq else 1

    return [
        {
            "word": item["word"],
            "count": item["count"],
            "weight": round(item["count"] / max_count, 4),
        }
        for item in freq
    ]

# 10  N-Gram Extraction

def extract_ngrams(
    df: pd.DataFrame, column: str, n: int = 2, top_n: int = 20
) -> list[dict]:
    """Extract top bigrams (n=2) or trigrams (n=3)."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    stopwords = _get_stopwords()
    all_ngrams = []

    for text in df[column].fillna("").astype(str):
        tokens = _clean_text(text, stopwords)
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    counter = Counter(all_ngrams)
    return [
        {"ngram": ng, "count": count, "rank": rank + 1}
        for rank, (ng, count) in enumerate(counter.most_common(top_n))
    ]


# 11  LDA Topic Modelling

def lda_topics(
    df: pd.DataFrame, column: str, n_topics: int = 5, top_words: int = 8
) -> list[dict]:
    """Run LDA topic modelling and return top words per topic."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        corpus = df[column].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer(
            max_features=300,
            stop_words="english",
            min_df=2,
        )
        dtm = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
        )
        lda.fit(dtm)

        topics = []
        doc_topic = lda.transform(dtm)
        total_docs = len(corpus)

        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[::-1][:top_words]
            keywords = [feature_names[i] for i in top_indices]
            weights = [round(float(topic[i]), 4) for i in top_indices]

            # Proportion of documents where this topic dominates
            dominant_count = int((doc_topic.argmax(axis=1) == topic_idx).sum())

            topics.append({
                "topic_id": topic_idx,
                "keywords": keywords,
                "weights": weights,
                "proportion": round(dominant_count / total_docs * 100, 2),
            })

        return topics

    except Exception as e:
        logger.warning(f"LDA failed: {e}")
        return [{"error": str(e)}]


# 12  Topic-Keyword Mapping

def topic_keyword_map(
    df: pd.DataFrame, column: str, n_topics: int = 5
) -> dict:
    """Return a mapping of topic_id → keyword list for quick lookup."""
    topics = lda_topics(df, column, n_topics=n_topics)
    if topics and "error" in topics[0]:
        return topics[0]

    return {
        str(t["topic_id"]): t["keywords"]
        for t in topics
    }


# N3  Document-Topic Assignment

def document_topic_assignment(
    df: pd.DataFrame, column: str, n_topics: int = 5
) -> list[dict]:
    """Assign each document to its dominant topic."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        corpus = df[column].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer(
            max_features=300, stop_words="english", min_df=2
        )
        dtm = vectorizer.fit_transform(corpus)
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=10
        )
        doc_topic = lda.fit_transform(dtm)

        return [
            {
                "index": int(i),
                "dominant_topic": int(doc_topic[i].argmax()),
                "topic_score": round(float(doc_topic[i].max()), 4),
                "text_preview": corpus[i][:80],
            }
            for i in range(len(corpus))
        ]

    except Exception as e:
        return [{"error": str(e)}]


#   14  Sentence Embeddings (sampled)

def get_embeddings(
    df: pd.DataFrame, column: str
) -> dict:
    """
    Compute sentence embeddings using all-MiniLM-L6-v2.
    Sampled to EMBEDDING_SAMPLE_CAP rows for performance.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].fillna("").astype(str)
    sampled = s.sample(
        min(EMBEDDING_SAMPLE_CAP, len(s)), random_state=42
    )

    try:
        embedder = _get_embedder()
        embeddings = embedder.encode(sampled.tolist(), show_progress_bar=False)
        return {
            "indices": sampled.index.tolist(),
            "embeddings": embeddings.tolist(),
            "n_sampled": len(sampled),
            "embedding_dim": embeddings.shape[1],
        }
    except Exception as e:
        return {"error": str(e)}


# 15  Semantic Similarity Search

def semantic_similarity_search(
    df: pd.DataFrame, column: str, query: str, top_n: int = 5
) -> list[dict]:
    """Find the top N most semantically similar records to a query."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        s = df[column].fillna("").astype(str)
        sampled = s.sample(
            min(EMBEDDING_SAMPLE_CAP, len(s)), random_state=42
        )

        embedder = _get_embedder()
        doc_embeddings = embedder.encode(sampled.tolist(), show_progress_bar=False)
        query_embedding = embedder.encode([query], show_progress_bar=False)

        sims = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = sims.argsort()[::-1][:top_n]

        return [
            {
                "rank": rank + 1,
                "index": int(sampled.index[i]),
                "text": sampled.iloc[i][:200],
                "similarity": round(float(sims[i]), 4),
            }
            for rank, i in enumerate(top_indices)
        ]

    except Exception as e:
        return [{"error": str(e)}]


# 16  Most Similar Pairs

def most_similar_pairs(
    df: pd.DataFrame, column: str, top_n: int = 5
) -> list[dict]:
    """Find the top N most similar text pairs in the column."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        s = df[column].fillna("").astype(str)
        sampled = s.sample(min(200, len(s)), random_state=42)

        embedder = _get_embedder()
        embeddings = embedder.encode(sampled.tolist(), show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        pairs = []
        n = len(sampled)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, sim_matrix[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:top_n]

        return [
            {
                "rank": rank + 1,
                "text_a": sampled.iloc[i][:150],
                "text_b": sampled.iloc[j][:150],
                "similarity": round(float(sim), 4),
            }
            for rank, (i, j, sim) in enumerate(top_pairs)
        ]

    except Exception as e:
        return [{"error": str(e)}]


#  17   K-Means Text Clustering

def cluster_texts(
    df: pd.DataFrame, column: str, n_clusters: int = 5
) -> list[dict]:
    """Cluster texts using K-Means on sentence embeddings."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.cluster import KMeans

        s = df[column].fillna("").astype(str)
        sampled = s.sample(
            min(EMBEDDING_SAMPLE_CAP, len(s)), random_state=42
        )

        embedder = _get_embedder()
        embeddings = embedder.encode(sampled.tolist(), show_progress_bar=False)

        n_clusters = min(n_clusters, len(sampled))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        clusters = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_texts = [
                sampled.iloc[i] for i in range(len(sampled)) if mask[i]
            ]
            clusters.append({
                "cluster_id": cluster_id,
                "size": int(mask.sum()),
                "sample_texts": [t[:100] for t in cluster_texts[:3]],
            })

        return clusters

    except Exception as e:
        return [{"error": str(e)}]