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


from core.gemini_client import generate_json, render_prompt

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
LDA_SAMPLE_CAP = 5000

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
        prompt = render_prompt(
            "nlp_insights.txt",
            NLP_SUMMARY=json.dumps(summary, indent=2, default=str),
        )
        if prompt is None:
            return None

        parsed = await generate_json(prompt)
        if not parsed:
            return None

        insights = parsed.get("insights", [])
        if isinstance(insights, list) and len(insights) >= 2:
            return [str(i) for i in insights]

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
def _lda_corpus(df: pd.DataFrame, column: str) -> pd.Series:
    """Text of a column for topic modelling, randomly sampled when it is very large."""
    s = df[column].fillna("").astype(str)
    if len(s) > LDA_SAMPLE_CAP:
        s = s.sample(LDA_SAMPLE_CAP, random_state=42)
    return s


def lda_topics(
    df: pd.DataFrame, column: str, n_topics: int = 5, top_words: int = 8
) -> list[dict]:
    """Run LDA topic modelling and return top words per topic."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        corpus = _lda_corpus(df, column).tolist()
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


def document_topic_assignment(
    df: pd.DataFrame, column: str, n_topics: int = 5
) -> list[dict]:
    """Assign each document to its dominant topic. `index` is the row's index in the dataset."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        sampled = _lda_corpus(df, column)
        corpus = sampled.tolist()
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
                "index": int(sampled.index[i]),
                "dominant_topic": int(doc_topic[i].argmax()),
                "topic_score": round(float(doc_topic[i].max()), 4),
                "text_preview": corpus[i][:80],
            }
            for i in range(len(corpus))
        ]

    except Exception as e:
        return [{"error": str(e)}]


# Row cap for each analysis that works on a sample (see EMBEDDING_SAMPLE_CAP and LDA_SAMPLE_CAP).
SAMPLED_ANALYSES = {
    "topics": LDA_SAMPLE_CAP,
    "document_topics": LDA_SAMPLE_CAP,
    "clusters": EMBEDDING_SAMPLE_CAP,
    "cluster_keywords": EMBEDDING_SAMPLE_CAP,
    "cluster_sizes": EMBEDDING_SAMPLE_CAP,
    "similarity_search": EMBEDDING_SAMPLE_CAP,
    "similar_pairs": 200,
}


def sampling_warning(analysis_type: str, n_rows: int) -> str | None:
    """A sentence for the UI when an analysis used a sample instead of every row, else None."""
    cap = SAMPLED_ANALYSES.get(analysis_type)
    if cap and n_rows > cap:
        return f"This analysis used a random sample of {cap:,} of your {n_rows:,} rows to keep it fast."
    return None

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
    


# 18  Cluster Keyword Summary

def cluster_keywords(
    df: pd.DataFrame, column: str, n_clusters: int = 5, top_words: int = 5
) -> list[dict]:
    """Return top keywords per cluster using TF-IDF per cluster."""
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found."}]

    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        s = df[column].fillna("").astype(str)
        sampled = s.sample(
            min(EMBEDDING_SAMPLE_CAP, len(s)), random_state=42
        ).reset_index(drop=True)

        embedder = _get_embedder()
        embeddings = embedder.encode(sampled.tolist(), show_progress_bar=False)

        n_clusters = min(n_clusters, len(sampled))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        result = []
        vectorizer = TfidfVectorizer(stop_words="english", max_features=200)

        for cluster_id in range(n_clusters):
            cluster_docs = sampled[labels == cluster_id].tolist()
            if len(cluster_docs) < 2:
                result.append({
                    "cluster_id": cluster_id,
                    "size": len(cluster_docs),
                    "keywords": [],
                })
                continue

            tfidf = vectorizer.fit_transform(cluster_docs)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.asarray(tfidf.mean(axis=0)).flatten()
            top_idx = mean_scores.argsort()[::-1][:top_words]
            keywords = [feature_names[i] for i in top_idx]

            result.append({
                "cluster_id": cluster_id,
                "size": len(cluster_docs),
                "keywords": keywords,
                "representative_text": cluster_docs[0][:100] if cluster_docs else "",
            })

        return result

    except Exception as e:
        return [{"error": str(e)}]


# 19  Cluster Size Distribution

def cluster_size_distribution(
    df: pd.DataFrame, column: str, n_clusters: int = 5
) -> list[dict]:
    """Return cluster sizes as a distribution."""
    clusters = cluster_texts(df, column, n_clusters=n_clusters)
    if clusters and "error" in clusters[0]:
        return clusters

    total = sum(c["size"] for c in clusters)
    return [
        {
            "cluster_id": c["cluster_id"],
            "size": c["size"],
            "pct": round(c["size"] / total * 100, 2) if total > 0 else 0,
        }
        for c in clusters
    ]


# 20  Rule-Based NLP Insights

def rule_based_nlp_insights(
    sentiment: dict,
    keywords: list[dict],
    topics: list[dict],
    quality: dict,
) -> list[str]:
    """Generate rule-based NLP insights from analysis results."""
    insights = []

    # Sentiment insight
    if sentiment and "overall_sentiment" in sentiment:
        overall = sentiment["overall_sentiment"]
        pos_pct = sentiment.get("positive_pct", 0)
        neg_pct = sentiment.get("negative_pct", 0)
        if pos_pct > 60:
            insights.append(
                f"The text is predominantly positive ({pos_pct:.1f}% positive sentiment)."
            )
        elif neg_pct > 40:
            insights.append(
                f"The text shows significant negative sentiment ({neg_pct:.1f}% negative)."
            )
        else:
            insights.append(
                f"The text has mixed sentiment — {pos_pct:.1f}% positive, "
                f"{neg_pct:.1f}% negative."
            )

    # Quality insight
    if quality and "quality_score" in quality:
        score = quality["quality_score"]
        if score < 60:
            insights.append(
                f"Text quality is {quality['quality_label']} (score: {score}/100). "
                f"{quality.get('empty_count', 0)} empty and "
                f"{quality.get('duplicate_count', 0)} duplicate entries detected."
            )

    # Top keywords
    if keywords and len(keywords) >= 3 and "keyword" in keywords[0]:
        top_words = [k["keyword"] for k in keywords[:5]]
        insights.append(
            f"Top keywords: {', '.join(top_words)}."
        )

    # Topic insight
    if topics and len(topics) >= 2 and "keywords" in topics[0]:
        top_topic = topics[0]
        insights.append(
            f"The dominant topic covers: {', '.join(top_topic['keywords'][:4])} "
            f"({top_topic['proportion']:.1f}% of documents)."
        )

    return insights if insights else ["NLP analysis complete. Review individual results for details."]


# 21 Gemini NLP Narrative

async def generate_nlp_insights(
    column: str,
    sentiment: dict,
    keywords: list[dict],
    topics: list[dict],
    quality: dict,
) -> dict:
    """Generate NLP insights — Gemini narrative or rule-based fallback."""
    rule_insights = rule_based_nlp_insights(
        sentiment, keywords, topics, quality
    )

    summary = {
        "column": column,
        "sentiment": {
            "overall": sentiment.get("overall_sentiment"),
            "positive_pct": sentiment.get("positive_pct"),
            "negative_pct": sentiment.get("negative_pct"),
            "neutral_pct": sentiment.get("neutral_pct"),
        },
        "top_keywords": [k["keyword"] for k in keywords[:10] if "keyword" in k],
        "topics": [
            {"keywords": t["keywords"][:5], "proportion": t["proportion"]}
            for t in topics[:3]
            if "keywords" in t
        ],
        "quality": {
            "score": quality.get("quality_score"),
            "label": quality.get("quality_label"),
            "empty_pct": quality.get("empty_pct"),
            "duplicate_pct": quality.get("duplicate_pct"),
        },
    }

    gemini_insights = await _get_gemini_nlp_insights(summary)

    return {
        "insights": gemini_insights if gemini_insights else rule_insights,
        "source": "gemini" if gemini_insights else "rule_based",
    }


# 22 Text Statistics

def text_statistics(df: pd.DataFrame, column: str) -> dict:
    """Compute avg length, word count, vocab size, sentence count."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].dropna().astype(str)

    stopwords = _get_stopwords()
    all_tokens = []
    for text in s:
        all_tokens.extend(_clean_text(text, stopwords))

    avg_length = float(s.str.len().mean())
    avg_words = float(s.str.split().str.len().mean())
    vocab_size = len(set(all_tokens))

    # Approximate sentence count
    sentence_count = int(
        s.str.count(r"[.!?]+").sum()
    )

    return {
        "column": column,
        "total_documents": len(s),
        "avg_char_length": round(avg_length, 2),
        "avg_word_count": round(avg_words, 2),
        "vocabulary_size": vocab_size,
        "total_sentences": sentence_count,
        "avg_sentences_per_doc": round(
            sentence_count / len(s), 2
        ) if len(s) > 0 else 0,
    }


# 23 Readability Estimate (Flesch-Kincaid approximation)

def readability_estimate(df: pd.DataFrame, column: str) -> dict:
    """
    Approximate Flesch Reading Ease score.
    Score: 90-100 very easy, 60-70 standard, 0-30 very difficult.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].dropna().astype(str)
    scores = []

    for text in s.sample(min(200, len(s)), random_state=42):
        sentences = max(1, len(re.findall(r"[.!?]+", text)))
        words = text.split()
        if not words:
            continue
        syllables = sum(
            max(1, len(re.findall(r"[aeiouAEIOU]", w))) for w in words
        )
        fre = (
            206.835
            - 1.015 * (len(words) / sentences)
            - 84.6 * (syllables / len(words))
        )
        scores.append(fre)

    if not scores:
        return {"column": column, "avg_flesch_score": 0, "label": "unknown"}

    avg_score = float(np.mean(scores))
    if avg_score >= 80:
        label = "very easy"
    elif avg_score >= 60:
        label = "standard"
    elif avg_score >= 40:
        label = "difficult"
    else:
        label = "very difficult"

    return {
        "column": column,
        "avg_flesch_score": round(avg_score, 2),
        "label": label,
        "sample_size": len(scores),
    }


# 24  Named Entity Pattern Detection

def named_entity_patterns(df: pd.DataFrame, column: str) -> dict:
    """
    Detect common named entity patterns using regex:
    emails, URLs, phone numbers, dates, monetary amounts.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].fillna("").astype(str)

    patterns = {
        "emails": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "urls": r"https?://[^\s]+",
        "phone_numbers": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "monetary": r"\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP)",
        "hashtags": r"#\w+",
        "mentions": r"@\w+",
    }

    results = {}
    for name, pattern in patterns.items():
        matches = s.str.findall(pattern)
        total_matches = matches.str.len().sum()
        docs_with_matches = (matches.str.len() > 0).sum()
        results[name] = {
            "total_occurrences": int(total_matches),
            "documents_with_pattern": int(docs_with_matches),
            "sample": list(matches[matches.str.len() > 0].iloc[0])[:3]
            if docs_with_matches > 0 else [],
        }

    return {"column": column, "patterns": results}


# 25 Duplicate Text Detection

def duplicate_text_detection(df: pd.DataFrame, column: str) -> dict:
    """Find exact and near-duplicate texts."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = df[column].fillna("").astype(str).str.strip().str.lower()

    exact_dups = s[s.duplicated(keep=False)]
    exact_count = int(s.duplicated().sum())

    dup_groups = (
        exact_dups.value_counts().head(5).to_dict()
        if exact_count > 0
        else {}
    )

    return {
        "column": column,
        "total": len(s),
        "exact_duplicate_count": exact_count,
        "exact_duplicate_pct": round(exact_count / len(s) * 100, 2),
        "top_duplicates": {str(k): int(v) for k, v in dup_groups.items()},
    }


# 26 Empty/Noise Detection

def empty_noise_detection(df: pd.DataFrame, column: str) -> dict:
    """Identify empty, whitespace-only, and noise entries."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    total = len(df)
    null_count = int(df[column].isna().sum())
    s = df[column].fillna("").astype(str)

    whitespace_only = int((s.str.strip() == "").sum()) - null_count
    whitespace_only = max(0, whitespace_only)

    single_char = int((s.str.strip().str.len() == 1).sum())
    numeric_only = int(s.str.strip().str.match(r"^\d+$").sum())

    return {
        "column": column,
        "total": total,
        "null_count": null_count,
        "whitespace_only_count": whitespace_only,
        "single_char_count": single_char,
        "numeric_only_count": numeric_only,
        "usable_count": total - null_count - whitespace_only,
        "usable_pct": round(
            (total - null_count - whitespace_only) / total * 100, 2
        ) if total > 0 else 0,
    }


# 27 NLP Summary for Report

def nlp_summary(
    column: str,
    sentiment: dict,
    keywords: list[dict],
    topics: list[dict],
    quality: dict,
    text_stats: dict,
) -> dict:
    """Generate an NLP summary for the report stage."""
    return {
        "column": column,
        "quality": {
            "score": quality.get("quality_score"),
            "label": quality.get("quality_label"),
            "total_documents": text_stats.get("total_documents"),
            "avg_word_count": text_stats.get("avg_word_count"),
            "vocabulary_size": text_stats.get("vocabulary_size"),
        },
        "sentiment": {
            "overall": sentiment.get("overall_sentiment"),
            "positive_pct": sentiment.get("positive_pct"),
            "negative_pct": sentiment.get("negative_pct"),
            "neutral_pct": sentiment.get("neutral_pct"),
            "avg_compound": sentiment.get("avg_compound"),
        },
        "top_keywords": [
            k["keyword"] for k in keywords[:10] if "keyword" in k
        ],
        "topics": [
            {
                "id": t["topic_id"],
                "keywords": t["keywords"][:5],
                "proportion": t["proportion"],
            }
            for t in topics[:5]
            if "topic_id" in t
        ],
    }












