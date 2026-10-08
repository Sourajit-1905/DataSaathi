"""
Stage 7 Tests — NLP Service.
Covers text detection, quality, sentiment, keywords, topics,
similarity, clustering, insights, and router endpoints.

Tests that need optional heavy dependencies (vaderSentiment, langdetect,
sentence-transformers) are skipped automatically if the dependency or the
embedding model is unavailable, so the suite never fails on a missing download.
"""

import io
import pytest
import pandas as pd
import numpy as np


# Helpers & Fixtures

POSITIVE = [
    "The product quality is excellent and I love the design",
    "Great value for the price and the delivery was fast",
    "Amazing service and the staff were very helpful",
    "Wonderful experience and the product works perfectly",
]
NEGATIVE = [
    "Terrible quality and the product broke after one week",
    "Very disappointed with the slow delivery and poor service",
    "Awful experience and the staff were rude and unhelpful",
    "Horrible design and the product does not work at all",
]
NEUTRAL = [
    "The package arrived on Tuesday in a brown box",
    "The product is available in three different sizes",
    "The store opens at nine and closes at five",
    "The manual explains how to install the product",
]


def _make_reviews(n: int = 40) -> pd.DataFrame:
    pool = POSITIVE + NEGATIVE + NEUTRAL
    texts = [f"{pool[i % len(pool)]} number {i}" for i in range(n)]
    return pd.DataFrame({
        "Review": texts,
        "Category": ["A", "B", "C", "D"] * (n // 4),
        "Rating": [5, 1, 3, 4] * (n // 4),
    })


@pytest.fixture
def review_df():
    return _make_reviews(40)


@pytest.fixture
def positive_df():
    texts = [f"{POSITIVE[i % 4]} number {i}" for i in range(20)]
    return pd.DataFrame({"Review": texts})


@pytest.fixture
def negative_df():
    texts = [f"{NEGATIVE[i % 4]} number {i}" for i in range(20)]
    return pd.DataFrame({"Review": texts})


@pytest.fixture(scope="session")
def vader_available():
    pytest.importorskip("vaderSentiment")
    return True


@pytest.fixture(scope="session")
def embedder_available():
    """Skip embedding tests if sentence-transformers or the model is unavailable."""
    pytest.importorskip("sentence_transformers")
    try:
        from services.nlp_service import _get_embedder
        _get_embedder()
    except Exception as e:
        pytest.skip(f"Embedding model unavailable: {e}")
    return True


def _upload_df(client, df) -> str:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    resp = client.post(
        "/api/dataset/upload",
        files={"file": ("data.csv", buf.getvalue(), "text/csv")},
    )
    return resp.json()["session_id"]


#  1   Text Column Detection

def test_n1_detects_text_column(review_df):
    from services.nlp_service import detect_text_columns
    result = detect_text_columns(review_df)
    by_col = {r["column"]: r for r in result}
    assert by_col["Review"]["is_text_column"] is True


def test_n1_short_categorical_not_text(review_df):
    from services.nlp_service import detect_text_columns
    result = detect_text_columns(review_df)
    by_col = {r["column"]: r for r in result}
    assert by_col["Category"]["is_text_column"] is False


def test_n1_numeric_columns_excluded(review_df):
    from services.nlp_service import detect_text_columns
    cols = [r["column"] for r in detect_text_columns(review_df)]
    assert "Rating" not in cols


# N2   Text Quality

def test_n2_quality_keys(review_df):
    from services.nlp_service import text_quality_assessment
    result = text_quality_assessment(review_df, "Review")
    for key in ["column", "total", "empty_count", "duplicate_count",
                "very_short_count", "noise_count", "quality_score",
                "quality_label"]:
        assert key in result


def test_n2_clean_text_scores_high(review_df):
    from services.nlp_service import text_quality_assessment
    result = text_quality_assessment(review_df, "Review")
    assert result["quality_score"] >= 80
    assert result["quality_label"] == "excellent"


def test_n2_empty_values_counted():
    from services.nlp_service import text_quality_assessment
    df = pd.DataFrame({"T": ["good product here", None, "   ", "another good one"]})
    result = text_quality_assessment(df, "T")
    assert result["empty_count"] == 2
    assert 0 <= result["quality_score"] <= 100


def test_n2_invalid_column():
    from services.nlp_service import text_quality_assessment
    result = text_quality_assessment(pd.DataFrame({"A": ["x"]}), "Nope")
    assert "error" in result


# 3  Language Detection

def test_n3_detects_english(review_df):
    pytest.importorskip("langdetect")
    from services.nlp_service import detect_language
    result = detect_language(review_df, "Review")
    assert result["dominant_language"] == "en"


# N4-N6 Sentiment

def test_n4_vader_scores_per_row(vader_available, review_df):
    from services.nlp_service import vader_sentiment_scores
    scores = vader_sentiment_scores(review_df, "Review")
    assert len(scores) == len(review_df)
    for key in ["compound", "positive", "negative", "neutral", "label"]:
        assert key in scores[0]


def test_n4_labels_valid(vader_available, review_df):
    from services.nlp_service import vader_sentiment_scores
    labels = {s["label"] for s in vader_sentiment_scores(review_df, "Review")}
    assert labels.issubset({"positive", "negative", "neutral"})


def test_n5_positive_text_is_positive(vader_available, positive_df):
    from services.nlp_service import sentiment_aggregation
    result = sentiment_aggregation(positive_df, "Review")
    assert result["overall_sentiment"] == "positive"
    assert result["positive_pct"] > result["negative_pct"]


def test_n5_negative_text_is_negative(vader_available, negative_df):
    from services.nlp_service import sentiment_aggregation
    result = sentiment_aggregation(negative_df, "Review")
    assert result["overall_sentiment"] == "negative"
    assert result["negative_pct"] > result["positive_pct"]


def test_n5_percentages_sum_to_100(vader_available, review_df):
    from services.nlp_service import sentiment_aggregation
    r = sentiment_aggregation(review_df, "Review")
    total = r["positive_pct"] + r["negative_pct"] + r["neutral_pct"]
    assert abs(total - 100.0) < 0.1
    assert r["positive_count"] + r["negative_count"] + r["neutral_count"] == r["total"]


def test_n6_distribution_counts_sum(vader_available, review_df):
    from services.nlp_service import sentiment_distribution
    result = sentiment_distribution(review_df, "Review")
    assert sum(result["histogram"]["counts"]) == len(review_df)
    assert len(result["histogram"]["counts"]) == 10
    assert len(result["histogram"]["bin_edges"]) == 11


# N7-N10  Keywords, Frequency, Word Cloud, N-grams

def test_n7_tfidf_keywords_structure(review_df):
    from services.nlp_service import tfidf_keywords
    result = tfidf_keywords(review_df, "Review", top_n=10)
    assert len(result) == 10
    assert result[0]["rank"] == 1
    assert "keyword" in result[0] and "score" in result[0]


def test_n7_tfidf_scores_descending(review_df):
    from services.nlp_service import tfidf_keywords
    scores = [k["score"] for k in tfidf_keywords(review_df, "Review", top_n=10)]
    assert scores == sorted(scores, reverse=True)


def test_n8_word_frequency_descending(review_df):
    from services.nlp_service import word_frequency
    result = word_frequency(review_df, "Review", top_n=10)
    counts = [w["count"] for w in result]
    assert counts == sorted(counts, reverse=True)
    assert result[0]["rank"] == 1


def test_n8_stopwords_or_short_tokens_removed(review_df):
    from services.nlp_service import word_frequency
    words = [w["word"] for w in word_frequency(review_df, "Review", top_n=50)]
    # Tokens of length <= 2 are always removed by _clean_text
    assert all(len(w) > 2 for w in words)


def test_n9_word_cloud_weights(review_df):
    from services.nlp_service import word_cloud_data
    result = word_cloud_data(review_df, "Review", top_n=20)
    assert max(w["weight"] for w in result) == 1.0
    assert all(0 < w["weight"] <= 1.0 for w in result)


def test_n10_bigrams_have_two_words(review_df):
    from services.nlp_service import extract_ngrams
    result = extract_ngrams(review_df, "Review", n=2, top_n=10)
    assert len(result) > 0
    assert all(len(r["ngram"].split()) == 2 for r in result)


def test_n10_trigrams_have_three_words(review_df):
    from services.nlp_service import extract_ngrams
    result = extract_ngrams(review_df, "Review", n=3, top_n=10)
    assert all(len(r["ngram"].split()) == 3 for r in result)


# N11-N13  Topics

def test_n11_lda_returns_requested_topics(review_df):
    from services.nlp_service import lda_topics
    topics = lda_topics(review_df, "Review", n_topics=3)
    assert len(topics) == 3
    for t in topics:
        assert "topic_id" in t and "keywords" in t and "proportion" in t
        assert 0 < len(t["keywords"]) <= 8


def test_n11_topic_proportions_sum_to_100(review_df):
    from services.nlp_service import lda_topics
    topics = lda_topics(review_df, "Review", n_topics=3)
    assert abs(sum(t["proportion"] for t in topics) - 100.0) < 0.1


def test_n12_topic_keyword_map(review_df):
    from services.nlp_service import topic_keyword_map
    result = topic_keyword_map(review_df, "Review", n_topics=3)
    assert set(result.keys()) == {"0", "1", "2"}


def test_n13_document_topic_assignment(review_df):
    from services.nlp_service import document_topic_assignment
    result = document_topic_assignment(review_df, "Review", n_topics=3)
    assert len(result) == len(review_df)
    assert all(0 <= r["dominant_topic"] < 3 for r in result)


# N14-N19   Embeddings, Similarity, Clustering (skipped if no model)

def test_n14_embeddings_shape(embedder_available, review_df):
    from services.nlp_service import get_embeddings
    result = get_embeddings(review_df, "Review")
    assert result["n_sampled"] == len(review_df)
    assert len(result["embeddings"]) == len(review_df)
    assert result["embedding_dim"] > 0


def test_n15_similarity_search_returns_top_n(embedder_available, review_df):
    from services.nlp_service import semantic_similarity_search
    result = semantic_similarity_search(
        review_df, "Review", "fast delivery and great price", top_n=3
    )
    assert len(result) == 3
    assert result[0]["rank"] == 1
    sims = [r["similarity"] for r in result]
    assert sims == sorted(sims, reverse=True)


def test_n16_most_similar_pairs(embedder_available, review_df):
    from services.nlp_service import most_similar_pairs
    result = most_similar_pairs(review_df, "Review", top_n=3)
    assert len(result) == 3
    assert all("text_a" in r and "text_b" in r for r in result)


def test_n17_clusters_cover_all_rows(embedder_available, review_df):
    from services.nlp_service import cluster_texts
    result = cluster_texts(review_df, "Review", n_clusters=3)
    assert len(result) == 3
    assert sum(c["size"] for c in result) == len(review_df)


def test_n18_cluster_keywords(embedder_available, review_df):
    from services.nlp_service import cluster_keywords
    result = cluster_keywords(review_df, "Review", n_clusters=3)
    assert len(result) == 3
    assert all("keywords" in c for c in result)


def test_n19_cluster_sizes_sum_to_100(embedder_available, review_df):
    from services.nlp_service import cluster_size_distribution
    result = cluster_size_distribution(review_df, "Review", n_clusters=3)
    assert abs(sum(c["pct"] for c in result) - 100.0) < 0.1


# N20-N27   Insights, Stats, Readability, Patterns, Duplicates, Noise

def test_n20_rule_based_insights_returns_list():
    from services.nlp_service import rule_based_nlp_insights
    sentiment = {"overall_sentiment": "positive", "positive_pct": 70.0,
                 "negative_pct": 10.0, "neutral_pct": 20.0}
    keywords = [{"keyword": "quality"}, {"keyword": "price"}, {"keyword": "service"}]
    topics = [{"topic_id": 0, "keywords": ["quality", "design", "price", "fast"],
               "proportion": 40.0},
              {"topic_id": 1, "keywords": ["service"], "proportion": 30.0}]
    quality = {"quality_score": 50, "quality_label": "fair",
               "empty_count": 3, "duplicate_count": 2}
    insights = rule_based_nlp_insights(sentiment, keywords, topics, quality)
    assert isinstance(insights, list)
    assert len(insights) >= 3
    assert all(isinstance(s, str) for s in insights)


def test_n20_rule_based_insights_empty_inputs_fallback():
    from services.nlp_service import rule_based_nlp_insights
    insights = rule_based_nlp_insights({}, [], [], {})
    assert len(insights) == 1


@pytest.mark.asyncio
async def test_n21_generate_insights_has_source(vader_available, review_df):
    from services import nlp_service as n
    sentiment = n.sentiment_aggregation(review_df, "Review")
    keywords = n.tfidf_keywords(review_df, "Review", 10)
    topics = n.lda_topics(review_df, "Review", 3)
    quality = n.text_quality_assessment(review_df, "Review")
    result = await n.generate_nlp_insights("Review", sentiment, keywords, topics, quality)
    assert result["source"] in ["gemini", "rule_based"]
    assert len(result["insights"]) >= 1


def test_n22_text_statistics(review_df):
    from services.nlp_service import text_statistics
    result = text_statistics(review_df, "Review")
    assert result["total_documents"] == len(review_df)
    assert result["avg_word_count"] > 3
    assert result["vocabulary_size"] > 0


def test_n23_readability_label_valid(review_df):
    from services.nlp_service import readability_estimate
    result = readability_estimate(review_df, "Review")
    assert result["label"] in ["very easy", "standard", "difficult", "very difficult"]
    assert "avg_flesch_score" in result


def test_n24_entity_patterns_detected():
    from services.nlp_service import named_entity_patterns
    df = pd.DataFrame({"T": [
        "Contact me at john@example.com for details",
        "Visit https://example.com/page today",
        "Loving this product #amazing #deal",
        "Nothing special in this sentence",
    ]})
    result = named_entity_patterns(df, "T")["patterns"]
    assert result["emails"]["documents_with_pattern"] == 1
    assert "john@example.com" in result["emails"]["sample"]
    assert result["urls"]["documents_with_pattern"] == 1
    assert result["hashtags"]["total_occurrences"] == 2


def test_n24_no_matches_returns_empty_sample(review_df):
    from services.nlp_service import named_entity_patterns
    result = named_entity_patterns(review_df, "Review")["patterns"]
    assert result["emails"]["sample"] == []
    assert result["emails"]["total_occurrences"] == 0


def test_n25_duplicate_detection():
    from services.nlp_service import duplicate_text_detection
    df = pd.DataFrame({"T": ["a b c", "a b c", "A B C ", "d e f"]})
    result = duplicate_text_detection(df, "T")
    assert result["exact_duplicate_count"] == 2
    assert result["total"] == 4


def test_n26_empty_noise_detection():
    from services.nlp_service import empty_noise_detection
    df = pd.DataFrame({"T": ["real text here", None, "   ", "7", "12345"]})
    result = empty_noise_detection(df, "T")
    assert result["null_count"] == 1
    assert result["whitespace_only_count"] == 1
    assert result["numeric_only_count"] == 2
    assert result["usable_count"] == 3


def test_n27_nlp_summary_keys():
    from services.nlp_service import nlp_summary
    result = nlp_summary(
        "Review",
        {"overall_sentiment": "positive", "positive_pct": 60.0,
         "negative_pct": 20.0, "neutral_pct": 20.0, "avg_compound": 0.4},
        [{"keyword": "quality"}],
        [{"topic_id": 0, "keywords": ["a", "b"], "proportion": 50.0}],
        {"quality_score": 90, "quality_label": "excellent"},
        {"total_documents": 40, "avg_word_count": 9.0, "vocabulary_size": 60},
    )
    for key in ["column", "quality", "sentiment", "top_keywords", "topics"]:
        assert key in result


# Integration Tests  router endpoints

def test_detect_columns_endpoint(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.get(f"/api/nlp/{session_id}/detect-columns")
    assert resp.status_code == 200
    data = resp.json()
    assert "Review" in data["text_columns"]
    assert "Category" not in data["text_columns"]


def test_run_quality_endpoint(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "quality", "column": "Review"},
    )
    assert resp.status_code == 200
    assert "quality_score" in resp.json()["result"]


def test_run_keywords_endpoint(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "keywords", "column": "Review",
              "options": {"top_n": 5}},
    )
    assert resp.status_code == 200
    assert len(resp.json()["result"]) == 5


def test_run_sentiment_endpoint(client, vader_available):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "sentiment", "column": "Review"},
    )
    assert resp.status_code == 200
    assert "overall_sentiment" in resp.json()["result"]


def test_run_topics_endpoint(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "topics", "column": "Review",
              "options": {"n_topics": 3}},
    )
    assert resp.status_code == 200
    assert len(resp.json()["result"]) == 3


def test_run_entities_endpoint(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "entities", "column": "Review"},
    )
    assert resp.status_code == 200
    assert "patterns" in resp.json()["result"]


def test_run_logs_to_history(client):
    session_id = _upload_df(client, _make_reviews(40))
    client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "quality", "column": "Review"},
    )
    client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "text_stats", "column": "Review"},
    )
    resp = client.get(f"/api/nlp/{session_id}/history")
    assert resp.status_code == 200
    assert resp.json()["total_analyses"] == 2


def test_run_invalid_analysis_type_returns_400(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "nonexistent", "column": "Review"},
    )
    assert resp.status_code == 400


def test_run_invalid_column_returns_400(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/run",
        json={"analysis_type": "quality", "column": "NoSuchColumn"},
    )
    assert resp.status_code == 400


def test_run_invalid_session_returns_404(client):
    resp = client.post(
        "/api/nlp/00000000-0000-4000-8000-000000000000/run",
        json={"analysis_type": "quality", "column": "Review"},
    )
    assert resp.status_code == 404


def test_insights_endpoint(client, vader_available):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.get(f"/api/nlp/{session_id}/insights", params={"column": "Review"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["source"] in ["gemini", "rule_based"]


def test_question_endpoint_similarity(client, embedder_available):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/question",
        json={"column": "Review", "query": "fast delivery", "top_n": 3},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 3


def test_question_short_query_returns_400(client):
    session_id = _upload_df(client, _make_reviews(40))
    resp = client.post(
        f"/api/nlp/{session_id}/question",
        json={"column": "Review", "query": "a"},
    )
    assert resp.status_code == 400