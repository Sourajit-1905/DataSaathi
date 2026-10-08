"""
NLP Router
----------
Endpoints:
  GET  /api/nlp/{session_id}/detect-columns   — detect free-text columns
  POST /api/nlp/{session_id}/run              — run an NLP analysis
  POST /api/nlp/{session_id}/question         — similarity search for a query
  GET  /api/nlp/{session_id}/insights         — NLP insights for a text column
  GET  /api/nlp/{session_id}/history          — NLP analysis history
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from core.security import validate_session_id
from core.cache import append_nlp_history, get_nlp_history
from services import session_service
from services import nlp_service

router = APIRouter(prefix="/api/nlp", tags=["nlp"])
logger = logging.getLogger(__name__)


#  Request Bodies 

class RunRequest(BaseModel):
    analysis_type: str
    column: str
    options: Optional[dict[str, Any]] = {}


class QuestionRequest(BaseModel):
    column: str
    query: str
    top_n: Optional[int] = 5


#  Helpers 

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


def _check_column(df, column: str):
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found.")


VALID_ANALYSES = {
    "quality", "language", "sentiment", "sentiment_scores",
    "sentiment_distribution", "keywords", "word_frequency",
    "word_cloud", "ngrams", "topics", "document_topics",
    "similar_pairs", "clusters", "cluster_keywords",
    "cluster_sizes", "text_stats", "readability",
    "entities", "duplicates", "noise",
}


#  GET /api/nlp/{session_id}/detect-columns 

@router.get("/{session_id}/detect-columns")
def detect_columns(session_id: str):
    """Detect which columns contain free-form text."""
    _check_session(session_id)
    df = session_service.get_working_df(session_id)
    columns = nlp_service.detect_text_columns(df)
    return {
        "session_id": session_id,
        "columns": columns,
        "text_columns": [c["column"] for c in columns if c["is_text_column"]],
    }


#  POST /api/nlp/{session_id}/run 

@router.post("/{session_id}/run")
def run_analysis(session_id: str, body: RunRequest):
    """
    Run a single NLP analysis on a text column.
    Heavy analyses (embeddings, clustering) sample to 2000 rows.
    """
    _check_session(session_id)

    if body.analysis_type not in VALID_ANALYSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown analysis_type '{body.analysis_type}'. "
                   f"Valid types: {sorted(VALID_ANALYSES)}",
        )

    df = session_service.get_working_df(session_id)
    _check_column(df, body.column)

    opts = body.options or {}
    col = body.column
    t = body.analysis_type

    try:
        if t == "quality":
            result = nlp_service.text_quality_assessment(df, col)
        elif t == "language":
            result = nlp_service.detect_language(df, col)
        elif t == "sentiment":
            result = nlp_service.sentiment_aggregation(df, col)
        elif t == "sentiment_scores":
            result = nlp_service.vader_sentiment_scores(df, col)[:200]
        elif t == "sentiment_distribution":
            result = nlp_service.sentiment_distribution(df, col)
        elif t == "keywords":
            result = nlp_service.tfidf_keywords(df, col, opts.get("top_n", 20))
        elif t == "word_frequency":
            result = nlp_service.word_frequency(df, col, opts.get("top_n", 30))
        elif t == "word_cloud":
            result = nlp_service.word_cloud_data(df, col, opts.get("top_n", 50))
        elif t == "ngrams":
            result = nlp_service.extract_ngrams(
                df, col, opts.get("n", 2), opts.get("top_n", 20)
            )
        elif t == "topics":
            result = nlp_service.lda_topics(
                df, col, opts.get("n_topics", 5), opts.get("top_words", 8)
            )
        elif t == "document_topics":
            result = nlp_service.document_topic_assignment(
                df, col, opts.get("n_topics", 5)
            )[:200]
        elif t == "similar_pairs":
            result = nlp_service.most_similar_pairs(df, col, opts.get("top_n", 5))
        elif t == "clusters":
            result = nlp_service.cluster_texts(df, col, opts.get("n_clusters", 5))
        elif t == "cluster_keywords":
            result = nlp_service.cluster_keywords(
                df, col, opts.get("n_clusters", 5), opts.get("top_words", 5)
            )
        elif t == "cluster_sizes":
            result = nlp_service.cluster_size_distribution(
                df, col, opts.get("n_clusters", 5)
            )
        elif t == "text_stats":
            result = nlp_service.text_statistics(df, col)
        elif t == "readability":
            result = nlp_service.readability_estimate(df, col)
        elif t == "entities":
            result = nlp_service.named_entity_patterns(df, col)
        elif t == "duplicates":
            result = nlp_service.duplicate_text_detection(df, col)
        elif t == "noise":
            result = nlp_service.empty_noise_detection(df, col)
        else:
            raise HTTPException(status_code=400, detail=f"Unhandled analysis_type: {t}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"NLP analysis failed: {e}")
        raise HTTPException(status_code=500, detail="NLP analysis failed.")

    # Service-level errors
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    if isinstance(result, list) and result and isinstance(result[0], dict) and "error" in result[0]:
        raise HTTPException(status_code=400, detail=result[0]["error"])

    append_nlp_history(session_id, {
        "analysis_type": t,
        "column": col,
        "options": opts,
    })

    logger.info(f"NLP run | session_id={session_id} | type={t} | col={col}")

    return {
        "session_id": session_id,
        "analysis_type": t,
        "column": col,
        "result": result,
    }


#  POST /api/nlp/{session_id}/question 

@router.post("/{session_id}/question")
def similarity_question(session_id: str, body: QuestionRequest):
    """Find the records most semantically similar to a query."""
    _check_session(session_id)

    if not body.query or len(body.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query is too short.")

    df = session_service.get_working_df(session_id)
    _check_column(df, body.column)

    results = nlp_service.semantic_similarity_search(
        df, body.column, body.query, body.top_n or 5
    )

    if results and "error" in results[0]:
        raise HTTPException(status_code=400, detail=results[0]["error"])

    append_nlp_history(session_id, {
        "analysis_type": "similarity_search",
        "column": body.column,
        "query": body.query,
    })

    return {
        "session_id": session_id,
        "query": body.query,
        "column": body.column,
        "results": results,
    }


#  GET /api/nlp/{session_id}/insights 

@router.get("/{session_id}/insights")
async def get_insights(session_id: str, column: str):
    """Generate NLP insights (Gemini narrative or rule-based fallback)."""
    _check_session(session_id)

    df = session_service.get_working_df(session_id)
    _check_column(df, column)

    sentiment = nlp_service.sentiment_aggregation(df, column)
    keywords = nlp_service.tfidf_keywords(df, column, 10)
    topics = nlp_service.lda_topics(df, column, 3)
    quality = nlp_service.text_quality_assessment(df, column)

    # Drop error placeholders so downstream helpers get clean inputs
    if keywords and "error" in keywords[0]:
        keywords = []
    if topics and "error" in topics[0]:
        topics = []

    try:
        result = await nlp_service.generate_nlp_insights(
            column, sentiment, keywords, topics, quality
        )
    except Exception as e:
        logger.exception(f"NLP insights failed: {e}")
        raise HTTPException(status_code=500, detail="Insight generation failed.")

    return {
        "session_id": session_id,
        "column": column,
        "insights": result["insights"],
        "source": result["source"],
        "count": len(result["insights"]),
    }


#  GET /api/nlp/{session_id}/history 

@router.get("/{session_id}/history")
def get_history(session_id: str):
    """Return the NLP analysis history for this session."""
    _check_session(session_id)
    history = get_nlp_history(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "total_analyses": len(history),
    }