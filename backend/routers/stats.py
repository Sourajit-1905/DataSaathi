"""
Stats Router
------------
Endpoints:
  POST /api/stats/{session_id}/recommend   — recommend test based on question + columns
  POST /api/stats/{session_id}/run         — run a specific statistical test
  GET  /api/stats/{session_id}/history     — return stats test history
  POST /api/stats/{session_id}/question    — answer a stats question in plain English
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from core.security import validate_session_id
from core.cache import append_stats_history, get_stats_history
from core.exceptions import ColumnNotFoundError
from services import session_service
from services import stats_service

router = APIRouter(prefix="/api/stats", tags=["stats"])
logger = logging.getLogger(__name__)


#    Request Bodies  

class RecommendRequest(BaseModel):
    question: str
    columns: list[str]


class RunTestRequest(BaseModel):
    test_name: str
    options: dict[str, Any] = {}


class StatsQuestionRequest(BaseModel):
    question: str
    columns: Optional[list[str]] = []


#    Helpers  

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


VALID_TESTS = {
    "one_sample_t", "independent_t", "paired_t",
    "one_way_anova", "two_way_anova", "repeated_measures_anova",
    "pearson", "spearman", "kendall",
    "linear_regression", "multiple_regression",
    "mann_whitney", "wilcoxon", "kruskal_wallis",
    "chi_square_independence", "chi_square_goodness",
    "fisher_exact", "shapiro_wilk", "ks_test",
    "levene", "bartlett",
}


#    POST /api/stats/{session_id}/recommend  

@router.post("/{session_id}/recommend")
def recommend(session_id: str, body: RecommendRequest):
    """
    Recommend the appropriate statistical test based on
    the research question and selected columns.
    Returns recommended test + rationale + assumption checks.
    """
    _check_session(session_id)

    if not body.question or len(body.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question is too short.")

    if not body.columns:
        raise HTTPException(status_code=400, detail="At least one column must be provided.")

    df = session_service.get_working_df(session_id)

    for col in body.columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")

    result = stats_service.recommend_test(df, body.question, body.columns)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    from services.stats_service import _sanitize_result
    return {
        "session_id": session_id,
        "question": body.question,
        "recommendation": _sanitize_result(result),
    }


#   POST /api/stats/{session_id}/run  

@router.post("/{session_id}/run")
async def run_test(session_id: str, body: RunTestRequest):
    """
    Run a specific statistical test on the working dataset.
    Returns test statistic, p-value, effect size, assumption
    checks, and plain English explanation (Gemini or rule-based).
    Result is appended to stats history.
    """
    _check_session(session_id)

    if body.test_name not in VALID_TESTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown test_name '{body.test_name}'. "
                   f"Valid tests: {sorted(VALID_TESTS)}",
        )

    df = session_service.get_working_df(session_id)

    payload = {"test_name": body.test_name}
    payload.update(body.options)

    try:
        result = await stats_service.run_test(df, payload)
    except ColumnNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Stats test failed: {e}")
        raise HTTPException(status_code=500, detail="Statistical test failed.")

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Append to stats history
    append_stats_history(session_id, {
        "test_name": body.test_name,
        "options": body.options,
        "significant": result.get("significant"),
        "p_value": result.get("p_value"),
        "plain_english": result.get("plain_english"),
    })

    logger.info(
        f"Stats test run | session_id={session_id} | "
        f"test={body.test_name} | p={result.get('p_value')}"
    )

    return {
        "session_id": session_id,
        "test_name": body.test_name,
        "result": result,
    }


#   GET /api/stats/{session_id}/history  

@router.get("/{session_id}/history")
def get_history(session_id: str):
    """
    Return the stats test history for this session.
    Frontend uses this to populate the History Panel.
    """
    _check_session(session_id)

    history = get_stats_history(session_id)

    return {
        "session_id": session_id,
        "history": history,
        "total_tests": len(history),
    }


#   POST /api/stats/{session_id}/question  

@router.post("/{session_id}/question")
def stats_question(session_id: str, body: StatsQuestionRequest):
    """
    Answer a plain English stats question by routing to
    the recommend endpoint and returning the recommendation
    with a plain English rationale.
    """
    _check_session(session_id)

    if not body.question or len(body.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question is too short.")

    df = session_service.get_working_df(session_id)

    columns = body.columns or df.columns.tolist()[:3]

    result = stats_service.recommend_test(df, body.question, columns)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    from services.stats_service import _sanitize_result
    clean = _sanitize_result(result)
    return {
        "session_id": session_id,
        "question": body.question,
        "recommended_test": clean.get("recommended_test"),
        "rationale": clean.get("rationale"),
        "assumptions": clean.get("assumptions"),
    }




