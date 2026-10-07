"""
EDA Router
----------
Endpoints:
  POST /api/eda/{session_id}/analyze          — run a specific EDA analysis
  POST /api/eda/{session_id}/question         — answer a natural language question
  GET  /api/eda/{session_id}/recommend-chart  — recommend chart for column(s)
  GET  /api/eda/{session_id}/insights         — auto-generate EDA insights
  GET  /api/eda/{session_id}/history          — return EDA history
  GET  /api/eda/{session_id}/summary          — return end-of-stage EDA summary
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from core.security import validate_session_id
from core.cache import append_eda_history, get_eda_history
from services import session_service
from services import eda_service

router = APIRouter(prefix="/api/eda", tags=["eda"])
logger = logging.getLogger(__name__)


#   Request Bodies  

class AnalyzeRequest(BaseModel):
    analysis_type: str
    column: Optional[str] = None
    col_x: Optional[str] = None
    col_y: Optional[str] = None
    columns: Optional[list[str]] = None
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    freq: Optional[str] = "M"
    filters: Optional[list[dict[str, Any]]] = []
    options: Optional[dict[str, Any]] = {}


class QuestionRequest(BaseModel):
    question: str
    filters: Optional[list[dict[str, Any]]] = []


#   Helpers  

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


VALID_ANALYSIS_TYPES = {
    "univariate_numerical",
    "univariate_categorical",
    "bivariate_num_num",
    "bivariate_num_cat",
    "bivariate_cat_cat",
    "time_series",
    "multivariate",
    "outlier",
}


#   POST /api/eda/{session_id}/analyze  

@router.post("/{session_id}/analyze")
async def analyze(session_id: str, body: AnalyzeRequest):
    """
    Run a specific EDA analysis on the working dataset.
    Filters are applied in memory — working CSV is never modified.
    Result is appended to EDA history.
    """
    _check_session(session_id)

    if body.analysis_type not in VALID_ANALYSIS_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown analysis_type '{body.analysis_type}'. "
                   f"Valid types: {sorted(VALID_ANALYSIS_TYPES)}",
        )

    df = session_service.get_working_df(session_id)

    # Apply filters (read-only)
    if body.filters:
        df = eda_service.apply_filters(df, body.filters)
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="Filters resulted in an empty dataset. Relax your filter conditions.",
            )

    try:
        if body.analysis_type == "univariate_numerical":
            if not body.column:
                raise HTTPException(status_code=400, detail="'column' is required for univariate_numerical.")
            result = eda_service.numerical_univariate(df, body.column)

        elif body.analysis_type == "univariate_categorical":
            if not body.column:
                raise HTTPException(status_code=400, detail="'column' is required for univariate_categorical.")
            result = eda_service.categorical_univariate(df, body.column)

        elif body.analysis_type == "bivariate_num_num":
            col_x = body.col_x or body.column
            col_y = body.col_y
            if not col_x or not col_y:
                raise HTTPException(status_code=400, detail="'col_x' and 'col_y' are required for bivariate_num_num.")
            result = eda_service.numerical_vs_numerical(df, col_x, col_y)

        elif body.analysis_type == "bivariate_num_cat":
            if not body.col_x or not body.col_y:
                raise HTTPException(status_code=400, detail="'col_x' (numerical) and 'col_y' (categorical) are required.")
            result = eda_service.numerical_vs_categorical(df, body.col_x, body.col_y)

        elif body.analysis_type == "bivariate_cat_cat":
            if not body.col_x or not body.col_y:
                raise HTTPException(status_code=400, detail="'col_x' and 'col_y' are required for bivariate_cat_cat.")
            result = eda_service.categorical_vs_categorical(df, body.col_x, body.col_y)

        elif body.analysis_type == "time_series":
            if not body.date_col or not body.value_col:
                raise HTTPException(status_code=400, detail="'date_col' and 'value_col' are required for time_series.")
            result = eda_service.time_based_eda(df, body.date_col, body.value_col, body.freq or "M")

        elif body.analysis_type == "multivariate":
            result = eda_service.multivariate_analysis(df, body.columns)

        elif body.analysis_type == "outlier":
            if not body.column:
                raise HTTPException(status_code=400, detail="'column' is required for outlier analysis.")
            result = eda_service.outlier_investigation(df, body.column)

        else:
            raise HTTPException(status_code=400, detail=f"Unhandled analysis_type: {body.analysis_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"EDA analyze failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")

    # Check for service-level errors
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Append to EDA history
    append_eda_history(session_id, {
        "analysis_type": body.analysis_type,
        "column": body.column,
        "col_x": body.col_x,
        "col_y": body.col_y,
        "filters": body.filters,
        "filter_count": len(body.filters or []),
        "rows_analysed": len(df),
    })

    logger.info(
        f"EDA analyze | session_id={session_id} | "
        f"type={body.analysis_type} | rows={len(df)}"
    )

    return {
        "session_id": session_id,
        "analysis_type": body.analysis_type,
        "filters_applied": len(body.filters or []),
        "rows_analysed": len(df),
        "result": result,
    }


#   POST /api/eda/{session_id}/question  

@router.post("/{session_id}/question")
async def question(session_id: str, body: QuestionRequest):
    """
    Answer a natural language question about the dataset.
    Gemini understands the question and routes to the right analysis.
    Falls back to keyword-based routing if Gemini is unavailable.
    """
    _check_session(session_id)

    if not body.question or len(body.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question is too short.")

    df = session_service.get_working_df(session_id)

    if body.filters:
        df = eda_service.apply_filters(df, body.filters)

    try:
        result = await eda_service.answer_question(df, body.question)
    except Exception as e:
        logger.exception(f"EDA question failed: {e}")
        raise HTTPException(status_code=500, detail="Question answering failed.")

    # Append to EDA history
    append_eda_history(session_id, {
        "analysis_type": "question",
        "question": body.question,
        "filters": body.filters,
        "rows_analysed": len(df),
        "source": result.get("source"),
    })

    return {
        "session_id": session_id,
        "question": body.question,
        "answer": result.get("answer"),
        "suggested_analysis": result.get("suggested_analysis"),
        "source": result.get("source"),
    }


#   GET /api/eda/{session_id}/recommend-chart  

@router.get("/{session_id}/recommend-chart")
def recommend_chart(
    session_id: str,
    col_x: str,
    col_y: Optional[str] = None,
):
    """
    Recommend the best chart type for the given column(s).
    Frontend uses this to pre-select the chart type.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)

    if col_x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col_x}' not found.")

    if col_y and col_y not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col_y}' not found.")

    result = eda_service.recommend_chart(df, col_x, col_y)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"session_id": session_id, "chart_recommendation": result}


#   GET /api/eda/{session_id}/insights  

@router.get("/{session_id}/insights")
async def get_insights(session_id: str):
    """
    Auto-generate EDA insights for the working dataset.
    Gemini rewrites rule-based insights as a narrative.
    Falls back to rule-based if Gemini is unavailable.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)

    try:
        result = await eda_service.generate_eda_insights(df)
    except Exception as e:
        logger.exception(f"EDA insights failed: {e}")
        raise HTTPException(status_code=500, detail="Insight generation failed.")

    return {
        "session_id": session_id,
        "insights": result["insights"],
        "source": result["source"],
        "count": len(result["insights"]),
    }


#   GET /api/eda/{session_id}/history  

@router.get("/{session_id}/history")
def get_history(session_id: str):
    """
    Return the EDA history for this session.
    Frontend uses this to show the analysis log.
    """
    _check_session(session_id)

    history = get_eda_history(session_id)

    return {
        "session_id": session_id,
        "history": history,
        "total_analyses": len(history),
    }


# GET /api/eda/{session_id}/summary 

@router.get("/{session_id}/summary")
def get_summary(session_id: str):
    """
    Return an end-of-stage EDA summary.
    Combines auto-insights with history of analyses run.
    Used by Stage 9 (Report) as Section 5.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)
    history = get_eda_history(session_id)

    result = eda_service.generate_eda_summary(df, history)

    return {
        "session_id": session_id,
        "summary": result,
    }