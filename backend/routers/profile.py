"""
Profile Router
--------------
Endpoints:
  POST /api/profile/{session_id}/run       — run all 22 profiling functions, cache result
  GET  /api/profile/{session_id}/result    — return cached profiling result
  GET  /api/profile/{session_id}/insights  — return only the insights list
  GET  /api/profile/{session_id}/quality   — return only the quality summary
"""

import logging
from fastapi import APIRouter, HTTPException

from core.security import validate_session_id
from core.cache import set_profile, get_profile
from core.exceptions import SessionNotFoundError
from services import session_service
from services.profile_service import run_full_profile

router = APIRouter(prefix="/api/profile", tags=["profile"])
logger = logging.getLogger(__name__)


def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


def _check_profile_cached(session_id: str) -> dict:
    result = get_profile(session_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Profiling has not been run yet. Call POST /run first.",
        )
    return result


#  POST /api/profile/{session_id}/run 

@router.post("/{session_id}/run")
async def run_profile(session_id: str):
    """
    Run all 22 profiling functions on the working dataset.
    Results are cached — re-running overwrites the cache.
    Gemini is called here to generate narrative insights.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)

    try:
        result = await run_full_profile(df)
    except Exception as e:
        logger.exception(f"Profiling failed for session_id={session_id}: {e}")
        raise HTTPException(status_code=500, detail="Profiling failed. Please try again.")

    set_profile(session_id, result)
    logger.info(
        f"Profiling complete | session_id={session_id} | "
        f"quality={result['quality_summary']['score']} | "
        f"insights_source={result['insights_source']}"
    )

    return {
        "session_id": session_id,
        "status": "complete",
        "quality_summary": result["quality_summary"],
        "insights_source": result["insights_source"],
        "insights_count": len(result["insights"]),
    }


# GET /api/profile/{session_id}/result 

@router.get("/{session_id}/result")
def get_profile_result(session_id: str):
    """
    Return the full cached profiling result — all 22 function outputs.
    Frontend uses this to populate the entire Profile page.
    """
    _check_session(session_id)
    result = _check_profile_cached(session_id)
    return {"session_id": session_id, "profile": result}


# GET /api/profile/{session_id}/insights 

@router.get("/{session_id}/insights")
def get_insights(session_id: str):
    """
    Return only the insights list and their source (gemini or rule_based).
    Frontend uses this to populate the Insights Panel.
    """
    _check_session(session_id)
    result = _check_profile_cached(session_id)
    return {
        "session_id": session_id,
        "insights": result["insights"],
        "insights_source": result["insights_source"],
    }


# GET /api/profile/{session_id}/quality 

@router.get("/{session_id}/quality")
def get_quality(session_id: str):
    """
    Return only the quality summary — score, label, breakdown.
    Frontend uses this for the Summary Banner stat cards.
    """
    _check_session(session_id)
    result = _check_profile_cached(session_id)
    return {
        "session_id": session_id,
        "quality_summary": result["quality_summary"],
        "dimensions": result["dimensions"],
        "missing_values": result["missing_values"],
        "duplicate_rows": result["duplicate_rows"],
    }