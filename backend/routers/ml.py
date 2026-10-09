"""
ML Strategy Router
------------------
Advisory only: no models are trained at any endpoint.

Endpoints:
  POST /api/ml/{session_id}/recommend   - build the full ML strategy, cache it
  GET  /api/ml/{session_id}/summary     - compact summary (used by the report)
  GET  /api/ml/{session_id}/models      - ranked models + comparison table
  GET  /api/ml/{session_id}/roadmap     - roadmap + pipeline architecture
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import validate_session_id
from core.cache import set_ml_result, get_ml_result
from core.exceptions import ColumnNotFoundError
from services import session_service
from services import ml_service

router = APIRouter(prefix="/api/ml", tags=["ml"])
logger = logging.getLogger(__name__)


#    Request Bodies  

class RecommendRequest(BaseModel):
    target: Optional[str] = None          # omit for clustering / unsupervised
    objective: Optional[str] = ""         # e.g. "predict which customers will churn"
    problem_type: Optional[str] = None    # optional manual override


#    Helpers  

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


def _get_cached_strategy(session_id: str) -> dict:
    result = get_ml_result(session_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No ML strategy found. Call POST /recommend first.",
        )
    return result


#    POST /api/ml/{session_id}/recommend  

@router.post("/{session_id}/recommend")
async def recommend(session_id: str, body: RecommendRequest):
    """
    Build the complete ML strategy for the working dataset.
    Detects the problem type, ranks models, recommends preprocessing,
    imbalance handling, validation and metrics, and builds a roadmap.
    Gemini tailors the reasoning to the user's stated objective;
    rule-based reasoning is used if Gemini is unavailable.
    The result is cached for the other three endpoints.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)

    objective = (body.objective or "").strip()
    if len(objective) > 500:
        raise HTTPException(
            status_code=400,
            detail="Objective is too long. Keep it under 500 characters.",
        )

    if body.target and body.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{body.target}' not found.")

    try:
        strategy = await ml_service.build_ml_strategy(
            df,
            target=body.target,
            objective=objective,
            problem_type_override=body.problem_type,
        )
    except ColumnNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        # Invalid problem_type override, or supervised override without a target
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"ML strategy failed for session_id={session_id}: {e}")
        raise HTTPException(status_code=500, detail="ML strategy generation failed.")

    # Target cannot be used (constant, or too many classes)
    if not strategy.get("valid", False):
        raise HTTPException(
            status_code=400,
            detail=strategy.get("message", "The selected target cannot be used."),
        )

    set_ml_result(session_id, strategy)

    logger.info(
        f"ML strategy built | session_id={session_id} | "
        f"problem={strategy['problem']['problem_type']} | "
        f"top_model={strategy['models'][0]['name'] if strategy['models'] else None} | "
        f"reasoning={strategy['reasoning_source']}"
    )

    return {"session_id": session_id, "strategy": strategy}


#    GET /api/ml/{session_id}/summary  

@router.get("/{session_id}/summary")
def get_summary(session_id: str):
    """
    Compact ML strategy summary: problem type, readiness, top models,
    primary metric and roadmap titles. Used by Stage 9 (Report).
    """
    _check_session(session_id)
    strategy = _get_cached_strategy(session_id)
    return {"session_id": session_id, "summary": ml_service.ml_summary(strategy)}


#    GET /api/ml/{session_id}/models  

@router.get("/{session_id}/models")
def get_models(session_id: str):
    """
    Ranked model recommendations and the comparison table.
    Frontend uses this for the model cards and comparison table.
    """
    _check_session(session_id)
    strategy = _get_cached_strategy(session_id)
    return {
        "session_id": session_id,
        "problem_type": strategy["problem"]["problem_type"],
        "models": strategy["models"],
        "model_comparison": strategy["model_comparison"],
    }


#   GET /api/ml/{session_id}/roadmap  

@router.get("/{session_id}/roadmap")
def get_roadmap(session_id: str):
    """
    Step-by-step modelling roadmap and pipeline architecture.
    Frontend uses this for the roadmap list and the architecture diagram.
    """
    _check_session(session_id)
    strategy = _get_cached_strategy(session_id)
    return {
        "session_id": session_id,
        "roadmap": strategy["roadmap"],
        "architecture": strategy["architecture"],
    }