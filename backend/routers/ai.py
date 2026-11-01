"""
AI Router
---------
Endpoints:
  POST /api/ai/{session_id}/chat          send a message, get an answer
  GET  /api/ai/{session_id}/history        the stored conversation
  POST /api/ai/{session_id}/clear          delete the conversation
  GET  /api/ai/{session_id}/insights       structured insight cards (cached)
  POST /api/ai/{session_id}/explain        explain a term, column or operation
  GET  /api/ai/{session_id}/summarize      a short narrative summary
  GET  /api/ai/{session_id}/suggestions    starter questions (no Gemini call)
  GET  /api/ai/status                      badge status (no session needed)

Read-only GETs use the async session read (aget_working_df) so a slow read
never blocks another request. Chat, explain and insights read-then-write, so
they hold the session lock for the whole operation via ai_service, which in
turn uses session_service's locked helpers.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.security import validate_session_id
from services import ai_service, session_service

router = APIRouter(prefix="/api/ai", tags=["ai"])
logger = logging.getLogger(__name__)


#   Request Bodies  

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ExplainRequest(BaseModel):
    kind: str = "term"
    target: str = Field(..., min_length=1)
    extra: Optional[dict[str, Any]] = None


#   Helpers  

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


#   POST /api/ai/{session_id}/chat  

@router.post("/{session_id}/chat")
async def chat(session_id: str, body: ChatRequest):
    """Send a message to the AI assistant and get an answer."""
    _check_session(session_id)

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    df = await session_service.aget_working_df(session_id)

    try:
        result = await ai_service.chat(session_id, message, df=df)
    except ai_service.RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=f"Too many messages. Please wait {e.retry_after} seconds and try again.",
            headers={"Retry-After": str(e.retry_after)},
        )
    except Exception:
        logger.exception("AI chat failed | session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="The assistant could not answer. Please try again.")

    return {"session_id": session_id, **result}


#   GET /api/ai/{session_id}/history  

@router.get("/{session_id}/history")
def get_history(session_id: str):
    """The stored conversation, oldest first."""
    _check_session(session_id)
    thread = session_service.get_chat(session_id)
    return {"session_id": session_id, "messages": thread, "total_messages": len(thread)}


#   POST /api/ai/{session_id}/clear  

@router.post("/{session_id}/clear")
def clear_history(session_id: str):
    """Delete the stored conversation for this session."""
    _check_session(session_id)
    session_service.clear_chat(session_id)
    return {"session_id": session_id, "status": "cleared", "message": "The conversation has been cleared."}


#   GET /api/ai/{session_id}/insights  

@router.get("/{session_id}/insights")
async def get_insights(session_id: str, refresh: bool = False):
    """Structured insight cards, cached until the analysis results change."""
    _check_session(session_id)
    df = await session_service.aget_working_df(session_id)

    try:
        result = await ai_service.get_insights(session_id, df=df, refresh=refresh)
    except Exception:
        logger.exception("AI insights failed | session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="Could not generate insights. Please try again.")

    return {"session_id": session_id, **result}


#   POST /api/ai/{session_id}/explain  

VALID_EXPLAIN_KINDS = {"term", "column", "operation"}


@router.post("/{session_id}/explain")
async def explain(session_id: str, body: ExplainRequest):
    """Explain a term, a column, or a cleaning/transform operation."""
    _check_session(session_id)

    kind = body.kind.lower().strip()
    if kind not in VALID_EXPLAIN_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind '{body.kind}'. Valid kinds: {sorted(VALID_EXPLAIN_KINDS)}",
        )

    df = await session_service.aget_working_df(session_id)

    try:
        result = await ai_service.explain(session_id, kind, body.target, extra=body.extra, df=df)
    except Exception:
        logger.exception("AI explain failed | session_id=%s | kind=%s", session_id, kind)
        raise HTTPException(status_code=500, detail="Could not generate an explanation. Please try again.")

    return {"session_id": session_id, "kind": kind, "target": body.target.strip()[:200], **result}


#   GET /api/ai/{session_id}/summarize  

@router.get("/{session_id}/summarize")
async def summarize(session_id: str):
    """A short narrative summary of the analysis so far."""
    _check_session(session_id)
    df = await session_service.aget_working_df(session_id)

    try:
        result = await ai_service.summarize(session_id, df=df)
    except Exception:
        logger.exception("AI summarize failed | session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="Could not generate a summary. Please try again.")

    return {"session_id": session_id, **result}


#   GET /api/ai/{session_id}/suggestions  

@router.get("/{session_id}/suggestions")
def get_suggestions(session_id: str):
    """Stage-aware starter questions. Never calls Gemini."""
    _check_session(session_id)
    return {"session_id": session_id, "suggestions": ai_service.get_suggestions(session_id)}


#   GET /api/ai/status  

@router.get("/status")
def get_status():
    """AI badge status: whether Gemini is configured, cooling down, or unavailable."""
    return ai_service.get_status()