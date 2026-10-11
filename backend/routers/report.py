"""
Report Router
-------------
Endpoints:
  GET  /api/report/{session_id}/sections             - which sections have data (checklist badges)
  POST /api/report/{session_id}/generate             - build the report as PDF or HTML
  GET  /api/report/{session_id}/download/{format}    - stream the generated file
  GET  /api/report/{session_id}/history-export       - download the operation history as JSON

The router never returns a filesystem path. Clients get download URLs only.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from core.security import validate_session_id
from services import session_service
from services import report_service
from services.report_service import PDFEngineUnavailable

router = APIRouter(prefix="/api/report", tags=["report"])
logger = logging.getLogger(__name__)


#    Request Bodies     

class GenerateRequest(BaseModel):
    sections: Optional[list[str]] = None   # omit for all sections
    format: str = "html"                   # "pdf" | "html"


#    Helpers  

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


def _check_format(fmt: str) -> str:
    fmt = (fmt or "").lower()
    if fmt not in report_service.VALID_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{fmt}'. Valid formats: {sorted(report_service.VALID_FORMATS)}",
        )
    return fmt


MEDIA_TYPES = {"pdf": "application/pdf", "html": "text/html; charset=utf-8"}


#    GET /api/report/{session_id}/sections  

@router.get("/{session_id}/sections")
def get_sections(session_id: str):
    """
    List the 10 report sections and whether each has data yet.
    Frontend uses this for the checklist's completion badges.
    Also reports whether PDF export is available on this server.
    """
    _check_session(session_id)
    return {
        "session_id": session_id,
        "sections": report_service.get_section_availability(session_id),
        "formats": {"html": True, "pdf": report_service.pdf_available()},
    }


#   POST /api/report/{session_id}/generate  

@router.post("/{session_id}/generate")
async def generate(session_id: str, body: GenerateRequest):
    """
    Build the report from the completed stages and write it as PDF or HTML.
    Sections that were never run are included as "Not performed".
    The HTML version is always written so it can be previewed.
    """
    _check_session(session_id)

    fmt = _check_format(body.format)

    if body.sections is not None:
        if len(body.sections) == 0:
            raise HTTPException(status_code=400, detail="Select at least one section.")
        unknown = [s for s in body.sections if s not in report_service.SECTION_KEYS]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown section(s): {unknown}. Valid sections: {report_service.SECTION_KEYS}",
            )

    try:
        result = await report_service.generate_report(session_id, body.sections, fmt)
    except PDFEngineUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Report generation failed for session_id={session_id}: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed. Please try again.")

    return {
        "session_id": session_id,
        "status": "generated",
        **result,
        "download_url": f"/api/report/{session_id}/download/{fmt}",
        "preview_url": f"/api/report/{session_id}/download/html?inline=true",
    }


#   GET /api/report/{session_id}/download/{format}  

@router.get("/{session_id}/download/{fmt}")
def download(session_id: str, fmt: str, inline: bool = False):
    """
    Stream the generated report.
    Default is a file download (Content-Disposition: attachment).
    ?inline=true displays it in the browser instead, for the preview pane.
    """
    _check_session(session_id)
    fmt = _check_format(fmt)

    path = report_service.get_report_file(session_id, fmt)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {fmt.upper()} report found. Call POST /generate with format '{fmt}' first.",
        )

    headers = {}
    if fmt == "html":
        # The report has no scripts or external resources; make sure it stays that way.
        headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; img-src data:"
        )
        headers["X-Content-Type-Options"] = "nosniff"

    return FileResponse(
        path,
        media_type=MEDIA_TYPES[fmt],
        filename=f"datasaathi_report.{fmt}",
        content_disposition_type="inline" if inline else "attachment",
        headers=headers,
    )


#   GET /api/report/{session_id}/history-export  

@router.get("/{session_id}/history-export")
def history_export(session_id: str):
    """
    Download the complete operation history as JSON: every cleaning and
    transformation, plus the EDA, statistics and NLP analyses that were run.
    """
    _check_session(session_id)

    payload = report_service.build_history_export(session_id)

    return Response(
        content=json.dumps(payload, indent=2, default=str),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="datasaathi_history_{session_id[:8]}.json"',
        },
    )






