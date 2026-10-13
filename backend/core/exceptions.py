"""
Exception handling
------------------
Every error the API returns has the same shape:

    { "error": "...", "detail": "...", "code": 4xx|5xx, "request_id": "..." }

  error       the message the UI shows in its toast
  detail      a longer explanation (often the same text)
  code        the HTTP status
  request_id  present when the request-ID middleware ran; quote it when
              reporting a problem, because it appears in the server log

Nothing sent to a client contains a traceback, a filesystem path or a secret:
every message goes through safe_detail().
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.security import GENERIC_ERROR_DETAIL, safe_detail, sanitize_filename

logger = logging.getLogger(__name__)


# Domain exceptions

class SessionNotFoundError(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class InvalidFileTypeError(Exception):
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"Invalid file type: {filename}")


class FileTooLargeError(Exception):
    def __init__(self, size_mb: float, max_mb: int):
        super().__init__(f"File size {size_mb:.1f}MB exceeds limit of {max_mb}MB")


class ColumnNotFoundError(Exception):
    def __init__(self, column: str):
        self.column = column
        super().__init__(f"Column not found in dataset: {column}")


class InvalidFormulaError(Exception):
    """`reason` is shown to the user, so it should explain what is wrong."""

    def __init__(self, reason: str = "The formula is not valid."):
        self.reason = reason
        super().__init__(reason)


class ProfileNotRunError(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"Profiling has not been run for session: {session_id}")


class SnapshotNotFoundError(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"No snapshot found to restore for session: {session_id}")


# Response builder

def _request_id(request: Request) -> str | None:
    rid = getattr(getattr(request, "state", None), "request_id", None)
    return rid if isinstance(rid, str) and rid else None


def error_response(
    request: Request,
    status_code: int,
    error: str,
    detail: str | None = None,
    headers: dict | None = None,
    **extra,
) -> JSONResponse:
    """Build the standard error response. All text is made safe first."""
    body = {
        "error": safe_detail(error),
        "detail": safe_detail(detail if detail is not None else error),
        "code": status_code,
    }
    out_headers = dict(headers or {})

    rid = _request_id(request)
    if rid:
        body["request_id"] = rid
        out_headers["X-Request-ID"] = rid

    body.update(extra)
    return JSONResponse(status_code=status_code, content=body, headers=out_headers or None)


# Handlers

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Covers errors our routers raise and errors Starlette raises itself
    (unknown URL, wrong method), because FastAPI's HTTPException subclasses
    Starlette's.
    """
    if exc.status_code >= 500:
        logger.warning(
            "HTTP %s | request_id=%s | %s %s | %s",
            exc.status_code, _request_id(request), request.method, request.url.path, exc.detail,
        )
    return error_response(
        request, exc.status_code, exc.detail, headers=getattr(exc, "headers", None)
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Pydantic validation failures (422). FastAPI's default response echoes the
    submitted input back; this one reports only the field and the problem.
    """
    errors = exc.errors()
    fields = []
    for err in errors[:5]:
        if err.get("type") == "json_invalid":
            fields.append({"field": "body", "message": "The request body is not valid JSON."})
            continue
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path", "header", "cookie")]
        fields.append({
            "field": ".".join(loc) or "request",
            "message": safe_detail(err.get("msg", "Invalid value"), max_len=120),
        })

    summary = "; ".join(f"{f['field']}: {f['message']}" for f in fields) or "The request is not valid."
    if len(errors) > len(fields):
        summary += f" (and {len(errors) - len(fields)} more)"

    return error_response(request, 422, f"Invalid request. {summary}", fields=fields)


async def generic_exception_handler(request: Request, exc: Exception):
    """Anything unexpected. The client gets a generic message; the log gets everything."""
    rid = _request_id(request)
    logger.error(
        "Unhandled exception | request_id=%s | %s %s",
        rid, request.method, request.url.path, exc_info=exc,
    )
    return error_response(
        request,
        500,
        GENERIC_ERROR_DETAIL,
        detail=f"Reference: {rid}" if rid else GENERIC_ERROR_DETAIL,
    )


def _domain_handler(status_code: int, title: str, detail_fn=str):
    async def handler(request: Request, exc: Exception):
        return error_response(request, status_code, title, detail=detail_fn(exc))
    return handler


# Kept as named module-level handlers so older code importing them still works.
session_not_found_handler = _domain_handler(404, "Session not found")
invalid_file_type_handler = _domain_handler(
    400,
    "Invalid file type",
    lambda e: f"File '{sanitize_filename(e.filename)}' is not a supported type. "
              f"Upload .csv, .xlsx, or .xls files only.",
)
file_too_large_handler = _domain_handler(413, "File too large")
column_not_found_handler = _domain_handler(400, "Column not found")
invalid_formula_handler = _domain_handler(400, "Invalid formula")
profile_not_run_handler = _domain_handler(404, "Profile not run")
snapshot_not_found_handler = _domain_handler(404, "Nothing to undo")


# Registration

def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the app. main.py calls this once."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SessionNotFoundError, session_not_found_handler)
    app.add_exception_handler(InvalidFileTypeError, invalid_file_type_handler)
    app.add_exception_handler(FileTooLargeError, file_too_large_handler)
    app.add_exception_handler(ColumnNotFoundError, column_not_found_handler)
    app.add_exception_handler(InvalidFormulaError, invalid_formula_handler)
    app.add_exception_handler(ProfileNotRunError, profile_not_run_handler)
    app.add_exception_handler(SnapshotNotFoundError, snapshot_not_found_handler)
    app.add_exception_handler(Exception, generic_exception_handler)