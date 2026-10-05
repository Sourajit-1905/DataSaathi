from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import logging

logger = logging.getLogger(__name__)


# ── Custom Exception Classes ─────────────────────────────────

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
    def __init__(self, formula: str):
        self.formula = formula
        super().__init__(f"Formula contains disallowed content: {formula}")


class ProfileNotRunError(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"Profiling has not been run for session: {session_id}")


class SnapshotNotFoundError(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"No snapshot found to restore for session: {session_id}")


# ── FastAPI Exception Handlers ───────────────────────────────

async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": str(exc.detail),
            "code": exc.status_code,
        },
    )


async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Session not found",
            "detail": str(exc),
            "code": 404,
        },
    )


async def invalid_file_type_handler(request: Request, exc: InvalidFileTypeError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid file type",
            "detail": f"File '{exc.filename}' is not a supported type. Upload .csv, .xlsx, or .xls files only.",
            "code": 400,
        },
    )


async def file_too_large_handler(request: Request, exc: FileTooLargeError):
    return JSONResponse(
        status_code=413,
        content={
            "error": "File too large",
            "detail": str(exc),
            "code": 413,
        },
    )


async def column_not_found_handler(request: Request, exc: ColumnNotFoundError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Column not found",
            "detail": str(exc),
            "code": 400,
        },
    )


async def invalid_formula_handler(request: Request, exc: InvalidFormulaError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid formula",
            "detail": str(exc),
            "code": 400,
        },
    )


async def profile_not_run_handler(request: Request, exc: ProfileNotRunError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Profile not run",
            "detail": str(exc),
            "code": 404,
        },
    )


async def snapshot_not_found_handler(request: Request, exc: SnapshotNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Snapshot not found",
            "detail": str(exc),
            "code": 404,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
            "code": 500,
        },
    )
