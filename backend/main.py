"""
Initializes the FastAPI application, registers all routers,
configures CORS, registers exception handlers, and mounts health check.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from core.config import settings
from core.logging_config import setup_logging
from core.exceptions import (
    SessionNotFoundError,
    InvalidFileTypeError,
    FileTooLargeError,
    ColumnNotFoundError,
    InvalidFormulaError,
    ProfileNotRunError,
    SnapshotNotFoundError,
    http_exception_handler,
    session_not_found_handler,
    invalid_file_type_handler,
    file_too_large_handler,
    column_not_found_handler,
    invalid_formula_handler,
    profile_not_run_handler,
    snapshot_not_found_handler,
    generic_exception_handler,
)

from fastapi.responses import JSONResponse
import math
import json

class NaNSafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _replace_nan(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif isinstance(obj, dict):
            return {k: NaNSafeJSONResponse._replace_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NaNSafeJSONResponse._replace_nan(v) for v in obj]
        return obj

    def render(self, content) -> bytes:
        clean = NaNSafeJSONResponse._replace_nan(content)
        return json.dumps(clean, ensure_ascii=False).encode("utf-8")
#  Routers 
from routers import dataset, profile, clean, transform, eda, stats, nlp, ml, report, ai

#  Logging 
setup_logging()


#  Lifespan 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure upload directories exist on startup."""
    for path in [
        settings.original_path,
        settings.working_path,
        settings.history_path,
        settings.snapshots_path,
        settings.reports_path,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    yield


#  App 
app = FastAPI(
    title="DataSaathi API",
    description="Intelligent Data Analysis Platform — Backend",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
    default_response_class=NaNSafeJSONResponse,
)

#  CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Exception Handlers 
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(SessionNotFoundError, session_not_found_handler)
app.add_exception_handler(InvalidFileTypeError, invalid_file_type_handler)
app.add_exception_handler(FileTooLargeError, file_too_large_handler)
app.add_exception_handler(ColumnNotFoundError, column_not_found_handler)
app.add_exception_handler(InvalidFormulaError, invalid_formula_handler)
app.add_exception_handler(ProfileNotRunError, profile_not_run_handler)
app.add_exception_handler(SnapshotNotFoundError, snapshot_not_found_handler)
app.add_exception_handler(Exception, generic_exception_handler)

#  Routers 
app.include_router(dataset.router)
app.include_router(profile.router)
app.include_router(clean.router)
app.include_router(transform.router)
app.include_router(eda.router)
app.include_router(stats.router)
app.include_router(nlp.router)
app.include_router(ml.router)
app.include_router(report.router)
app.include_router(ai.router)


#  Health Check 
@app.get("/api/health", tags=["health"])
def health_check():
    """Simple liveness check."""
    return {"status": "ok", "version": "1.0.0"}


