"""
DataSaathi — Backend Entry Point
=================================
Creates the FastAPI application, wires middleware, error handling and routers,
and exposes the health check.

Run with ONE worker. The profile, EDA, stats, NLP and ML results are cached in
process memory, so several uvicorn workers would not see each other's data.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.exceptions import register_exception_handlers
from core.gemini_client import gemini_status
from core.logging_config import setup_logging
from core.middleware import NaNSafeJSONResponse, RequestContextMiddleware

#  Routers 
from routers import dataset, profile, clean, transform, eda, stats, nlp, ml, report, ai

#  Logging 
setup_logging()
logger = logging.getLogger(__name__)


def _cors_origins() -> list[str]:
    """CORS_ORIGIN may hold one origin or several, separated by commas."""
    return [o.strip() for o in settings.CORS_ORIGIN.split(",") if o.strip()]


#  Lifespan 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the upload directories and report the AI configuration."""
    for path in [
        settings.original_path,
        settings.working_path,
        settings.history_path,
        settings.snapshots_path,
        settings.reports_path,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    status = gemini_status()
    if status["configured"] and not status["sdk_installed"]:
        logger.warning(
            "GEMINI_API_KEY is set but the google-genai package is not installed. "
            "Run: pip install google-genai"
        )
    elif status["configured"]:
        logger.info("Gemini enabled (model=%s)", status["model"])
    else:
        logger.info("Gemini is not configured; AI features will use rule-based answers.")

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

#  Middleware 
# The last middleware added is the outermost. CORS must wrap the request
# context so that every response, including error responses, carries CORS headers.
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Content-Disposition"],
)

#  Error handling 
register_exception_handlers(app)

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




