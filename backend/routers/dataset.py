"""
Dataset Router
--------------
Handles file upload, preview, reset, and dataset info.

Endpoints:
  POST /api/dataset/upload            — upload CSV/XLSX, create session
  GET  /api/dataset/{session_id}/preview — first 10 rows + dtypes
  POST /api/dataset/{session_id}/reset   — restore original, clear history
  GET  /api/dataset/{session_id}/info    — rows, cols, dtypes, memory
"""

import io
import logging
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from core.config import settings
from core.security import validate_session_id, validate_mime_type
from core.exceptions import SessionNotFoundError
from services import session_service
from models.schemas import (
    UploadResponse,
    PreviewResponse,
    DatasetInfoResponse,
    ResetResponse,
)

router = APIRouter(prefix="/api/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _check_session(session_id: str):
    """Validate UUID format and existence, raise HTTPException if invalid."""
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


# POST /api/dataset/upload 

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a .csv, .xlsx, or .xls file (max 50 MB).
    XLSX/XLS files are converted to CSV.
    Returns session_id + dataset metadata.
    """
    filename = file.filename or "upload"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    # Extension check 
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Upload .csv, .xlsx, or .xls files only.",
        )

    # MIME type check 
    if not validate_mime_type(file.content_type, filename):
        raise HTTPException(
            status_code=400,
            detail=f"File MIME type '{file.content_type}' is not allowed for '{filename}'.",
        )

    # Read file bytes 
    contents = await file.read()

    # Size check 
    if len(contents) > settings.max_upload_bytes:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size {size_mb:.1f}MB exceeds the {settings.MAX_UPLOAD_MB}MB limit.",
        )

    # Parse to DataFrame 
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        else:
            # XLSX / XLS → convert to CSV in memory
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl" if ext == ".xlsx" else "xlrd")
            logger.info(f"Converted {ext} → CSV in memory | filename={filename}")
    except Exception as e:
        logger.warning(f"Failed to parse uploaded file '{filename}': {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse file '{filename}'. Ensure it is a valid CSV or Excel file.",
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no data.")

    # Create session 
    try:
        result = session_service.create_session(df, filename)
    except Exception as e:
        logger.exception(f"Failed to create session for '{filename}': {e}")
        raise HTTPException(status_code=500, detail="Failed to create session. Please try again.")

    return UploadResponse(**result)


# GET /api/dataset/{session_id}/preview 

@router.get("/{session_id}/preview", response_model=PreviewResponse)
def preview_dataset(session_id: str):
    """
    Return first 10 rows of the working dataset as a list of dicts,
    plus a dict of column → dtype string.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)
    preview_rows = df.head(10).where(pd.notnull(df.head(10)), None).to_dict(orient="records")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return PreviewResponse(
        session_id=session_id,
        rows=preview_rows,
        column_dtypes=dtypes,
        total_rows=len(df),
        total_columns=len(df.columns),
    )


# POST /api/dataset/{session_id}/reset 

@router.post("/{session_id}/reset", response_model=ResetResponse)
def reset_dataset(session_id: str):
    """
    Restore the working file to the original upload and clear all history.
    Also invalidates the profile cache.
    """
    _check_session(session_id)

    # Invalidate profile cache
    from core.cache import invalidate_profile
    invalidate_profile(session_id)

    session_service.reset_to_original(session_id)

    return ResetResponse(
        status="reset",
        message="Dataset restored to original upload. All cleaning and transformation history cleared.",
        session_id=session_id,
    )


# GET /api/dataset/{session_id}/info 

@router.get("/{session_id}/info", response_model=DatasetInfoResponse)
def dataset_info(session_id: str):
    """
    Return rows, columns, column names, dtype map, and memory usage.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4)

    return DatasetInfoResponse(
        session_id=session_id,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
        column_dtypes=dtypes,
        memory_usage_mb=memory_mb,
    )
