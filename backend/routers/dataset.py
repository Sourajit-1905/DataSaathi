"""
Dataset Router
--------------
Handles file upload, preview, reset, and dataset info.

Endpoints:
  POST /api/dataset/upload               upload CSV/XLSX/XLS, create session
  GET  /api/dataset/{session_id}/preview first 10 rows + dtypes
  POST /api/dataset/{session_id}/reset   restore original, clear history and cached results
  GET  /api/dataset/{session_id}/info    rows, cols, dtypes, memory

Upload checks, in order: extension, MIME type, size (also enforced earlier by
BodyLimitMiddleware), file signature, .xlsx zip safety, parse, shape.
The upload and preview/reset/info endpoints are plain `def`, so FastAPI runs them
in a worker thread and parsing a large file never blocks other requests.
"""

import io
import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from core.config import settings
from core.security import (
    ALLOWED_EXTENSIONS,
    SNIFF_BYTES,
    UPLOAD_CHUNK_BYTES,
    check_xlsx_zip,
    get_extension,
    sanitize_filename,
    validate_mime_type,
    validate_session_id,
    validate_upload_content,
)
from models.schemas import (
    DatasetInfoResponse,
    PreviewResponse,
    ResetResponse,
    UploadResponse,
)
from services import session_service

router = APIRouter(prefix="/api/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)

MAX_COLUMNS = 1000


def _check_session(session_id: str):
    """Validate UUID format and existence, raise HTTPException if invalid."""
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


def _read_upload(file: UploadFile, limit: int) -> bytes:
    """Read the upload in chunks, stopping as soon as it exceeds the limit."""
    buffer = io.BytesIO()
    total = 0
    while True:
        chunk = file.file.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"The file is too large. The maximum upload size is {settings.MAX_UPLOAD_MB} MB.",
            )
        buffer.write(chunk)
    return buffer.getvalue()


def _parse(contents: bytes, kind: str, filename: str) -> pd.DataFrame:
    """Parse by what the file really is (kind), turning parser errors into clear messages."""
    try:
        if kind == "csv":
            return pd.read_csv(io.BytesIO(contents), low_memory=False)
        engine = "openpyxl" if kind == "xlsx" else "xlrd"
        return pd.read_excel(io.BytesIO(contents), engine=engine)
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="The file is not UTF-8 encoded. In Excel use Save As, choose CSV UTF-8, and upload that.",
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file contains no data.")
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400,
            detail="The file could not be read as a CSV. Check that every row has the same number of columns.",
        )
    except Exception as e:
        logger.warning("Could not parse upload %r: %s: %s", filename, type(e).__name__, str(e)[:200])
        raise HTTPException(
            status_code=400,
            detail="Could not read the file. Make sure it is a valid CSV or Excel file.",
        )


#   POST /api/dataset/upload  

@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a .csv, .xlsx, or .xls file (size limit: MAX_UPLOAD_MB).
    Returns session_id + dataset metadata.
    """
    filename = sanitize_filename(file.filename)
    ext = get_extension(filename)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Upload .csv, .xlsx, or .xls files only.",
        )

    if not validate_mime_type(file.content_type, filename):
        raise HTTPException(
            status_code=400,
            detail="This file type is not allowed. Upload a .csv, .xlsx, or .xls file.",
        )

    contents = _read_upload(file, settings.max_upload_bytes)

    kind, error = validate_upload_content(contents[:SNIFF_BYTES], ext)
    if error:
        raise HTTPException(status_code=400, detail=error)

    if kind == "xlsx":
        ok, message = check_xlsx_zip(contents)
        if not ok:
            raise HTTPException(status_code=400, detail=message)

    df = _parse(contents, kind, filename)

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file contains no data.")
    if len(df.columns) > MAX_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"The file has {len(df.columns)} columns; the maximum is {MAX_COLUMNS}.",
        )

    # Excel headers can be numbers or dates; column names are always text from here on.
    df.columns = [str(c) for c in df.columns]

    try:
        result = session_service.create_session(df, filename)
    except Exception:
        logger.exception("Failed to create session for %r", filename)
        raise HTTPException(status_code=500, detail="Failed to create session. Please try again.")

    return UploadResponse(**result)


#   GET /api/dataset/{session_id}/preview  

@router.get("/{session_id}/preview", response_model=PreviewResponse)
def preview_dataset(session_id: str):
    """First 10 rows of the working dataset, plus a column to dtype map."""
    _check_session(session_id)

    df = session_service.get_working_df(session_id)
    head = df.head(10)
    preview_rows = head.where(pd.notnull(head), None).to_dict(orient="records")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return PreviewResponse(
        session_id=session_id,
        rows=preview_rows,
        column_dtypes=dtypes,
        total_rows=len(df),
        total_columns=len(df.columns),
    )


#   POST /api/dataset/{session_id}/reset  

@router.post("/{session_id}/reset", response_model=ResetResponse)
def reset_dataset(session_id: str):
    """
    Restore the working file to the original upload and clear all history.
    Cached profile, EDA, stats, NLP and ML results and any generated reports are
    cleared too, because they described data that no longer exists.
    """
    _check_session(session_id)
    session_service.reset_to_original(session_id)

    return ResetResponse(
        status="reset",
        message="Dataset restored to original upload. All cleaning and transformation history, "
                "analysis results and reports have been cleared.",
        session_id=session_id,
    )


#   GET /api/dataset/{session_id}/info  

@router.get("/{session_id}/info", response_model=DatasetInfoResponse)
def dataset_info(session_id: str):
    """Rows, columns, column names, dtype map, and memory usage."""
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