"""
Pydantic request/response schemas for all DataSaathi endpoints.
Schemas for later stages are added here as they are implemented.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime


# Common 

class StatusResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: int


class HealthResponse(BaseModel):
    status: str
    version: str


# Dataset / Stage 1 

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    file_size_kb: float
    upload_time: str


class PreviewResponse(BaseModel):
    session_id: str
    rows: list[dict[str, Any]]
    column_dtypes: dict[str, str]
    total_rows: int
    total_columns: int


class DatasetInfoResponse(BaseModel):
    session_id: str
    rows: int
    columns: int
    column_names: list[str]
    column_dtypes: dict[str, str]
    memory_usage_mb: float


class ResetResponse(BaseModel):
    status: str
    message: str
    session_id: str
