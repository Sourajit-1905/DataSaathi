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

# Profile / Stage 2  

class QualitySummary(BaseModel):
    score: int
    label: str
    breakdown: list[str]


class DimensionsSummary(BaseModel):
    rows: int
    columns: int
    memory_mb: float
    total_cells: int


class MissingValueEntry(BaseModel):
    count: int
    pct: float
    is_critical: bool


class DuplicateRowSummary(BaseModel):
    count: int
    pct: float
    sample_indices: list[int]


class OutlierEntry(BaseModel):
    outlier_count: int
    pct: float
    lower_bound: float
    upper_bound: float


class DistributionEntry(BaseModel):
    skewness: float
    kurtosis: float
    distribution_label: str


class CorrelationPair(BaseModel):
    col1: str
    col2: str
    correlation: float


class PotentialTarget(BaseModel):
    column: str
    score: int
    suitability: str
    reasoning: str


class ProfileRunResponse(BaseModel):
    session_id: str
    status: str
    quality_summary: QualitySummary
    insights_source: str
    insights_count: int


class ProfileResultResponse(BaseModel):
    session_id: str
    profile: dict[str, Any]


class InsightsResponse(BaseModel):
    session_id: str
    insights: list[str]
    insights_source: str


class QualityResponse(BaseModel):
    session_id: str
    quality_summary: QualitySummary
    dimensions: DimensionsSummary
    missing_values: dict[str, MissingValueEntry]
    duplicate_rows: DuplicateRowSummary