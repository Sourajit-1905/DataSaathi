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


# Clean / Stage 3 

class MissingNumericalIssue(BaseModel):
    column: str
    missing_count: int
    missing_pct: float
    skewness: float


class MissingCategoricalIssue(BaseModel):
    column: str
    missing_count: int
    missing_pct: float
    mode: str


class DuplicateIssue(BaseModel):
    duplicate_count: int
    duplicate_pct: float
    sample_rows: list[dict[str, Any]]


class OutlierIssue(BaseModel):
    column: str
    outlier_count: int
    outlier_pct: float
    lower_bound: float
    upper_bound: float


class StringIssue(BaseModel):
    column: str
    whitespace_count: int
    mixed_case: bool


class ConstantColumnIssue(BaseModel):
    column: str
    dominant_value: str
    frequency_pct: float


class NegativeValueIssue(BaseModel):
    column: str
    negative_count: int
    negative_pct: float
    minimum_value: float
    maximum_value: float
    semantic_class: str
    semantic_confidence: str
    semantic_reason: str
    tiny_negative_noise: bool


class RecommendationResult(BaseModel):
    issue_type: str
    column: Optional[str] = None
    recommended_strategy: str
    available_strategies: list[str]
    explanation: str
    stats: Optional[dict[str, Any]] = None


class CleanApplyResponse(BaseModel):
    session_id: str
    status: str
    issue_type: str
    strategy: str
    column: Optional[str] = None
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    warning: Optional[str] = None


class CleanUndoResponse(BaseModel):
    session_id: str
    status: str
    message: str


class CleanHistoryEntry(BaseModel):
    stage: str
    issue_type: str
    column: Optional[str] = None
    strategy: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    timestamp: str


class CleanHistoryResponse(BaseModel):
    session_id: str
    history: list[CleanHistoryEntry]
    total_operations: int



# Transform / Stage 4 

class ColumnStats(BaseModel):
    column: str
    dtype: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    has_negative: Optional[bool] = None
    has_zero: Optional[bool] = None
    unique_count: Optional[int] = None
    top_values: Optional[list[Any]] = None


class TransformRecommendation(BaseModel):
    column: str
    recommended_transform: str
    available_transforms: list[str]
    stats: dict[str, Any]
    explanation: str


class TransformPreviewStats(BaseModel):
    before: Optional[dict[str, Any]] = None
    after: Optional[dict[str, Any]] = None
    note: Optional[str] = None
    new_columns: Optional[list[str]] = None
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    cols_before: Optional[int] = None
    cols_after: Optional[int] = None


class TransformApplyResponse(BaseModel):
    session_id: str
    status: str
    transform_type: str
    column: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    new_columns: list[str]
    note: Optional[str] = None


class TransformUndoResponse(BaseModel):
    session_id: str
    status: str
    message: str


class TransformHistoryEntry(BaseModel):
    stage: str
    transform_type: str
    column: str
    options: Optional[dict[str, Any]] = None
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    new_columns: list[str]
    timestamp: str


class TransformHistoryResponse(BaseModel):
    session_id: str
    history: list[TransformHistoryEntry]
    total_operations: int




#   EDA / Stage 5  

class FilterCondition(BaseModel):
    column: str
    operator: str
    value: Optional[Any] = None


class EDAAnalyzeResponse(BaseModel):
    session_id: str
    analysis_type: str
    filters_applied: int
    rows_analysed: int
    result: dict[str, Any]


class EDAQuestionResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    suggested_analysis: Optional[str] = None
    source: str


class EDAChartRecommendation(BaseModel):
    col_x: str
    col_y: Optional[str] = None
    recommended_chart: str
    reason: str
    alternatives: Optional[list[str]] = None


class EDAInsightsResponse(BaseModel):
    session_id: str
    insights: list[str]
    source: str
    count: int


class EDAHistoryEntry(BaseModel):
    analysis_type: str
    column: Optional[str] = None
    col_x: Optional[str] = None
    col_y: Optional[str] = None
    filters: Optional[list[dict[str, Any]]] = None
    filter_count: Optional[int] = None
    rows_analysed: Optional[int] = None
    question: Optional[str] = None
    source: Optional[str] = None


class EDAHistoryResponse(BaseModel):
    session_id: str
    history: list[EDAHistoryEntry]
    total_analyses: int


class EDASummaryResponse(BaseModel):
    session_id: str
    summary: dict[str, Any]



#    Stats / Stage 6  

class AssumptionCheck(BaseModel):
    passed: bool
    reason: Optional[str] = None
    test: Optional[str] = None
    statistic: Optional[float] = None
    p_value: Optional[float] = None


class EffectSize(BaseModel):
    label: str
    cohen_d: Optional[float] = None
    r: Optional[float] = None
    eta_squared: Optional[float] = None
    cramers_v: Optional[float] = None
    r_squared: Optional[float] = None
    odds_ratio: Optional[float] = None


class StatsTestResult(BaseModel):
    test: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: dict[str, Any]
    interpretation: str
    plain_english: str
    assumption_checks: dict[str, Any]


class StatsRecommendation(BaseModel):
    recommended_test: str
    rationale: str
    column_types: dict[str, str]
    assumptions: dict[str, Any]
    available_tests: list[str]


class StatsRunResponse(BaseModel):
    session_id: str
    test_name: str
    result: dict[str, Any]


class StatsHistoryEntry(BaseModel):
    test_name: str
    options: dict[str, Any]
    significant: Optional[bool] = None
    p_value: Optional[float] = None
    plain_english: Optional[str] = None


class StatsHistoryResponse(BaseModel):
    session_id: str
    history: list[StatsHistoryEntry]
    total_tests: int


class StatsQuestionResponse(BaseModel):
    session_id: str
    question: str
    recommended_test: str
    rationale: str
    assumptions: dict[str, Any]
