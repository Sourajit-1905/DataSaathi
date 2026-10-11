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


#   NLP / Stage 7  

class TextColumnInfo(BaseModel):
    column: str
    is_text_column: bool
    avg_word_count: float
    avg_length: float
    unique_ratio: float
    sample: list[str]


class NLPDetectColumnsResponse(BaseModel):
    session_id: str
    columns: list[TextColumnInfo]
    text_columns: list[str]


class TextQualityResult(BaseModel):
    column: str
    total: int
    empty_count: int
    empty_pct: float
    duplicate_count: int
    duplicate_pct: float
    very_short_count: int
    noise_count: int
    quality_score: int
    quality_label: str


class SentimentAggregate(BaseModel):
    column: str
    total: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    avg_compound: float
    overall_sentiment: str


class KeywordEntry(BaseModel):
    keyword: str
    score: float
    rank: int


class WordCloudEntry(BaseModel):
    word: str
    count: int
    weight: float


class TopicEntry(BaseModel):
    topic_id: int
    keywords: list[str]
    weights: list[float]
    proportion: float


class SimilarityMatch(BaseModel):
    rank: int
    index: int
    text: str
    similarity: float


class ClusterEntry(BaseModel):
    cluster_id: int
    size: int
    sample_texts: Optional[list[str]] = None
    keywords: Optional[list[str]] = None
    representative_text: Optional[str] = None


class NLPRunRequest(BaseModel):
    analysis_type: str
    column: str
    options: Optional[dict[str, Any]] = {}


class NLPRunResponse(BaseModel):
    session_id: str
    analysis_type: str
    column: str
    result: Any


class NLPQuestionRequest(BaseModel):
    column: str
    query: str
    top_n: Optional[int] = 5


class NLPQuestionResponse(BaseModel):
    session_id: str
    query: str
    column: str
    results: list[SimilarityMatch]


class NLPInsightsResponse(BaseModel):
    session_id: str
    column: str
    insights: list[str]
    source: str
    count: int


class NLPHistoryEntry(BaseModel):
    analysis_type: str
    column: str
    options: Optional[dict[str, Any]] = None
    query: Optional[str] = None


class NLPHistoryResponse(BaseModel):
    session_id: str
    history: list[NLPHistoryEntry]
    total_analyses: int


# ML Strategy / Stage 8  

class MLRecommendRequest(BaseModel):
    target: Optional[str] = None
    objective: Optional[str] = ""
    problem_type: Optional[str] = None


class MLProblemInfo(BaseModel):
    problem_type: str
    target: Optional[str] = None
    n_classes: Optional[int] = None
    confidence: str
    source: str
    reasoning: str


class MLReadiness(BaseModel):
    score: int
    label: str
    n_rows: int
    n_features: int
    rows_per_feature: Optional[float] = None
    feature_missing_pct: float
    warnings: list[str]


class MLLeakageFlag(BaseModel):
    column: str
    reason: str
    correlation: Optional[float] = None


class MLLeakage(BaseModel):
    flagged_features: list[MLLeakageFlag]
    identifier_columns: list[str]
    risk: str


class MLCollinearPair(BaseModel):
    col1: str
    col2: str
    correlation: float


class MLMulticollinearity(BaseModel):
    has_multicollinearity: bool
    pairs: list[MLCollinearPair]


class MLImbalance(BaseModel):
    applicable: bool
    is_imbalanced: bool
    severity: Optional[str] = None
    imbalance_ratio: Optional[float] = None
    majority_class: Optional[str] = None
    majority_pct: Optional[float] = None
    minority_class: Optional[str] = None
    minority_pct: Optional[float] = None
    minority_count: Optional[int] = None


class MLModelCard(BaseModel):
    rank: int
    name: str
    score: int
    suitability: str
    dots: int
    recommended: bool
    complexity: str
    training_speed: str
    interpretability: str
    strengths: list[str]
    weaknesses: list[str]
    needs_scaling: bool
    handles_missing: bool
    handles_categorical: bool
    reasons: list[str]
    reasoning: Optional[str] = None


class MLModelComparisonRow(BaseModel):
    rank: int
    model: str
    score: int
    suitability: str
    complexity: str
    training_speed: str
    interpretability: str
    needs_scaling: str
    handles_missing: str
    handles_categorical: str


class MLPreprocessingStep(BaseModel):
    technique: str
    priority: str
    applies_to: list[str]
    trigger: str
    reason: str


class MLFeatureEngineeringStep(BaseModel):
    technique: str
    applies_to: list[str]
    trigger: str
    reason: str


class MLImbalanceTechnique(BaseModel):
    technique: str
    priority: str
    trigger: str
    reason: str


class MLMetrics(BaseModel):
    primary: str
    secondary: list[str]
    avoid: list[str]
    reason: str


class MLArchitectureStep(BaseModel):
    step: int
    name: str
    description: str


class MLRoadmapStep(BaseModel):
    step_number: int
    title: str
    description: str


class MLStrategy(BaseModel):
    valid: bool
    objective: str
    problem: MLProblemInfo
    target_analysis: dict[str, Any]
    feature_columns: list[str]
    excluded_columns: dict[str, list[str]]
    column_roles: dict[str, list[str]]
    readiness: MLReadiness
    leakage: MLLeakage
    multicollinearity: MLMulticollinearity
    imbalance: MLImbalance
    models: list[MLModelCard]
    model_comparison: list[MLModelComparisonRow]
    preprocessing: list[MLPreprocessingStep]
    feature_engineering: list[MLFeatureEngineeringStep]
    imbalance_handling: list[MLImbalanceTechnique]
    split_strategy: dict[str, Any]
    cross_validation: dict[str, Any]
    metrics: MLMetrics
    architecture: list[MLArchitectureStep]
    roadmap: list[MLRoadmapStep]
    summary: str
    reasoning_source: str


class MLRecommendResponse(BaseModel):
    session_id: str
    strategy: MLStrategy


class MLSummaryResponse(BaseModel):
    session_id: str
    summary: dict[str, Any]


class MLModelsResponse(BaseModel):
    session_id: str
    problem_type: str
    models: list[MLModelCard]
    model_comparison: list[MLModelComparisonRow]


class MLRoadmapResponse(BaseModel):
    session_id: str
    roadmap: list[MLRoadmapStep]
    architecture: list[MLArchitectureStep]




#  Report / Stage 9  

class ReportGenerateRequest(BaseModel):
    sections: Optional[list[str]] = None   # omit for all 10 sections
    format: str = "html"                   # "pdf" | "html"


class ReportSectionAvailability(BaseModel):
    key: str
    title: str
    completed: bool
    detail: str


class ReportFormatSupport(BaseModel):
    html: bool
    pdf: bool


class ReportSectionsResponse(BaseModel):
    session_id: str
    sections: list[ReportSectionAvailability]
    formats: ReportFormatSupport


class ReportIncludedSection(BaseModel):
    key: str
    title: str
    completed: bool


class ReportGenerateResponse(BaseModel):
    session_id: str
    status: str
    format: str
    generated_at: str
    file_size_kb: float
    sections_included: list[ReportIncludedSection]
    sections_not_performed: list[str]
    download_url: str
    preview_url: str


class ReportSection(BaseModel):
    """
    Internal shape of one assembled section (rendered by the template)
    """
    key: str
    title: str
    number: Optional[int] = None
    performed: bool
    note: Optional[str] = None
    data: dict[str, Any] = {}


class ReportDocument(BaseModel):
    """
    Internal shape of the assembled report (rendered by the template)
    """
    title: str
    generated_at: str
    dataset: dict[str, int]
    sections: list[ReportSection]
    not_performed: list[str]


class ReportHistoryExport(BaseModel):
    session_id: str
    exported_at: str
    operations: list[dict[str, Any]]
    eda_analyses: list[dict[str, Any]]
    statistical_tests: list[dict[str, Any]]
    nlp_analyses: list[dict[str, Any]]












