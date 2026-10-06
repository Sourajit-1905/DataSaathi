import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from core.security import validate_session_id
from core.cache import invalidate_profile
from core.exceptions import ColumnNotFoundError
from services import session_service
from services import clean_service

router = APIRouter(prefix="/api/clean", tags=["clean"])
logger = logging.getLogger(__name__)


# Request Bodies 

class RecommendRequest(BaseModel):
    issue_type: str
    column: Optional[str] = None


class PreviewRequest(BaseModel):
    issue_type: str
    column: Optional[str] = None
    strategy: str
    fill_value: Optional[float] = 0.0
    fill_value_str: Optional[str] = "Unknown"


class ApplyRequest(BaseModel):
    issue_type: str
    column: Optional[str] = None
    strategy: str
    fill_value: Optional[float] = 0.0
    fill_value_str: Optional[str] = "Unknown"


# Helpers 

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


VALID_ISSUE_TYPES = {
    "missing_numerical",
    "missing_categorical",
    "duplicates",
    "outliers",
    "type_conversion",
    "string_standardisation",
    "constant_columns",
    "identifier_columns",
    "negative_values",
}

VALID_STRATEGIES = {
    "missing_numerical":     ["mean", "median", "constant", "drop_rows", "drop_column"],
    "missing_categorical":   ["mode", "constant", "drop_rows", "drop_column"],
    "duplicates":            ["drop_duplicates", "none"],
    "outliers":              ["iqr_cap", "zscore_cap", "drop_rows", "keep"],
    "type_conversion":       ["to_numeric", "to_datetime"],
    "string_standardisation":["trim_whitespace", "lowercase", "titlecase"],
    "constant_columns":      ["drop_column"],
    "identifier_columns":    ["drop_column"],
    "negative_values":       ["keep", "set_zero", "set_nan", "drop_rows", "abs_value", "flag_only"],
}


def _validate_issue_and_strategy(issue_type: str, strategy: str):
    if issue_type not in VALID_ISSUE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown issue_type '{issue_type}'. Valid types: {list(VALID_ISSUE_TYPES)}",
        )
    if strategy not in VALID_STRATEGIES.get(issue_type, []):
        raise HTTPException(
            status_code=400,
            detail=f"Strategy '{strategy}' is not valid for issue_type '{issue_type}'. "
                   f"Valid strategies: {VALID_STRATEGIES[issue_type]}",
        )


# GET /api/clean/{session_id}/issues 

@router.get("/{session_id}/issues")
def get_issues(session_id: str):
    """
    Run all 9 issue detectors on the current working dataset.
    Negative value results filtered to only show non_negative
    and ambiguous columns — signed and encoded are not issues.
    Frontend uses this to populate the Issue List panel.
    """
    _check_session(session_id)
    df = session_service.get_working_df(session_id)
    issues = clean_service.get_all_issues(df)

    # Filter negative_values to only actionable issues
    issues["negative_values"] = [
        item for item in issues["negative_values"]
        if item.get("semantic_class") in ("non_negative", "ambiguous")
    ]

    total_issues = sum(
        len(v) if isinstance(v, list)
        else (1 if v.get("duplicate_count", 0) > 0 else 0)
        for v in issues.values()
    )
    return {
        "session_id": session_id,
        "issues": issues,
        "total_issues": total_issues,
    }


# POST /api/clean/{session_id}/recommend 

@router.post("/{session_id}/recommend")
async def recommend(session_id: str, body: RecommendRequest):
    """
    Return a recommended strategy + Gemini explanation for one issue type.
    Frontend uses this to populate the Issue Detail Panel.
    """
    _check_session(session_id)

    if body.issue_type not in VALID_ISSUE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown issue_type '{body.issue_type}'.",
        )

    df = session_service.get_working_df(session_id)

    try:
        if body.issue_type == "missing_numerical":
            result = await clean_service.recommend_missing_numerical(df, body.column)
        elif body.issue_type == "missing_categorical":
            result = await clean_service.recommend_missing_categorical(df, body.column)
        elif body.issue_type == "duplicates":
            result = await clean_service.recommend_duplicates(df)
        elif body.issue_type == "outliers":
            result = await clean_service.recommend_outliers(df, body.column)
        elif body.issue_type == "negative_values":
            result = await clean_service.recommend_negative_values(df, body.column)
        else:
            result = {
                "issue_type": body.issue_type,
                "column": body.column,
                "recommended_strategy": VALID_STRATEGIES[body.issue_type][0],
                "available_strategies": VALID_STRATEGIES[body.issue_type],
                "explanation": f"Apply '{VALID_STRATEGIES[body.issue_type][0]}' to fix this issue.",
            }
    except ColumnNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Recommend failed: {e}")
        raise HTTPException(status_code=500, detail="Recommendation failed.")

    return {"session_id": session_id, "recommendation": result}


# POST /api/clean/{session_id}/preview 

@router.post("/{session_id}/preview")
def preview(session_id: str, body: PreviewRequest):
    """
    Preview what the dataset will look like after applying a strategy.
    Does NOT save anything — working CSV is untouched.
    """
    _check_session(session_id)
    _validate_issue_and_strategy(body.issue_type, body.strategy)

    df = session_service.get_working_df(session_id)

    try:
        if body.issue_type == "missing_numerical":
            result = clean_service.preview_missing_numerical(
                df, body.column, body.strategy, body.fill_value
            )
        else:
            result = {
                "note": f"Preview not available for '{body.issue_type}'. Apply to see changes.",
                "strategy": body.strategy,
                "column": body.column,
            }
    except ColumnNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"session_id": session_id, "preview": result}


# POST /api/clean/{session_id}/apply 

@router.post("/{session_id}/apply")
def apply(session_id: str, body: ApplyRequest):
    """
    Apply a cleaning strategy to the working dataset.
    Snapshot taken before apply — enables single undo.
    Profile cache invalidated after every apply.
    """
    _check_session(session_id)
    _validate_issue_and_strategy(body.issue_type, body.strategy)

    df = session_service.get_working_df(session_id)
    rows_before = len(df)
    cols_before = len(df.columns)

    # Take snapshot before mutating (enables undo)
    session_service.take_snapshot(session_id)

    try:
        if body.issue_type == "missing_numerical":
            df = clean_service.apply_missing_numerical(
                df, body.column, body.strategy, body.fill_value
            )
        elif body.issue_type == "missing_categorical":
            df = clean_service.apply_missing_categorical(
                df, body.column, body.strategy, body.fill_value_str
            )
        elif body.issue_type == "duplicates":
            df = clean_service.apply_duplicates(df)
        elif body.issue_type == "outliers":
            df = clean_service.apply_outliers(df, body.column, body.strategy)
        elif body.issue_type == "type_conversion":
            df = clean_service.apply_type_conversion(df, body.column, body.strategy)
        elif body.issue_type == "string_standardisation":
            df = clean_service.apply_string_standardisation(df, body.column, body.strategy)
        elif body.issue_type in ("constant_columns", "identifier_columns"):
            df = clean_service.apply_drop_column(df, body.column)
        elif body.issue_type == "negative_values":
            df = clean_service.apply_negative_values(df, body.column, body.strategy)

    except ColumnNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Apply failed for {body.issue_type}: {e}")
        raise HTTPException(status_code=500, detail="Apply operation failed.")

    # Save working df
    session_service.save_working_df(session_id, df)

    # Invalidate profile cache — data has changed
    invalidate_profile(session_id)

    # Log operation
    session_service.log_operation(session_id, {
        "stage": "clean",
        "issue_type": body.issue_type,
        "column": body.column,
        "strategy": body.strategy,
        "rows_before": rows_before,
        "rows_after": len(df),
        "cols_before": cols_before,
        "cols_after": len(df.columns),
    })

    logger.info(
        f"Clean applied | session_id={session_id} | "
        f"issue={body.issue_type} | strategy={body.strategy} | "
        f"rows {rows_before}→{len(df)} | cols {cols_before}→{len(df.columns)}"
    )

    # Warn if set_nan created new missing values
    warning = None
    if body.strategy == "set_nan" and body.column:
        new_nulls = int(df[body.column].isna().sum()) if body.column in df.columns else 0
        if new_nulls > 0:
            warning = (
                f"'{body.column}' now has {new_nulls} missing values created by set_nan. "
                f"Consider re-running missing value cleaning for this column."
            )

    # Warn if flag_only added a new column
    if body.strategy == "flag_only" and body.column:
        flag_col = f"{body.column}_negative_flag"
        if flag_col in df.columns and not warning:
            warning = (
                f"A new column '{flag_col}' was added to flag negative values. "
                f"Exclude it from ML features if not needed."
            )

    return {
        "session_id": session_id,
        "status": "applied",
        "issue_type": body.issue_type,
        "strategy": body.strategy,
        "column": body.column,
        "rows_before": rows_before,
        "rows_after": len(df),
        "cols_before": cols_before,
        "cols_after": len(df.columns),
        "warning": warning,
    }


# POST /api/clean/{session_id}/undo 

@router.post("/{session_id}/undo")
def undo(session_id: str):
    """
    Restore the working dataset from the last snapshot.
    Single-level undo — only the most recent operation can be undone.
    Profile cache invalidated after undo.
    """
    _check_session(session_id)

    try:
        session_service.restore_snapshot(session_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    invalidate_profile(session_id)

    logger.info(f"Undo applied | session_id={session_id}")

    return {
        "session_id": session_id,
        "status": "undone",
        "message": "Last cleaning operation has been reversed.",
    }


# GET /api/clean/{session_id}/history 

@router.get("/{session_id}/history")
def get_history(session_id: str):
    """
    Return the full operation history for this session.
    Frontend uses this to populate the History Panel.
    """
    _check_session(session_id)
    history = session_service.get_history(session_id)
    clean_history = [op for op in history if op.get("stage") == "clean"]
    return {
        "session_id": session_id,
        "history": clean_history,
        "total_operations": len(clean_history),
    }