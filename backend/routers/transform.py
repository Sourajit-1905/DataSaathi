"""
Transform Router
----------------
Endpoints:
  GET  /api/transform/{session_id}/recommend  — recommend transform for a column
  POST /api/transform/{session_id}/preview    — preview before/after stats without saving
  POST /api/transform/{session_id}/apply      — apply transform to working CSV
  POST /api/transform/{session_id}/undo       — restore snapshot (single undo)
  GET  /api/transform/{session_id}/history    — return transform operation history
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from core.cache import invalidate_profile
from core.exceptions import ColumnNotFoundError, InvalidFormulaError, SnapshotNotFoundError
from core.security import safe_detail, validate_session_id
from services import session_service
from services import transform_service

router = APIRouter(prefix="/api/transform", tags=["transform"])
logger = logging.getLogger(__name__)


# Request Bodies 

class RecommendRequest(BaseModel):
    column: str


class PreviewRequest(BaseModel):
    transform_type: str
    column: str
    options: Optional[dict[str, Any]] = {}


class ApplyRequest(BaseModel):
    transform_type: str
    column: str
    options: Optional[dict[str, Any]] = {}


# Helpers 

def _check_session(session_id: str):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    if not session_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


VALID_TRANSFORMS = {
    "minmax", "zscore", "robust", "log", "power",
    "label_encode", "onehot", "ordinal", "frequency", "target_encode",
    "equal_width", "equal_freq", "custom_bins",
    "polynomial", "interaction", "formula", "lag", "rolling_mean",
    "extract_year", "extract_month", "extract_day", "extract_dow", "extract_hour",
    "pca", "exclude_ml",
}

def _payload(transform_type: str, column: str, options: Optional[dict]) -> dict:
    """Options first, so a client cannot override transform_type or column through them."""
    return {**(options or {}), "transform_type": transform_type, "column": column}

# GET /api/transform/{session_id}/recommend 

@router.get("/{session_id}/recommend")
async def recommend(session_id: str, column: str):
    """
    Recommend the best transform for a column based on its stats.
    Gemini explains the reasoning. Falls back to rule-based.
    """
    _check_session(session_id)

    df = session_service.get_working_df(session_id)

    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found.")

    try:
        result = await transform_service.recommend_transform(df, column)
    except Exception as e:
        logger.exception(f"Recommend failed for column={column}: {e}")
        raise HTTPException(status_code=500, detail="Recommendation failed.")

    return {"session_id": session_id, "recommendation": result}


# POST /api/transform/{session_id}/preview 

@router.post("/{session_id}/preview")
def preview(session_id: str, body: PreviewRequest):
    """
    Return before/after stats for a transform without saving.
    Working CSV is never touched.
    """
    _check_session(session_id)

    if body.transform_type not in VALID_TRANSFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown transform_type '{body.transform_type}'.",
        )

    df = session_service.get_working_df(session_id)

    if body.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{body.column}' not found.")

    try:
        if body.transform_type == "minmax":
            stats = transform_service.preview_minmax(df, body.column)
        elif body.transform_type == "zscore":
            stats = transform_service.preview_zscore(df, body.column)
        elif body.transform_type == "log":
            stats = transform_service.preview_log(df, body.column)
        else:
            # For transforms without a dedicated preview,
            # apply in memory and compute stats — do NOT save
            payload = _payload(body.transform_type, body.column, body.options)
            df_after = transform_service.apply_transform(df, payload)
            # Find what changed
            if body.column in df_after.columns:
                stats = transform_service._before_after_stats(
                    df[body.column], df_after[body.column]
                )
            else:
                new_cols = [c for c in df_after.columns if c not in df.columns]
                stats = {
                    "note": f"Transform adds new columns: {new_cols}",
                    "new_columns": new_cols,
                    "rows_before": len(df),
                    "rows_after": len(df_after),
                    "cols_before": len(df.columns),
                    "cols_after": len(df_after.columns),
                }
    except (ColumnNotFoundError, InvalidFormulaError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"The transform could not be previewed: {safe_detail(e)}")
    except Exception as e:
        logger.exception(f"Preview failed: {e}")
        raise HTTPException(status_code=500, detail="Preview failed.")

    return {"session_id": session_id, "preview": stats}


#  POST /api/transform/{session_id}/apply 
@router.post("/{session_id}/apply")
def apply(session_id: str, body: ApplyRequest):
    """
    Apply a transformation to the working dataset.

    The session is held from read to commit, so two quick clicks cannot lose an
    update. Nothing is saved, snapshotted or logged unless the transform
    succeeds, so a failed apply leaves data, undo snapshot and history as they were.
    """
    _check_session(session_id)

    if body.transform_type not in VALID_TRANSFORMS:
        raise HTTPException(status_code=400, detail=f"Unknown transform_type '{body.transform_type}'.")

    with session_service.session_lock(session_id):
        df = session_service.get_working_df(session_id)

        if body.column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{body.column}' not found.")

        rows_before, cols_before = len(df), len(df.columns)

        # Flagging a column for exclusion changes no data, but is still recorded and undoable.
        if body.transform_type == "exclude_ml":
            result = transform_service.flag_exclude_ml(body.column)
            session_service.commit_change(session_id, None, {
                "stage": "transform",
                "transform_type": body.transform_type,
                "column": body.column,
                "options": body.options,
                "rows_before": rows_before,
                "rows_after": rows_before,
                "cols_before": cols_before,
                "cols_after": cols_before,
                "new_columns": [],
            })
            logger.info(
                "Transform applied | session_id=%s | type=exclude_ml | col=%s | rows %d→%d | cols %d→%d",
                session_id, body.column, rows_before, rows_before, cols_before, cols_before,
            )
            return {
                "session_id": session_id,
                "status": "applied",
                "transform_type": body.transform_type,
                "column": body.column,
                "rows_before": rows_before,
                "rows_after": rows_before,
                "cols_before": cols_before,
                "cols_after": cols_before,
                "new_columns": [],
                "note": result["note"],
            }

        try:
            df_new = transform_service.apply_transform(
                df, _payload(body.transform_type, body.column, body.options)
            )
        except InvalidFormulaError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ColumnNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"The transform could not be applied: {safe_detail(e)}",
            )
        except Exception:
            logger.exception(
                "Transform apply failed | session_id=%s | type=%s | col=%s",
                session_id, body.transform_type, body.column,
            )
            raise HTTPException(status_code=500, detail="Transform failed.")

        new_columns = [c for c in df_new.columns if c not in df.columns]

        session_service.commit_change(session_id, df_new, {
            "stage": "transform",
            "transform_type": body.transform_type,
            "column": body.column,
            "options": body.options,
            "rows_before": rows_before,
            "rows_after": len(df_new),
            "cols_before": cols_before,
            "cols_after": len(df_new.columns),
            "new_columns": new_columns,
        })

    invalidate_profile(session_id)

    logger.info(
        "Transform applied | session_id=%s | type=%s | col=%s | rows %d→%d | cols %d→%d",
        session_id, body.transform_type, body.column,
        rows_before, len(df_new), cols_before, len(df_new.columns),
    )

    return {
        "session_id": session_id,
        "status": "applied",
        "transform_type": body.transform_type,
        "column": body.column,
        "rows_before": rows_before,
        "rows_after": len(df_new),
        "cols_before": cols_before,
        "cols_after": len(df_new.columns),
        "new_columns": new_columns,
    }


@router.post("/{session_id}/undo")
def undo(session_id: str):
    """
    Undo the last change. Single level: the snapshot is consumed, so a second
    undo returns 404 until another change is applied.
    """
    _check_session(session_id)

    try:
        session_service.restore_snapshot(session_id)
    except SnapshotNotFoundError:
        raise HTTPException(status_code=404, detail="There is nothing to undo.")

    invalidate_profile(session_id)
    logger.info("Transform undo applied | session_id=%s", session_id)

    return {
        "session_id": session_id,
        "status": "undone",
        "message": "The last change has been reversed.",
    }


# GET /api/transform/{session_id}/history 

@router.get("/{session_id}/history")
def get_history(session_id: str):
    """
    Return the transform operation history for this session.
    Frontend uses this to populate the History Panel.
    """
    _check_session(session_id)
    history = session_service.get_history(session_id)
    transform_history = [op for op in history if op.get("stage") == "transform"]
    return {
        "session_id": session_id,
        "history": transform_history,
        "total_operations": len(transform_history),
    }