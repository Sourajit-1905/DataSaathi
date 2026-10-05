"""
Session Service
---------------
Manages the lifecycle of a DataSaathi analysis session.

Each session is identified by a UUID and maps to:
  uploads/original/{session_id}.csv  — sacred, written once, never mutated
  uploads/working/{session_id}.csv   — active working copy (all ops applied here)
  uploads/history/{session_id}.json  — append-only operation log
  uploads/snapshots/{session_id}_snap.csv — single-slot undo snapshot
"""

import uuid
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.config import settings
from core.exceptions import SessionNotFoundError, SnapshotNotFoundError

logger = logging.getLogger(__name__)


# Path helpers 

def _original_path(session_id: str) -> Path:
    return settings.original_path / f"{session_id}.csv"


def _working_path(session_id: str) -> Path:
    return settings.working_path / f"{session_id}.csv"


def _history_path(session_id: str) -> Path:
    return settings.history_path / f"{session_id}.json"


def _snapshot_path(session_id: str) -> Path:
    return settings.snapshots_path / f"{session_id}_snap.csv"


def _ensure_dirs():
    """Ensure all upload subdirectories exist."""
    for path in [
        settings.original_path,
        settings.working_path,
        settings.history_path,
        settings.snapshots_path,
        settings.reports_path,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# Public API 

def session_exists(session_id: str) -> bool:
    """Return True if the working file exists for this session."""
    return _working_path(session_id).exists()


def create_session(df: pd.DataFrame, original_filename: str) -> dict:
    """
    Create a new session from an already-parsed DataFrame.

    Steps:
      1. Generate a UUID session_id
      2. Save df → uploads/original/{session_id}.csv  (sacred)
      3. Copy   → uploads/working/{session_id}.csv
      4. Write empty [] → uploads/history/{session_id}.json

    Returns a dict with session metadata.
    """
    _ensure_dirs()

    session_id = str(uuid.uuid4())

    # Save original (never touched again)
    df.to_csv(_original_path(session_id), index=False)

    # Copy to working
    shutil.copy2(_original_path(session_id), _working_path(session_id))

    # Init empty history
    _history_path(session_id).write_text(json.dumps([], indent=2), encoding="utf-8")

    logger.info(
        f"Session created | session_id={session_id} | "
        f"file={original_filename} | rows={len(df)} | cols={len(df.columns)}"
    )

    return {
        "session_id": session_id,
        "filename": original_filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "file_size_kb": round(_original_path(session_id).stat().st_size / 1024, 2),
        "upload_time": datetime.now(timezone.utc).isoformat(),
    }


def get_working_df(session_id: str) -> pd.DataFrame:
    """Load and return the current working DataFrame."""
    if not session_exists(session_id):
        raise SessionNotFoundError(session_id)
    return pd.read_csv(_working_path(session_id), low_memory=False)


def save_working_df(session_id: str, df: pd.DataFrame) -> None:
    """Overwrite the working CSV with the given DataFrame."""
    if not session_exists(session_id):
        raise SessionNotFoundError(session_id)
    df.to_csv(_working_path(session_id), index=False)
    logger.debug(f"Working df saved | session_id={session_id} | rows={len(df)}")


def get_original_df(session_id: str) -> pd.DataFrame:
    """Load and return the original (sacred) DataFrame."""
    orig = _original_path(session_id)
    if not orig.exists():
        raise SessionNotFoundError(session_id)
    return pd.read_csv(orig, low_memory=False)


def reset_to_original(session_id: str) -> None:
    """
    Reset working file to original and clear history.
    Snapshot is also cleared.
    """
    if not session_exists(session_id):
        raise SessionNotFoundError(session_id)

    shutil.copy2(_original_path(session_id), _working_path(session_id))
    _history_path(session_id).write_text(json.dumps([], indent=2), encoding="utf-8")

    # Remove snapshot if it exists
    snap = _snapshot_path(session_id)
    if snap.exists():
        snap.unlink()

    logger.info(f"Session reset to original | session_id={session_id}")


def log_operation(session_id: str, operation: dict) -> None:
    """Append an operation entry to the session history JSON."""
    if not session_exists(session_id):
        raise SessionNotFoundError(session_id)

    history = get_history(session_id)
    operation["timestamp"] = datetime.now(timezone.utc).isoformat()
    history.append(operation)
    _history_path(session_id).write_text(json.dumps(history, indent=2), encoding="utf-8")


def get_history(session_id: str) -> list:
    """Read and return the operation history list."""
    hist_file = _history_path(session_id)
    if not hist_file.exists():
        return []
    try:
        return json.loads(hist_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"History file corrupted for session_id={session_id}, resetting.")
        return []


def take_snapshot(session_id: str) -> None:
    """Copy current working file to the snapshot slot (enables single-level undo)."""
    if not session_exists(session_id):
        raise SessionNotFoundError(session_id)
    shutil.copy2(_working_path(session_id), _snapshot_path(session_id))
    logger.debug(f"Snapshot taken | session_id={session_id}")


def restore_snapshot(session_id: str) -> None:
    """Restore the working file from the snapshot slot."""
    snap = _snapshot_path(session_id)
    if not snap.exists():
        raise SnapshotNotFoundError(session_id)

    shutil.copy2(snap, _working_path(session_id))

    # Remove the last history entry (the operation that was undone)
    history = get_history(session_id)
    if history:
        history.pop()
        _history_path(session_id).write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info(f"Snapshot restored (undo) | session_id={session_id}")


def get_working_path(session_id: str) -> Path:
    """Return the path to the working CSV file (for streaming downloads)."""
    if not session_exists(session_id):
        raise SessionNotFoundError(session_id)
    return _working_path(session_id)
