"""
Session Service
---------------
Manages the lifecycle of a DataSaathi analysis session.

Each session is identified by a UUID and maps to:
  original/{session_id}.csv         sacred, written once, never mutated
  working/{session_id}.csv          active working copy (all operations apply here)
  history/{session_id}.json         append-only operation log
  snapshots/{session_id}_snap.csv   single-slot undo snapshot
  reports/{session_id}_report.*     generated reports

Guarantees
  - Atomic writes. Files are written to a temporary name and swapped in, so a
    crash never leaves a half-written CSV or history file.
  - Apply and undo stay in step. commit_change() writes the new data aside,
    snapshots the current data, swaps, and logs, in that order, and only after the
    operation has already succeeded. A failed operation never touches the
    snapshot, and undo consumes the snapshot, so a second undo cannot pop history
    without restoring data.
  - One writer per session. session_lock() lets a router hold the session across
    read, compute and commit, so a double-clicked Apply cannot lose an update.
  - Housekeeping. Idle sessions are deleted by cleanup_expired_sessions().
"""
import asyncio
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import cache
from core.config import settings
from core.exceptions import SessionNotFoundError, SnapshotNotFoundError
from core.security import validate_session_id

logger = logging.getLogger(__name__)

STALE_TEMP_FILE_SECONDS = 3600

# Locks and activity tracking

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()
_last_access: dict[str, float] = {}


def _check_id(session_id) -> str:
    """Every path is built from a validated UUID, so a bad ID can never become a path."""
    if not validate_session_id(session_id):
        raise SessionNotFoundError(str(session_id)[:64])
    return session_id


def _lock_for(session_id: str) -> threading.RLock:
    with _locks_guard:
        lock = _locks.get(session_id)
        if lock is None:
            lock = _locks[session_id] = threading.RLock()
        return lock


@contextmanager
def session_lock(session_id: str):
    """
    Hold exclusive access to one session. Re-entrant, so a router can wrap
    read, compute and commit while commit_change() takes the lock again inside.
    """
    with _lock_for(_check_id(session_id)):
        yield


def _touch(session_id: str) -> None:
    _last_access[session_id] = time.time()


# Paths

def _original_path(session_id: str) -> Path:
    return settings.original_path / f"{_check_id(session_id)}.csv"


def _working_path(session_id: str) -> Path:
    return settings.working_path / f"{_check_id(session_id)}.csv"


def _history_path(session_id: str) -> Path:
    return settings.history_path / f"{_check_id(session_id)}.json"


def _snapshot_path(session_id: str) -> Path:
    return settings.snapshots_path / f"{_check_id(session_id)}_snap.csv"


def _report_paths(session_id: str) -> list[Path]:
    sid = _check_id(session_id)
    return [settings.reports_path / f"{sid}_report.html", settings.reports_path / f"{sid}_report.pdf"]

def _chat_path(session_id: str) -> Path:
    return settings.chat_path / f"{_check_id(session_id)}_chat.json"


def _session_dirs() -> list[Path]:
    return [
        settings.original_path, settings.working_path, settings.history_path,
        settings.snapshots_path, settings.reports_path, settings.chat_path,
    ]

def _all_session_files(session_id: str) -> list[Path]:
    return [
        _original_path(session_id), _working_path(session_id), _history_path(session_id),
        _snapshot_path(session_id), *_report_paths(session_id), _chat_path(session_id),
    ]

def _ensure_dirs():
    for path in _session_dirs():
        path.mkdir(parents=True, exist_ok=True)


# Atomic file helpers

def _tmp_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = _tmp_for(path)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_text(text: str, path: Path) -> None:
    tmp = _tmp_for(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = _tmp_for(dst)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _unlink(path: Path) -> bool:
    """Delete a file. On Windows a file that is open elsewhere cannot be deleted; the next cleanup retries."""
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError as e:
        logger.warning("Could not delete %s: %s", path.name, e)
        return False


# Public API

def session_exists(session_id: str) -> bool:
    """True if the session's working file exists. Counts as activity for expiry."""
    if not validate_session_id(session_id):
        return False
    exists = _working_path(session_id).exists()
    if exists:
        _touch(session_id)
    return exists


def create_session(df: pd.DataFrame, original_filename: str) -> dict:
    """
    Create a new session from an already-parsed DataFrame:
    save the original (never touched again), copy it to working, start an empty history.
    """
    _ensure_dirs()
    session_id = str(uuid.uuid4())

    with session_lock(session_id):
        _atomic_write_csv(df, _original_path(session_id))
        _copy_atomic(_original_path(session_id), _working_path(session_id))
        _atomic_write_text(json.dumps([], indent=2), _history_path(session_id))

    _touch(session_id)
    logger.info(
        "Session created | session_id=%s | file=%s | rows=%d | cols=%d",
        session_id, original_filename, len(df), len(df.columns),
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
    """Load the current working DataFrame."""
    with session_lock(session_id):
        path = _working_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        return pd.read_csv(path, low_memory=False)


async def aget_working_df(session_id: str) -> pd.DataFrame:
    """get_working_df for async endpoints: any wait for the session lock happens off the event loop."""
    return await asyncio.to_thread(get_working_df, session_id)

def save_working_df(session_id: str, df: pd.DataFrame) -> None:
    """Replace the working CSV. Prefer commit_change(), which also snapshots and logs."""
    with session_lock(session_id):
        if not _working_path(session_id).exists():
            raise SessionNotFoundError(session_id)
        _atomic_write_csv(df, _working_path(session_id))


def get_original_df(session_id: str) -> pd.DataFrame:
    """Load the original (sacred) DataFrame."""
    with session_lock(session_id):
        path = _original_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        return pd.read_csv(path, low_memory=False)


def get_working_path(session_id: str) -> Path:
    """Path of the working CSV (for streaming downloads)."""
    if not session_exists(session_id):
        raise SessionNotFoundError(session_id)
    return _working_path(session_id)


#   History  

def get_history(session_id: str) -> list:
    """The operation history list."""
    hist_file = _history_path(session_id)
    if not hist_file.exists():
        return []
    try:
        return json.loads(hist_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("History file corrupted for session_id=%s, resetting.", session_id)
        return []


def log_operation(session_id: str, operation: dict) -> None:
    """Append an operation entry to the session history."""
    with session_lock(session_id):
        if not _working_path(session_id).exists():
            raise SessionNotFoundError(session_id)
        history = get_history(session_id)
        operation["timestamp"] = datetime.now(timezone.utc).isoformat()
        history.append(operation)
        _atomic_write_text(json.dumps(history, indent=2), _history_path(session_id))


#   Apply and undo  

def take_snapshot(session_id: str) -> None:
    """Copy the current working file to the snapshot slot. Prefer commit_change()."""
    with session_lock(session_id):
        if not _working_path(session_id).exists():
            raise SessionNotFoundError(session_id)
        _copy_atomic(_working_path(session_id), _snapshot_path(session_id))


def commit_change(session_id: str, df: pd.DataFrame | None, operation: dict) -> None:
    """
    Make one operation permanent. Call it only after the operation has succeeded.

    Order matters: (1) write the new data aside, (2) snapshot the current data,
    (3) swap the new data in, (4) log. If (1) or (2) fails, nothing has changed.
    df=None records an operation that changes no data (for example a flag).
    """
    with session_lock(session_id):
        working = _working_path(session_id)
        if not working.exists():
            raise SessionNotFoundError(session_id)

        new_tmp = None
        try:
            if df is not None:
                new_tmp = _tmp_for(working)
                df.to_csv(new_tmp, index=False)
            _copy_atomic(working, _snapshot_path(session_id))
            if new_tmp is not None:
                os.replace(new_tmp, working)
                new_tmp = None
            log_operation(session_id, operation)
        finally:
            if new_tmp is not None:
                new_tmp.unlink(missing_ok=True)


def restore_snapshot(session_id: str) -> None:
    """
    Undo the last committed change: restore the snapshot, drop the last history
    entry, and consume the snapshot (undo is single-level).
    """
    with session_lock(session_id):
        if not _working_path(session_id).exists():
            raise SessionNotFoundError(session_id)
        snap = _snapshot_path(session_id)
        if not snap.exists():
            raise SnapshotNotFoundError(session_id)

        _copy_atomic(snap, _working_path(session_id))

        history = get_history(session_id)
        if history:
            history.pop()
            _atomic_write_text(json.dumps(history, indent=2), _history_path(session_id))

        _unlink(snap)
        logger.info("Snapshot restored (undo) | session_id=%s", session_id)


def reset_to_original(session_id: str) -> None:
    """
    Back to the uploaded file: restore the original, clear history and snapshot,
    delete generated reports, and forget every cached result (profile, EDA, stats,
    NLP, ML), because they described data that no longer exists.
    """
    with session_lock(session_id):
        if not _original_path(session_id).exists():
            raise SessionNotFoundError(session_id)

        _copy_atomic(_original_path(session_id), _working_path(session_id))
        _atomic_write_text(json.dumps([], indent=2), _history_path(session_id))
        _unlink(_snapshot_path(session_id))
        for report in _report_paths(session_id):
            _unlink(report)
        _unlink(_chat_path(session_id))
        cache.clear_session(session_id)

    logger.info("Session reset to original | session_id=%s", session_id)


# Expiry
#   Chat (the AI assistant's conversation)  

def get_chat(session_id: str) -> list[dict]:
    """The stored conversation, oldest first. A missing or corrupt file reads as empty."""
    with session_lock(session_id):
        path = _chat_path(session_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Chat file corrupted for session_id=%s, ignoring it.", session_id)
            return []
        return data if isinstance(data, list) else []


def append_chat(session_id: str, messages: list[dict], max_messages: int = 200) -> list[dict]:
    """
    Add messages to the conversation and return the whole, trimmed thread. The read,
    append and write happen under the session lock, so two quick messages cannot
    overwrite each other. The oldest messages are dropped beyond max_messages.
    """
    with session_lock(session_id):
        if not _working_path(session_id).exists():
            raise SessionNotFoundError(session_id)
        thread = get_chat(session_id) + list(messages)
        if len(thread) > max_messages:
            thread = thread[-max_messages:]
        _atomic_write_text(json.dumps(thread, indent=2, default=str), _chat_path(session_id))
        return thread


def clear_chat(session_id: str) -> None:
    """Delete the stored conversation. Safe to call when there is none."""
    with session_lock(session_id):
        _unlink(_chat_path(session_id))


_SESSION_FILE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(?:_snap|_report|_chat)?$",
    re.IGNORECASE,
)


def get_last_activity(session_id: str) -> float | None:
    """Most recent of: any session file's modification time, and the last request seen."""
    times = [_last_access[session_id]] if session_id in _last_access else []
    for path in _all_session_files(session_id):
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def _session_ids_on_disk() -> set[str]:
    ids: set[str] = set()
    for directory in _session_dirs():
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            match = _SESSION_FILE.match(entry.stem)
            if match and entry.is_file():
                ids.add(match.group(1).lower())
    return ids


def delete_session(session_id: str) -> bool:
    """Delete every file and cached result of a session. Returns True if anything was removed."""
    if not validate_session_id(session_id):
        return False

    removed = False
    with session_lock(session_id):
        for path in _all_session_files(session_id):
            if path.exists() and _unlink(path):
                removed = True
        cache.clear_session(session_id)
        _last_access.pop(session_id, None)

    with _locks_guard:
        _locks.pop(session_id, None)
    return removed


def _remove_stale_temp_files() -> None:
    cutoff = time.time() - STALE_TEMP_FILE_SECONDS
    for directory in _session_dirs():
        if not directory.is_dir():
            continue
        for tmp in directory.glob("*.tmp"):
            try:
                if tmp.stat().st_mtime < cutoff:
                    _unlink(tmp)
            except OSError:
                continue


def cleanup_expired_sessions(max_age_hours: float) -> int:
    """
    Delete sessions idle for longer than max_age_hours. A session that is in use
    right now (its lock is held) is skipped. Returns how many were deleted.
    max_age_hours <= 0 disables cleanup.
    """
    if not max_age_hours or max_age_hours <= 0:
        return 0

    cutoff = time.time() - max_age_hours * 3600
    deleted = 0

    for session_id in _session_ids_on_disk():
        last = get_last_activity(session_id)
        if last is None or last >= cutoff:
            continue

        lock = _lock_for(session_id)
        if not lock.acquire(blocking=False):
            continue
        try:
            last = get_last_activity(session_id)   # re-check now that we hold the lock
            if last is not None and last < cutoff and delete_session(session_id):
                deleted += 1
        finally:
            lock.release()

    _remove_stale_temp_files()
    if deleted:
        logger.info("Session cleanup | deleted=%d | max_age_hours=%s", deleted, max_age_hours)
    return deleted