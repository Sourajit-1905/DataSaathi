"""
In-memory cache dictionaries keyed by session_id.
All caches are module-level dicts — they persist for the lifetime of the process.
Cache invalidation rules:
  - _profile_cache: invalidated on any clean or transform apply
  - Others: accumulate per session (history-like)
"""

# Profiling results cache: { session_id: dict }
_profile_cache: dict[str, dict] = {}

# EDA history cache: { session_id: list[dict] }
_eda_history: dict[str, list] = {}

# Stats history cache: { session_id: list[dict] }
_stats_history: dict[str, list] = {}

# NLP history cache: { session_id: list[dict] }
_nlp_history: dict[str, list] = {}

# ML history cache: { session_id: dict }
_ml_history: dict[str, dict] = {}


# Profile cache helpers 

def set_profile(session_id: str, result: dict) -> None:
    _profile_cache[session_id] = result


def get_profile(session_id: str) -> dict | None:
    return _profile_cache.get(session_id)


def invalidate_profile(session_id: str) -> None:
    _profile_cache.pop(session_id, None)


# EDA history helpers 

def append_eda_history(session_id: str, entry: dict) -> None:
    _eda_history.setdefault(session_id, []).append(entry)


def get_eda_history(session_id: str) -> list:
    return _eda_history.get(session_id, [])


# Stats history helpers 

def append_stats_history(session_id: str, entry: dict) -> None:
    _stats_history.setdefault(session_id, []).append(entry)


def get_stats_history(session_id: str) -> list:
    return _stats_history.get(session_id, [])


# NLP history helpers 

def append_nlp_history(session_id: str, entry: dict) -> None:
    _nlp_history.setdefault(session_id, []).append(entry)


def get_nlp_history(session_id: str) -> list:
    return _nlp_history.get(session_id, [])


# ML history helpers 

def set_ml_result(session_id: str, result: dict) -> None:
    _ml_history[session_id] = result


def get_ml_result(session_id: str) -> dict | None:
    return _ml_history.get(session_id)

def clear_session(session_id: str) -> None:
    """Forget everything cached for a session: profile, EDA, stats, NLP and ML results."""
    _profile_cache.pop(session_id, None)
    _eda_history.pop(session_id, None)
    _stats_history.pop(session_id, None)
    _nlp_history.pop(session_id, None)
    _ml_history.pop(session_id, None)

