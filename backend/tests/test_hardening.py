"""
Stage 10 hardening tests.

  A  Error handling and middleware      self-contained toy app
  B  Request body limits                toy app and raw ASGI
  C  Logging redaction
  D  Gemini client                      the SDK call is faked; no key or network needed
  E  Session service                    isolated upload folder
  F  The event loop is never blocked    one loop, with controls that must fail
  G  Unit tests for the patched code
  H  Upload endpoint                    real dataset router
  I  Project integration                real routers: clean, transform, EDA, profile, report
  J  Gemini call sites                  the nine replaced helpers in your services

H, I and J use the real application (`from main import app`) with an isolated
upload folder, so they never touch your real sessions.

Event-loop tests use ONE loop (httpx.ASGITransport). Starlette's TestClient starts
a separate loop per request, so it cannot show that one request blocks another.
"""

import asyncio
import importlib
import inspect
import io
import logging
import os
import threading
import time
import uuid
import zipfile

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel

from core import cache
from core import gemini_client as gc
from core.config import settings
from core.exceptions import (
    InvalidFormulaError,
    SessionNotFoundError,
    SnapshotNotFoundError,
    register_exception_handlers,
)
from core.logging_config import RedactingFormatter, _resolve_level
from core.middleware import BodyLimitMiddleware, NaNSafeJSONResponse, RequestContextMiddleware
from services import session_service as ss

MB = 1024 * 1024
GOOGLE_KEY = "AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"
ORIGIN = {"Origin": "http://localhost:5173"}


# ═══════════════════════════════════════════════════════════════
# A. Error handling and middleware
# ═══════════════════════════════════════════════════════════════

class _Body(BaseModel):
    column: str
    top_n: int = 5


def build_app(upload_limit_mb: int = 1) -> FastAPI:
    """The real middleware and handlers around a few routes that misbehave on purpose."""
    app = FastAPI(default_response_class=NaNSafeJSONResponse)
    app.add_middleware(
        BodyLimitMiddleware,
        default_limit=2 * MB,
        upload_path="/upload",
        upload_limit=(upload_limit_mb + 1) * MB,
        upload_message=f"The file is too large. The maximum upload size is {upload_limit_mb} MB.",
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    register_exception_handlers(app)

    @app.get("/ok")
    def ok():
        return {"ok": True}

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/rid")
    def rid(request: Request):
        return {"rid": request.state.request_id}

    @app.get("/http/{code}")
    def http_error(code: int):
        raise HTTPException(status_code=code, detail="Session 'x' not found.")

    @app.get("/nan")
    def nan():
        return {"x": float("nan"), "y": [float("inf"), 1.5], "z": {"w": float("-inf")}, "s": "ok"}

    @app.get("/boom")
    def boom():
        raise RuntimeError(rf"kaboom at C:\Users\Biswajit\x.py with key={GOOGLE_KEY}")

    @app.post("/body")
    def body(b: _Body):
        return {"ok": True}

    @app.post("/upload")
    def upload(file: UploadFile = File(...)):
        return {"bytes": len(file.file.read())}

    @app.post("/small")
    async def small(request: Request):
        return {"bytes": len(await request.body())}

    return app


@pytest.fixture(scope="module")
def toy_app():
    return build_app()


@pytest.fixture(scope="module")
def tc(toy_app):
    return TestClient(toy_app, raise_server_exceptions=False)


def test_every_response_has_a_request_id_and_security_headers(tc):
    r = tc.get("/ok")
    assert r.status_code == 200
    assert len(r.headers["x-request-id"]) == 16
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"


def test_request_id_is_available_to_the_route(tc):
    r = tc.get("/rid")
    assert r.json()["rid"] == r.headers["x-request-id"]


def test_a_safe_incoming_request_id_is_honoured(tc):
    r = tc.get("/rid", headers={"X-Request-ID": "trace-abc-12345"})
    assert r.headers["x-request-id"] == "trace-abc-12345"
    assert r.json()["rid"] == "trace-abc-12345"


@pytest.mark.parametrize("unsafe", ["short", "x" * 100, "bad id with spaces", "<script>alert(1)</script>"])
def test_an_unsafe_incoming_request_id_is_replaced(tc, unsafe):
    r = tc.get("/rid", headers={"X-Request-ID": unsafe})
    assert r.headers["x-request-id"] != unsafe
    assert len(r.headers["x-request-id"]) == 16


def test_request_ids_are_unique(tc):
    assert tc.get("/ok").headers["x-request-id"] != tc.get("/ok").headers["x-request-id"]


@pytest.mark.parametrize("status", [400, 404, 500])
def test_router_errors_use_the_standard_shape(tc, status):
    r = tc.get(f"/http/{status}")
    body = r.json()
    assert r.status_code == status
    assert body["error"] == "Session 'x' not found."
    assert body["code"] == status
    assert body["request_id"] == r.headers["x-request-id"]


def test_unknown_route_and_wrong_method_use_the_standard_shape(tc):
    for r in (tc.get("/no/such/route"), tc.delete("/body")):
        assert {"error", "detail", "code", "request_id"} <= set(r.json())
    assert tc.delete("/body").status_code == 405
    assert "allow" in {k.lower() for k in tc.delete("/body").headers}


def test_validation_error_lists_fields_and_does_not_echo_input(tc):
    r = tc.post("/body", json={"top_n": "SECRET-VALUE-123"})
    body = r.json()
    assert r.status_code == 422
    assert "SECRET-VALUE-123" not in r.text
    assert {f["field"] for f in body["fields"]} >= {"column", "top_n"}
    assert body["error"].startswith("Invalid request.")


def test_malformed_json_body_gets_a_clear_message(tc):
    r = tc.post("/body", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "not valid JSON" in r.text


def test_at_most_five_validation_problems_are_listed(tc):
    r = tc.post("/body", json=[1, 2, 3])
    assert r.status_code == 422
    assert len(r.json()["fields"]) <= 5


def test_unhandled_error_is_generic_and_still_carries_cors_and_security_headers(tc):
    r = tc.get("/boom", headers=ORIGIN)
    body = r.json()
    assert r.status_code == 500
    assert body["error"] == "An unexpected error occurred. Please try again."
    assert body["detail"] == f"Reference: {body['request_id']}"
    # Without these the browser hides the response and the UI shows "Network error".
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-request-id"] == body["request_id"]
    for leak in ("kaboom", "Biswajit", "AIza", "Traceback", "RuntimeError"):
        assert leak not in r.text


def test_the_default_test_client_gets_a_500_instead_of_an_exception(toy_app):
    assert TestClient(toy_app).get("/boom").status_code == 500


def test_unhandled_error_is_logged_with_its_traceback(tc, caplog):
    with caplog.at_level(logging.ERROR):
        tc.get("/boom")
    records = [r for r in caplog.records if r.exc_info]
    assert any("kaboom" in str(r.exc_info[1]) for r in records)
    assert any("request_id=" in r.getMessage() for r in records)


def test_cors_allows_the_configured_origin_and_blocks_others(tc):
    ok = tc.get("/ok", headers=ORIGIN)
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "x-request-id" in ok.headers["access-control-expose-headers"].lower()
    other = tc.get("/ok", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers
    preflight = tc.options("/ok", headers={**ORIGIN, "Access-Control-Request-Method": "POST"})
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_nan_and_infinity_are_sent_as_null(tc):
    r = tc.get("/nan")
    assert r.status_code == 200
    assert r.json() == {"x": None, "y": [None, 1.5], "z": {"w": None}, "s": "ok"}


def test_one_log_line_per_request_and_no_log_forging(tc, caplog):
    with caplog.at_level(logging.DEBUG, logger="core.middleware"):
        tc.get("/api/health")
        tc.get("/http/404")
        tc.get("/boom")
        tc.get("/x%0Aforged%20line")
    lines = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "core.middleware"]
    assert len(lines) == 4
    assert lines[0][0] == logging.DEBUG and "/api/health -> 200" in lines[0][1]
    assert lines[1][0] == logging.INFO and "/http/404 -> 404" in lines[1][1]
    assert lines[2][0] == logging.WARNING and "/boom -> 500" in lines[2][1]
    assert "ms | request_id=" in lines[1][1]
    assert all("\n" not in message for _, message in lines)


# ═══════════════════════════════════════════════════════════════
# B. Request body limits
# ═══════════════════════════════════════════════════════════════

def test_small_upload_is_accepted(tc):
    r = tc.post("/upload", files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 200 and r.json()["bytes"] == 8


def test_oversized_upload_is_rejected_with_the_standard_shape(tc):
    r = tc.post("/upload", files={"file": ("big.csv", b"x" * (3 * MB), "text/csv")}, headers=ORIGIN)
    body = r.json()
    assert r.status_code == 413
    assert body["error"] == "The file is too large. The maximum upload size is 1 MB."
    assert body["request_id"] == r.headers["x-request-id"]
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_other_request_bodies_are_capped_at_2_mb(tc):
    r = tc.post("/small", content=b"x" * (3 * MB), headers={"content-type": "application/octet-stream"})
    assert r.status_code == 413
    assert "2 MB" in r.json()["detail"]


def test_small_bodies_and_get_requests_are_not_affected(tc):
    assert tc.post("/small", content=b"x" * 1000).json()["bytes"] == 1000
    assert tc.get("/ok").status_code == 200


async def _chunked_upload(app, total_bytes: int):
    """A multipart upload with no Content-Length, delivered in 64 KB chunks."""
    boundary = "XB"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="c.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\na\n"
    ).encode()
    chunk = b"1234567\n" * 8192
    sent = {"bytes": 0}
    state = {"i": -1}

    async def receive():
        state["i"] += 1
        if state["i"] == 0:
            body = head
        elif sent["bytes"] < total_bytes:
            body = chunk
            sent["bytes"] += len(chunk)
        else:
            body = f"\r\n--{boundary}--\r\n".encode()
        return {"type": "http.request", "body": body, "more_body": sent["bytes"] < total_bytes}

    out = {"status": None, "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    scope = {
        "type": "http", "method": "POST", "path": "/upload", "raw_path": b"/upload",
        "query_string": b"", "root_path": "", "scheme": "http", "http_version": "1.1",
        "server": ("t", 80), "client": ("c", 1),
        "headers": [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())],
    }
    await app(scope, receive, send)
    return out, sent


async def test_a_chunked_upload_is_stopped_part_way(toy_app):
    out, sent = await _chunked_upload(toy_app, 50 * MB)
    assert out["status"] == 413
    assert b"maximum upload size" in out["body"]
    assert sent["bytes"] < 3 * MB          # it did not read all 50 MB


# ═══════════════════════════════════════════════════════════════
# C. Logging redaction
# ═══════════════════════════════════════════════════════════════

def _capture_logger(name: str):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(RedactingFormatter(fmt="%(levelname)s | %(name)s | %(message)s"))
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, buf


def test_log_lines_have_keys_and_tokens_redacted():
    logger, buf = _capture_logger("test.redact.a")
    logger.warning("failed with %s here", GOOGLE_KEY)
    logger.warning("GEMINI_API_KEY=abc123def loaded")
    logger.warning("GET /v1/models?key=SECRETVALUE&alt=json")
    logger.warning("Authorization: Bearer tok123")
    out = buf.getvalue()
    for secret in (GOOGLE_KEY, "abc123def", "SECRETVALUE", "tok123"):
        assert secret not in out
    assert out.count("[redacted]") >= 4


def test_the_configured_key_is_redacted_wherever_it_appears(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "my-very-secret-key-123")
    logger, buf = _capture_logger("test.redact.b")
    logger.warning("sdk said: bad key my-very-secret-key-123")
    assert "my-very-secret" not in buf.getvalue()


def test_secrets_inside_a_traceback_are_redacted():
    logger, buf = _capture_logger("test.redact.c")
    try:
        raise RuntimeError(f"upstream rejected Authorization: Bearer abc.def.ghi and key {GOOGLE_KEY}")
    except RuntimeError:
        logger.exception("call failed")
    out = buf.getvalue()
    assert "Traceback (most recent call last)" in out and "RuntimeError" in out
    assert "abc.def.ghi" not in out and GOOGLE_KEY not in out


def test_ordinary_log_lines_are_untouched_and_paths_are_kept():
    logger, buf = _capture_logger("test.redact.d")
    sid = str(uuid.uuid4())
    line = f"Session created | session_id={sid} | file=test.csv | rows=5 | cols=4"
    logger.info(line)
    logger.info("POST /api/clean/%s/apply -> 200 | 38ms | request_id=3f9a1c2b4d5e6f70", sid)
    logger.info(r"Prompt template not found: C:\Users\Biswajit\x.txt")
    out = buf.getvalue()
    assert line in out
    assert f"/api/clean/{sid}/apply -> 200 | 38ms | request_id=3f9a1c2b4d5e6f70" in out
    assert r"C:\Users\Biswajit\x.txt" in out          # server logs keep paths for debugging


def test_a_malformed_log_call_does_not_crash_the_formatter():
    logger, _ = _capture_logger("test.redact.e")
    logging.raiseExceptions = False                    # logging reports its own errors on stderr; silence that
    try:
        logger.info("%s and %s", "only-one")
        logger.info(None)
        logger.info({"a": 1})
    finally:
        logging.raiseExceptions = True


@pytest.mark.parametrize("name, level, valid", [
    ("INFO", logging.INFO, True), ("debug", logging.DEBUG, True), ("WARNING", logging.WARNING, True),
    ("LOUD", logging.INFO, False), ("basic_format", logging.INFO, False),
])
def test_log_level_names(name, level, valid):
    assert _resolve_level(name) == (level, valid)


# ═══════════════════════════════════════════════════════════════
# D. Gemini client (the SDK call is faked)
# ═══════════════════════════════════════════════════════════════

class Boom(Exception):
    def __init__(self, code, message="boom"):
        super().__init__(message)
        self.code = code


@pytest.fixture
def gemini(monkeypatch):
    gc.reset_gemini_state()
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "k" * 20)
    monkeypatch.setattr(settings, "GEMINI_MAX_RETRIES", 3)
    monkeypatch.setattr(gc, "BACKOFF_BASE_SECONDS", 0.01)
    monkeypatch.setattr(gc, "BACKOFF_MAX_SECONDS", 0.05)

    def use(fn):
        monkeypatch.setattr(gc, "_sync_generate", fn)

    use.calls = []
    yield use
    gc.reset_gemini_state()


async def test_no_key_means_no_call(gemini, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    gemini(lambda p: gemini.calls.append(p) or "x")
    assert await gc.generate_text("hi") is None
    assert gemini.calls == []


async def test_a_reply_is_returned_stripped(gemini):
    gemini(lambda p: "  hello  ")
    assert await gc.generate_text("hi") == "hello"


async def test_blank_prompts_are_ignored(gemini):
    gemini(lambda p: "x")
    assert await gc.generate_text("   ") is None
    assert await gc.generate_text(None) is None


async def test_rate_limits_are_retried(gemini):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise Boom(429, "RESOURCE_EXHAUSTED")
        return "ok"

    gemini(flaky)
    assert await gc.generate_text("hi") == "ok"
    assert calls["n"] == 3 and gc.cooldown_remaining() == 0


async def test_server_errors_and_message_only_rate_limits_are_retried(gemini):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Boom(503)
        if calls["n"] == 2:
            raise RuntimeError("429 Too Many Requests: quota exceeded")
        return "fine"

    gemini(flaky)
    assert await gc.generate_text("hi") == "fine" and calls["n"] == 3


async def test_exhausted_retries_start_a_cooldown_that_skips_later_calls(gemini):
    calls = {"n": 0}

    def always_limited(prompt):
        calls["n"] += 1
        raise Boom(429)

    gemini(always_limited)
    assert await gc.generate_text("hi") is None
    assert calls["n"] == 4                              # first attempt + 3 retries
    assert gc.cooldown_remaining() > 0
    assert gc.gemini_status()["cooling_down"] is True
    assert await gc.generate_text("hi") is None
    assert calls["n"] == 4                              # skipped, not retried
    gc.reset_gemini_state()
    assert gc.cooldown_remaining() == 0


async def test_a_client_error_is_not_retried_and_starts_no_cooldown(gemini):
    calls = {"n": 0}

    def forbidden(prompt):
        calls["n"] += 1
        raise Boom(403, "API key not valid")

    gemini(forbidden)
    assert await gc.generate_text("hi") is None
    assert calls["n"] == 1 and gc.cooldown_remaining() == 0


async def test_a_timeout_returns_quickly_and_is_not_retried(gemini, monkeypatch):
    monkeypatch.setattr(gc, "_timeout", lambda: 0.2)
    monkeypatch.setattr(gc, "TIMEOUT_MARGIN_SECONDS", 0.1)
    calls = {"n": 0}

    def slow(prompt):
        calls["n"] += 1
        time.sleep(1.0)
        return "late"

    gemini(slow)
    started = time.perf_counter()
    assert await gc.generate_text("hi") is None
    assert time.perf_counter() - started < 0.8
    assert calls["n"] == 1 and gc.cooldown_remaining() > 0


async def test_an_empty_reply_is_treated_as_no_answer(gemini):
    gemini(lambda p: "   ")
    assert await gc.generate_text("hi") is None


async def test_a_missing_sdk_is_remembered(gemini):
    calls = {"n": 0}

    def no_sdk(prompt):
        calls["n"] += 1
        raise ImportError("cannot import name 'genai' from 'google'")

    gemini(no_sdk)
    assert await gc.generate_text("hi") is None
    assert await gc.generate_text("hi") is None
    assert calls["n"] == 1


async def test_unexpected_sdk_errors_never_escape(gemini):
    def weird(prompt):
        raise ValueError("weird")

    gemini(weird)
    assert await gc.generate_text("hi") is None


async def test_concurrent_calls_all_complete(gemini):
    gemini(lambda p: (time.sleep(0.1), p)[1])
    results = await asyncio.gather(*[gc.generate_text(f"p{i}") for i in range(8)])
    assert results == [f"p{i}" for i in range(8)]


@pytest.mark.parametrize("raw, expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": [1, 2]}\n```', {"a": [1, 2]}),
    ('Sure! Here you go:\n{"a": 1}\nHope that helps.', {"a": 1}),
    ("not json at all", None),
    ("[1, 2, 3]", None),
    ('{"a": ', None),
    (None, None),
])
def test_json_replies_are_parsed_leniently(raw, expected):
    assert gc.parse_json_response(raw) == expected


async def test_generate_json_returns_a_dict_or_none(gemini):
    gemini(lambda p: '```json\n{"insights": ["x"]}\n```')
    assert await gc.generate_json("hi") == {"insights": ["x"]}
    gemini(lambda p: "garbage")
    assert await gc.generate_json("hi") is None


def test_prompt_rendering(tmp_path, monkeypatch):
    (tmp_path / "demo.txt").write_text('A {{FIRST}} B {{SECOND}} C {"json": "braces stay"}\n', encoding="utf-8")
    monkeypatch.setattr(gc, "PROMPTS_DIR", tmp_path)
    out = gc.render_prompt("demo.txt", FIRST="one", SECOND="two")
    assert out.startswith("A one B two C") and '{"json": "braces stay"}' in out
    # a value is never re-scanned for placeholders
    assert gc.render_prompt("demo.txt", FIRST="{{SECOND}}", SECOND="two").startswith("A {{SECOND}} B two")
    # an unfilled placeholder is left as it is
    assert "{{SECOND}}" in gc.render_prompt("demo.txt", FIRST="x")
    assert gc.render_prompt("missing.txt") is None
    assert gc.render_prompt("../core/config.py") is None and gc.render_prompt("..\\x") is None


def test_the_status_never_includes_the_key(gemini):
    status = gc.gemini_status()
    assert set(status) == {"configured", "model", "sdk_installed", "cooling_down", "retry_after_seconds"}
    assert status["configured"] is True
    assert "kkkk" not in str(status)


# ═══════════════════════════════════════════════════════════════
# E. Session service
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """An empty, private upload folder."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    for directory in ss._session_dirs():
        directory.mkdir(parents=True, exist_ok=True)
    ss._last_access.clear()
    ss._locks.clear()
    yield tmp_path
    ss._last_access.clear()
    ss._locks.clear()


def frame(n=5, v=0):
    return pd.DataFrame({"a": range(n), "v": [v] * n})


def new_session(df=None):
    return ss.create_session(frame() if df is None else df, "t.csv")["session_id"]


def all_files():
    return [p for d in ss._session_dirs() for p in d.iterdir()]


def test_create_session_writes_the_files_and_returns_metadata(uploads):
    meta = ss.create_session(frame(), "demo.csv")
    sid = meta["session_id"]
    assert meta["rows"] == 5 and meta["columns"] == 2 and meta["column_names"] == ["a", "v"]
    assert meta["filename"] == "demo.csv" and meta["file_size_kb"] > 0
    assert all(p.exists() for p in (ss._original_path(sid), ss._working_path(sid), ss._history_path(sid)))
    assert ss.get_history(sid) == []
    assert not ss._snapshot_path(sid).exists()
    assert not [p for p in all_files() if p.suffix == ".tmp"]


def test_session_exists(uploads):
    sid = new_session()
    assert ss.session_exists(sid)
    assert not ss.session_exists("00000000-0000-4000-8000-000000000000")
    assert not ss.session_exists("../../etc/passwd")
    assert not ss.session_exists(None)


def test_editing_the_working_copy_never_touches_the_original(uploads):
    sid = new_session()
    ss.save_working_df(sid, frame(3))
    assert len(ss.get_working_df(sid)) == 3 and len(ss.get_original_df(sid)) == 5


def test_commit_then_undo(uploads):
    sid = new_session()
    before = ss.get_working_df(sid)
    ss.commit_change(sid, frame(5, v=1), {"stage": "clean", "strategy": "x"})

    assert ss.get_working_df(sid)["v"].tolist() == [1] * 5
    assert pd.read_csv(ss._snapshot_path(sid)).equals(before)
    history = ss.get_history(sid)
    assert len(history) == 1 and history[0]["strategy"] == "x" and "timestamp" in history[0]

    ss.restore_snapshot(sid)
    assert ss.get_working_df(sid).equals(before)
    assert ss.get_history(sid) == []
    assert not ss._snapshot_path(sid).exists()


def test_a_second_undo_changes_nothing(uploads):
    sid = new_session()
    ss.commit_change(sid, frame(5, v=1), {"n": 1})
    ss.commit_change(sid, frame(5, v=2), {"n": 2})
    ss.restore_snapshot(sid)
    with pytest.raises(SnapshotNotFoundError):
        ss.restore_snapshot(sid)
    assert ss.get_working_df(sid)["v"].tolist() == [1] * 5
    assert [e["n"] for e in ss.get_history(sid)] == [1]      # history was not popped again


def test_undo_still_works_after_an_operation_that_never_committed(uploads):
    """A failed operation never reaches commit_change, so it cannot disturb the snapshot."""
    sid = new_session()
    ss.commit_change(sid, frame(5, v=7), {"n": "A"})
    snapshot = pd.read_csv(ss._snapshot_path(sid))
    # operation B fails before commit: nothing is called
    ss.restore_snapshot(sid)
    assert ss.get_working_df(sid).equals(snapshot)
    assert ss.get_history(sid) == []


def test_commit_without_data_records_and_snapshots(uploads):
    sid = new_session()
    ss.commit_change(sid, None, {"stage": "transform", "transform_type": "exclude_ml"})
    assert ss.get_working_df(sid).equals(frame()) and len(ss.get_history(sid)) == 1
    assert ss._snapshot_path(sid).exists()
    ss.restore_snapshot(sid)
    assert ss.get_history(sid) == []


def test_a_failed_write_leaves_data_snapshot_and_history_untouched(uploads):
    sid = new_session()
    ss.commit_change(sid, frame(5, v=1), {"n": "good"})
    data = ss.get_working_df(sid).copy()
    snapshot = pd.read_csv(ss._snapshot_path(sid)).copy()
    history = ss.get_history(sid)

    class Exploding(pd.DataFrame):
        def to_csv(self, *args, **kwargs):
            raise OSError("disk full")

    with pytest.raises(OSError):
        ss.commit_change(sid, Exploding(frame(5, v=9)), {"n": "bad"})

    assert ss.get_working_df(sid).equals(data)
    assert pd.read_csv(ss._snapshot_path(sid)).equals(snapshot)
    assert ss.get_history(sid) == history
    assert not [p for p in all_files() if p.suffix == ".tmp"]


def test_reset_clears_everything_for_that_session_only(uploads):
    sid, other = new_session(), new_session()
    ss.commit_change(sid, frame(5, v=4), {"n": 1})
    cache.set_profile(sid, {"p": 1})
    cache.append_eda_history(sid, {"e": 1})
    cache.append_stats_history(sid, {"s": 1})
    cache.append_nlp_history(sid, {"n": 1})
    cache.set_ml_result(sid, {"m": 1})
    cache.set_profile(other, {"p": 2})
    cache.append_eda_history(other, {"e": 2})
    for report in ss._report_paths(sid):
        report.write_text("report")

    ss.reset_to_original(sid)

    assert ss.get_working_df(sid).equals(frame())
    assert ss.get_history(sid) == [] and not ss._snapshot_path(sid).exists()
    assert not any(p.exists() for p in ss._report_paths(sid))
    assert cache.get_profile(sid) is None and cache.get_eda_history(sid) == []
    assert cache.get_stats_history(sid) == [] and cache.get_nlp_history(sid) == []
    assert cache.get_ml_result(sid) is None
    assert cache.get_profile(other) == {"p": 2} and cache.get_eda_history(other) == [{"e": 2}]


def test_reset_of_an_unknown_session_raises(uploads):
    with pytest.raises(SessionNotFoundError):
        ss.reset_to_original("00000000-0000-4000-8000-000000000000")


@pytest.mark.parametrize("bad", ["../../etc/passwd", "..\\x", "abc", "", None])
def test_a_bad_id_can_never_become_a_path(uploads, bad):
    for fn in (ss.get_working_df, ss.get_original_df, ss.restore_snapshot, ss.reset_to_original, ss.take_snapshot):
        with pytest.raises(SessionNotFoundError):
            fn(bad)
    assert ss.delete_session(bad) is False


def test_a_corrupt_history_file_reads_as_empty(uploads):
    sid = new_session()
    ss._history_path(sid).write_text("{not json")
    assert ss.get_history(sid) == []


def test_concurrent_read_modify_commit_loses_no_update(uploads):
    sid = new_session(pd.DataFrame({"v": [0]}))

    def worker():
        with ss.session_lock(sid):
            df = ss.get_working_df(sid)
            time.sleep(0.01)
            df["v"] += 1
            ss.commit_change(sid, df, {"op": "inc"})     # re-entrant lock inside the caller's lock

    threads = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert ss.get_working_df(sid)["v"].tolist() == [12]
    assert len(ss.get_history(sid)) == 12


def test_readers_never_see_a_half_written_file(uploads):
    sid = new_session(pd.DataFrame({"x": range(2000)}))
    errors, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                if len(ss.get_working_df(sid)) not in (2000, 1000):
                    errors.append("partial file")
            except Exception as e:
                errors.append(repr(e))

    def writer():
        for i in range(30):
            ss.commit_change(sid, pd.DataFrame({"x": range(1000 if i % 2 else 2000)}), {"i": i})

    readers = [threading.Thread(target=reader) for _ in range(3)]
    w = threading.Thread(target=writer)
    [t.start() for t in readers]
    w.start()
    w.join()
    stop.set()
    [t.join() for t in readers]
    assert errors == []


def _age(session_id, hours):
    stamp = time.time() - hours * 3600
    for path in ss._all_session_files(session_id):
        if path.exists():
            os.utime(path, (stamp, stamp))
    ss._last_access.pop(session_id, None)


def test_cleanup_deletes_only_expired_sessions(uploads):
    old, recent, active = new_session(), new_session(), new_session()
    orphan = str(uuid.uuid4())
    for sid in (old, active):
        _age(sid, 5)
    ss._history_path(orphan).write_text("[]")
    os.utime(ss._history_path(orphan), (time.time() - 5 * 3600,) * 2)
    (settings.original_path / ".gitkeep").write_text("")
    (settings.original_path / "notes.txt").write_text("hi")
    (settings.working_path / "not-a-uuid.csv").write_text("a\n1")
    ss.session_exists(active)                              # a request touched it just now
    cache.set_profile(old, {"x": 1})
    cache.append_eda_history(old, {"e": 1})
    for report in ss._report_paths(old):
        report.write_text("r")
    ss.take_snapshot(old)
    _age(old, 5)

    assert ss.cleanup_expired_sessions(0) == 0 and ss._working_path(old).exists()   # 0 disables it
    assert ss.cleanup_expired_sessions(1) == 2                                      # the old session and the orphan

    assert not any(p.exists() for p in ss._all_session_files(old))
    assert cache.get_profile(old) is None and cache.get_eda_history(old) == []
    assert not ss._history_path(orphan).exists()
    assert ss._working_path(recent).exists()
    assert ss._working_path(active).exists()               # old files, but used a moment ago
    assert (settings.original_path / ".gitkeep").exists()
    assert (settings.original_path / "notes.txt").exists()
    assert (settings.working_path / "not-a-uuid.csv").exists()
    assert old not in ss._locks


def test_cleanup_skips_a_session_that_is_in_use(uploads):
    busy = new_session()
    _age(busy, 5)
    held, release = threading.Event(), threading.Event()

    def hold():
        with ss.session_lock(busy):
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait()
    try:
        assert ss.cleanup_expired_sessions(1) == 0 and ss._working_path(busy).exists()
    finally:
        release.set()
        t.join()
    assert ss.cleanup_expired_sessions(1) == 1 and not ss._working_path(busy).exists()


def test_cleanup_sweeps_stale_temp_files_only(uploads):
    stale = settings.working_path / f"{uuid.uuid4()}.csv.abcd1234.tmp"
    fresh = settings.working_path / f"{uuid.uuid4()}.csv.ffff0000.tmp"
    stale.write_text("x")
    fresh.write_text("x")
    os.utime(stale, (time.time() - 7200,) * 2)
    ss.cleanup_expired_sessions(1)
    assert not stale.exists() and fresh.exists()


def test_delete_session(uploads):
    sid = new_session()
    cache.set_ml_result(sid, {"m": 1})
    assert ss.delete_session(sid) is True
    assert not ss.session_exists(sid) and cache.get_ml_result(sid) is None
    assert ss.delete_session(sid) is False


async def test_the_cleanup_loop_runs_and_survives_a_failure(monkeypatch):
    import main

    calls = []

    def flaky(hours):
        calls.append(hours)
        if len(calls) == 1:
            raise RuntimeError("scan failed")
        return 0

    monkeypatch.setattr(main.session_service, "cleanup_expired_sessions", flaky)
    monkeypatch.setattr(main, "CLEANUP_INTERVAL_SECONDS", 0.05)
    task = asyncio.create_task(main._cleanup_loop(24))
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) >= 2 and set(calls) == {24}          # it kept going after the first failure


# ═══════════════════════════════════════════════════════════════
# F. The event loop is never blocked
# ═══════════════════════════════════════════════════════════════

async def measure(app, request_fn):
    """
    Run one slow request and /api/health together on a single event loop.
    Returns (response, ticks, total_seconds, seconds_until_health_answered).
    A ticker task counts how often the loop gets a turn: about total/0.02 when it
    is free, and almost none when something blocks it.
    """
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        ticker_task = asyncio.create_task(ticker())
        started = time.perf_counter()
        slow = asyncio.create_task(request_fn(ac))
        health = await ac.get("/api/health")
        health_after = time.perf_counter() - started
        response = await slow
        total = time.perf_counter() - started
        ticker_task.cancel()
        assert health.status_code == 200
        return response, ticks, total, health_after


def loop_is_free(ticks, total, health_after):
    return ticks >= 0.5 * (total / 0.02) and health_after < 0.3


def loop_is_blocked(ticks, total, health_after):
    return ticks < 0.2 * (total / 0.02) and health_after > 0.6 * total


def _control_app(uploads_sid=None) -> FastAPI:
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/blocking-sleep")
    async def blocking_sleep():
        time.sleep(0.8)                                   # control: blocks the loop on purpose
        return {"ok": True}

    @app.get("/async-read/{sid}")
    async def async_read(sid: str):
        return {"rows": len(await ss.aget_working_df(sid))}

    @app.get("/blocking-read/{sid}")
    async def blocking_read(sid: str):
        return {"rows": len(ss.get_working_df(sid))}      # control: a blocking read on the loop

    return app


async def test_the_harness_detects_a_blocked_loop():
    """If this passes, the checks below are able to fail."""
    _, ticks, total, health_after = await measure(_control_app(), lambda ac: ac.get("/blocking-sleep"))
    assert loop_is_blocked(ticks, total, health_after), (ticks, total, health_after)


def _hold_lock(sid, seconds):
    held, done = threading.Event(), threading.Event()

    def run():
        with ss.session_lock(sid):
            held.set()
            time.sleep(seconds)
        done.set()

    threading.Thread(target=run).start()
    held.wait()
    return done


async def test_aget_working_df_waits_for_the_lock_off_the_event_loop(uploads):
    sid = new_session(pd.DataFrame({"a": [1, 2, 3]}))
    done = _hold_lock(sid, 0.8)
    response, ticks, total, health_after = await measure(_control_app(), lambda ac: ac.get(f"/async-read/{sid}"))
    done.wait()
    assert response.status_code == 200 and response.json()["rows"] == 3
    assert total >= 0.6                                    # it really did wait for the lock
    assert loop_is_free(ticks, total, health_after), (ticks, total, health_after)


async def test_a_blocking_read_would_have_frozen_the_loop(uploads):
    """Control for the test above: the same wait, done on the loop, is detected as a block."""
    sid = new_session(pd.DataFrame({"a": [1, 2, 3]}))
    done = _hold_lock(sid, 0.8)
    _, ticks, total, health_after = await measure(_control_app(), lambda ac: ac.get(f"/blocking-read/{sid}"))
    done.wait()
    assert loop_is_blocked(ticks, total, health_after), (ticks, total, health_after)


# ═══════════════════════════════════════════════════════════════
# G. Unit tests for the patched code
# ═══════════════════════════════════════════════════════════════

def _corpus(n):
    pool = [
        "quality excellent design lovely", "terrible broke slow delivery poor",
        "arrived tuesday brown box package", "great value price fast service",
    ]
    return pd.DataFrame({"T": [f"{pool[i % 4]} number{i % 50}" for i in range(n)]})


@pytest.mark.parametrize("analysis, n_rows", [
    ("topics", 4999), ("topics", 5000), ("clusters", 2000), ("similar_pairs", 200),
    ("quality", 10 ** 6), ("sentiment", 10 ** 6), ("keywords", 10 ** 6), ("text_stats", 10 ** 6),
])
def test_no_sampling_warning_when_everything_was_used(analysis, n_rows):
    from services.nlp_service import sampling_warning
    assert sampling_warning(analysis, n_rows) is None


@pytest.mark.parametrize("analysis, n_rows, text", [
    ("topics", 12000, "random sample of 5,000 of your 12,000 rows"),
    ("document_topics", 5001, "random sample of 5,000 of your 5,001 rows"),
    ("clusters", 2001, "random sample of 2,000 of your 2,001 rows"),
    ("cluster_keywords", 9000, "2,000 of your 9,000"),
    ("cluster_sizes", 9000, "2,000 of your 9,000"),
    ("similarity_search", 9000, "2,000 of your 9,000"),
    ("similar_pairs", 300, "200 of your 300 rows"),
])
def test_sampling_warning_says_what_was_used(analysis, n_rows, text):
    from services.nlp_service import sampling_warning
    assert text in sampling_warning(analysis, n_rows)


def test_lda_corpus_is_untouched_when_small_and_capped_when_large():
    from services.nlp_service import LDA_SAMPLE_CAP, _lda_corpus
    small = _corpus(100)
    assert _lda_corpus(small, "T").index.tolist() == list(range(100))
    big = _corpus(LDA_SAMPLE_CAP + 1000)
    sample = _lda_corpus(big, "T")
    assert len(sample) == LDA_SAMPLE_CAP and sample.index.is_unique
    assert sample.index.tolist() == _lda_corpus(big, "T").index.tolist()      # deterministic


def test_topic_modelling_on_a_large_corpus_is_capped_and_keeps_original_row_numbers():
    pytest.importorskip("sklearn")
    from services.nlp_service import document_topic_assignment, lda_topics
    big = _corpus(6000)
    topics = lda_topics(big, "T", n_topics=3)
    assert len(topics) == 3 and "error" not in topics[0]
    assignments = document_topic_assignment(big, "T", n_topics=3)
    assert len(assignments) == 5000
    assert len({a["index"] for a in assignments}) == 5000
    assert all(0 <= a["index"] < 6000 for a in assignments)
    assert all(big["T"].iloc[a["index"]].startswith(a["text_preview"][:20]) for a in assignments[:50])


def test_topic_modelling_on_a_small_corpus_keeps_every_row_in_order():
    pytest.importorskip("sklearn")
    from services.nlp_service import document_topic_assignment, lda_topics
    small = _corpus(40)
    topics = lda_topics(small, "T", n_topics=3)
    assert len(topics) == 3 and abs(sum(t["proportion"] for t in topics) - 100) < 0.1
    assert [a["index"] for a in document_topic_assignment(small, "T", n_topics=3)] == list(range(40))


def test_the_multivariate_column_cap_is_reported():
    from services.eda_service import multivariate_analysis
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(size=(60, 34)), columns=[f"c{i}" for i in range(34)])

    capped = multivariate_analysis(wide)
    assert capped["n_columns"] == 20 and capped["total_numeric_columns"] == 34
    assert capped["columns_capped"] is True
    assert capped["warning"] == "Showing the first 20 of 34 numerical columns."

    narrow = multivariate_analysis(wide.iloc[:, :6])
    assert narrow["columns_capped"] is False and narrow["warning"] is None

    chosen = multivariate_analysis(wide, columns=["c0", "c1", "c2"])
    assert chosen["n_columns"] == 3 and chosen["columns_capped"] is False
    assert {"columns_analysed", "n_columns", "heatmap_data", "strong_pairs"} <= set(chosen)
    assert "error" in multivariate_analysis(wide.iloc[:, :1])


FORMULA_FRAME = pd.DataFrame({
    "Age": [25.0, 30.0, 35.0], "price": [4.0, 9.0, 16.0], "Total Sales": [1.0, 2.0, 3.0],
    "Open": [5.0, 6.0, 7.0], "Close": [6.0, 6.0, 9.0],
})


@pytest.mark.parametrize("formula", [
    "abs(Age - 30)", "sqrt(price)", "log(price)", "log10(price)", "log1p(price)", "exp(Age / 100)",
    "sin(Age)", "cos(Age)", "tan(Age)", "floor(price / 3)", "ceil(price / 3)",
    "`Total Sales` * 2", "Open - Close", "Age > 30 and price < 20", "not (Age > 30)", "Age ** 2",
])
def test_every_formula_the_validator_allows_really_evaluates(formula):
    from services.transform_service import apply_formula
    out = apply_formula(FORMULA_FRAME, formula, "result")
    assert "result" in out.columns and len(out) == 3
    assert "result" not in FORMULA_FRAME.columns                         # the input is not modified


def test_formula_values_are_correct():
    from services.transform_service import apply_formula
    assert apply_formula(FORMULA_FRAME, "`Total Sales` * 2", "d")["d"].tolist() == [2.0, 4.0, 6.0]
    assert apply_formula(FORMULA_FRAME, "Open - Close", "d")["d"].tolist() == [-1.0, 0.0, -2.0]


@pytest.mark.parametrize("formula, fragment", [
    ("import os", "disallowed keyword"),
    ("9**9**9", "chained"),
    ("Age = 5", "assignment"),
    ("Age.mean()", "attribute"),
    ("unknown + 1", "unknown column"),
    ("Age ** (1/2)", "exponent"),
])
def test_a_rejected_formula_explains_why(formula, fragment):
    from services.transform_service import apply_formula
    with pytest.raises(InvalidFormulaError) as caught:
        apply_formula(FORMULA_FRAME, formula, "x")
    assert fragment in str(caught.value).lower()
    assert caught.value.reason == str(caught.value)


@pytest.mark.parametrize("name", ["", "   ", "x" * 101, "bad\nname"])
def test_formula_new_column_names_are_checked(name):
    from services.transform_service import apply_formula
    with pytest.raises(InvalidFormulaError):
        apply_formula(FORMULA_FRAME, "Age + 1", name)


def test_formula_new_column_defaults_to_derived():
    from services.transform_service import apply_formula
    assert "derived" in apply_formula(FORMULA_FRAME, "Age + 1", None).columns


def test_an_evaluation_failure_does_not_leak_internals():
    from services.transform_service import apply_formula
    with pytest.raises(InvalidFormulaError) as caught:
        apply_formula(FORMULA_FRAME, "Age +", "x")
    assert "Traceback" not in str(caught.value) and "pandas" not in str(caught.value).lower()


# ═══════════════════════════════════════════════════════════════
# H. Upload endpoint (real dataset router, real application)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def api(tmp_path_factory):
    """The real app on an isolated upload folder. Never touches your real sessions."""
    from main import app
    patch = pytest.MonkeyPatch()
    patch.setattr(settings, "UPLOAD_DIR", str(tmp_path_factory.mktemp("uploads")))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    patch.undo()


def csv_of(df):
    return df.to_csv(index=False).encode()


def upload(client, df, name="data.csv"):
    r = client.post("/api/dataset/upload", files={"file": (name, csv_of(df), "text/csv")})
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def raw_upload(client, name, data, mime="text/csv"):
    return client.post("/api/dataset/upload", files={"file": (name, data, mime)})


CSV = b"a,b\n1,x\n2,y\n3,\n"
XLS = bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 600


class TestUploadEndpoint:
    def test_csv_upload(self, api):
        r = raw_upload(api, "sales.csv", CSV)
        body = r.json()
        assert r.status_code == 201
        assert body["rows"] == 3 and body["columns"] == 2 and body["column_names"] == ["a", "b"]
        assert body["filename"] == "sales.csv"

    @pytest.mark.parametrize("mime", ["application/vnd.ms-excel", "application/octet-stream", "text/csv; charset=utf-8"])
    def test_csv_with_the_mime_types_real_browsers_send(self, api, mime):
        assert raw_upload(api, "w.csv", CSV, mime).status_code == 201

    def test_upper_case_extension(self, api):
        assert raw_upload(api, "DATA.CSV", CSV).status_code == 201

    def test_the_file_name_is_reduced_to_a_base_name(self, api):
        r = raw_upload(api, "C:\\Users\\Biswajit\\Desktop\\secret folder\\data.csv", CSV)
        assert r.status_code == 201 and r.json()["filename"] == "data.csv" and "Biswajit" not in r.text
        r = raw_upload(api, "../../etc/evil\u202efdp.csv", CSV)
        assert "/" not in r.json()["filename"] and "\u202e" not in r.json()["filename"]

    def test_xlsx_upload_with_numeric_headers(self, api):
        pytest.importorskip("openpyxl")
        buf = io.BytesIO()
        pd.DataFrame({1: [1, 2], 2: [3, 4]}).to_excel(buf, index=False)
        r = raw_upload(api, "book.xlsx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        assert r.status_code == 201
        assert r.json()["column_names"] == ["1", "2"]            # headers are text, so the response validates

    def test_an_xlsx_saved_with_an_xls_extension_is_read_as_xlsx(self, api):
        pytest.importorskip("openpyxl")
        buf = io.BytesIO()
        pd.DataFrame({"a": [1, 2]}).to_excel(buf, index=False)
        assert raw_upload(api, "misnamed.xls", buf.getvalue(), "application/vnd.ms-excel").status_code == 201

    def test_xls_is_read_with_the_xlrd_engine(self, api, monkeypatch):
        engines = []

        def fake_read_excel(buffer, engine=None, **kwargs):
            engines.append(engine)
            return pd.DataFrame({"x": [1]})

        monkeypatch.setattr(pd, "read_excel", fake_read_excel)
        assert raw_upload(api, "legacy.xls", XLS, "application/vnd.ms-excel").status_code == 201
        assert engines == ["xlrd"]

    @pytest.mark.parametrize("name, data, mime, fragment", [
        ("d.json", b'{"a":1}', "application/json", "unsupported"),
        ("data", CSV, "text/csv", "unsupported"),
        ("d.csv", CSV, "application/json", "not allowed"),
        ("e.csv", b"", "text/csv", "empty"),
        ("p.csv", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 50, "text/csv", "binary"),
        ("x.xlsx", CSV, "text/csv", "real .xlsx"),
        ("o.xlsx", XLS, "text/csv", "password"),
        ("u.csv", "a,b\n1,2\n".encode("utf-16"), "text/csv", "utf-16"),
        ("l.csv", "name,city\nJosé,Zürich\n".encode("latin-1"), "text/csv", "utf-8"),
        ("r.csv", b"a,b\n1,2\n3,4,5\n", "text/csv", "same number of columns"),
        ("h.csv", b"a,b\n", "text/csv", "no data"),
        ("w.csv", b"\n\n\n", "text/csv", "no data"),
    ])
    def test_rejections_explain_themselves(self, api, name, data, mime, fragment):
        r = raw_upload(api, name, data, mime)
        assert r.status_code == 400
        assert {"error", "detail", "code"} <= set(r.json())
        assert fragment in r.json()["detail"].lower()

    def test_a_workbook_renamed_to_csv_is_rejected(self, api):
        pytest.importorskip("openpyxl")
        buf = io.BytesIO()
        pd.DataFrame({"a": [1]}).to_excel(buf, index=False)
        r = raw_upload(api, "x.csv", buf.getvalue())
        assert r.status_code == 400 and "named .csv" in r.json()["detail"]

    def test_a_zip_that_is_not_a_workbook_is_rejected(self, api):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("word/document.xml", "<x/>")
        r = raw_upload(api, "d.xlsx", buf.getvalue())
        assert r.status_code == 400 and "not an excel workbook" in r.json()["detail"].lower()

    def test_a_zip_bomb_is_rejected(self, api, monkeypatch):
        pytest.importorskip("openpyxl")
        from core import security
        buf = io.BytesIO()
        pd.DataFrame({"a": [1]}).to_excel(buf, index=False)
        monkeypatch.setattr(security, "MAX_XLSX_UNCOMPRESSED_BYTES", 10)
        r = raw_upload(api, "b.xlsx", buf.getvalue())
        assert r.status_code == 400 and "expands" in r.json()["detail"]

    def test_too_many_columns(self, api, monkeypatch):
        from routers import dataset
        monkeypatch.setattr(dataset, "MAX_COLUMNS", 3)
        r = raw_upload(api, "wide.csv", b"a,b,c,d\n1,2,3,4\n")
        assert r.status_code == 400 and "4 columns" in r.json()["detail"]

    def test_the_size_limit_is_exact(self, api, monkeypatch):
        monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
        exact = (b"a\n" + b"1\n" * MB)[:MB]
        assert len(exact) == MB and raw_upload(api, "exact.csv", exact).status_code == 201
        over = raw_upload(api, "over.csv", exact + b"1\n")
        assert over.status_code == 413
        assert over.json()["detail"] == "The file is too large. The maximum upload size is 1 MB."

    def test_an_upload_declared_far_too_large_is_rejected_without_being_read(self, api):
        r = api.post(
            "/api/dataset/upload",
            files={"file": ("a.csv", CSV, "text/csv")},
            headers={"content-length": str((settings.MAX_UPLOAD_MB + 5) * MB)},
        )
        assert r.status_code == 413
        assert {"error", "detail", "code", "request_id"} <= set(r.json())

    def test_other_bodies_are_capped_at_2_mb(self, api):
        r = api.post(
            "/api/dataset/00000000-0000-4000-8000-000000000000/reset",
            content=b"x" * (3 * MB), headers={"content-type": "application/octet-stream"},
        )
        assert r.status_code == 413 and "2 MB" in r.json()["detail"]
        small = api.post("/api/dataset/00000000-0000-4000-8000-000000000000/reset", content=b"x" * 1000)
        assert small.status_code == 404

    def test_error_responses_leak_nothing(self, api):
        responses = [
            raw_upload(api, "l.csv", "name,city\nJosé,Zürich\n".encode("latin-1")),
            raw_upload(api, "d.json", b"{}"),
            api.get("/no/such/route"),
            api.get("/api/dataset/not-a-uuid/info"),
        ]
        for r in responses:
            assert {"error", "detail", "code"} <= set(r.json())
            for leak in ("Traceback", "UnicodeDecodeError", "codec", "site-packages", "Biswajit"):
                assert leak not in r.text

    def test_preview_info_and_reset(self, api):
        sid = upload(api, pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]}))
        preview = api.get(f"/api/dataset/{sid}/preview").json()
        assert preview["total_rows"] == 3 and preview["rows"][1]["b"] is None
        info = api.get(f"/api/dataset/{sid}/info").json()
        assert info["rows"] == 3 and info["columns"] == 2 and info["memory_usage_mb"] >= 0

        from services import session_service
        session_service.commit_change(sid, pd.DataFrame({"a": [9]}), {"stage": "clean"})
        cache.set_profile(sid, {"p": 1})
        cache.append_eda_history(sid, {"e": 1})
        cache.set_ml_result(sid, {"m": 1})
        r = api.post(f"/api/dataset/{sid}/reset")
        assert r.status_code == 200 and r.json()["status"] == "reset"
        assert api.get(f"/api/dataset/{sid}/info").json()["rows"] == 3
        assert session_service.get_history(sid) == []
        assert cache.get_profile(sid) is None and cache.get_eda_history(sid) == [] and cache.get_ml_result(sid) is None

    @pytest.mark.parametrize("path", ["preview", "info"])
    def test_bad_and_unknown_ids(self, api, path):
        assert api.get(f"/api/dataset/not-a-uuid/{path}").status_code == 400
        assert api.get(f"/api/dataset/00000000-0000-4000-8000-000000000000/{path}").status_code == 404
        assert api.post("/api/dataset/not-a-uuid/reset").status_code == 400

    def test_the_upload_handlers_are_not_coroutines(self):
        """Plain `def` endpoints run in FastAPI's thread pool, so parsing never blocks the loop."""
        from routers import dataset
        for fn in (dataset.upload_dataset, dataset.preview_dataset, dataset.reset_dataset, dataset.dataset_info):
            assert not inspect.iscoroutinefunction(fn)


async def test_a_slow_upload_does_not_block_the_event_loop(api, monkeypatch):
    from main import app
    from routers import dataset
    original = dataset._parse

    def slow_parse(*args, **kwargs):
        time.sleep(0.8)
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "_parse", slow_parse)
    response, ticks, total, health_after = await measure(
        app, lambda ac: ac.post("/api/dataset/upload", files={"file": ("a.csv", CSV, "text/csv")})
    )
    assert response.status_code == 201 and total >= 0.7
    assert loop_is_free(ticks, total, health_after), (ticks, total, health_after)


# ═══════════════════════════════════════════════════════════════
# I. Project integration (real clean, transform, EDA, profile and report routers)
# ═══════════════════════════════════════════════════════════════

NUMBERS = pd.DataFrame({
    "Age": [25, None, 35, None, 45],
    "Income": [100.0, 200.0, None, 400.0, 500.0],
    "Revenue": [10.0, -5.0, 20.0, -1.0, 30.0],
    "Gender": ["M", "F", None, "F", "M"],
})


def _rows(client, sid):
    return client.get(f"/api/dataset/{sid}/preview").json()["rows"]


def _clean(client, sid, **body):
    return client.post(f"/api/clean/{sid}/apply", json=body)


def _transform(client, sid, **body):
    return client.post(f"/api/transform/{sid}/apply", json=body)


class TestProjectIntegration:
    def test_a_failed_apply_does_not_disturb_undo(self, api):
        sid = upload(api, NUMBERS)
        original = _rows(api, sid)

        assert _clean(api, sid, issue_type="missing_numerical", column="Age", strategy="median").status_code == 200
        bad = _clean(api, sid, issue_type="missing_numerical", column="Nope", strategy="median")
        assert bad.status_code == 400 and "not found" in bad.json()["detail"].lower()

        undo = api.post(f"/api/clean/{sid}/undo")
        assert undo.status_code == 200
        assert _rows(api, sid) == original                    # the LAST GOOD operation was undone
        assert api.get(f"/api/clean/{sid}/history").json()["total_operations"] == 0

    def test_a_second_undo_returns_404_and_changes_nothing(self, api):
        sid = upload(api, NUMBERS)
        _clean(api, sid, issue_type="missing_numerical", column="Age", strategy="median")
        assert api.post(f"/api/clean/{sid}/undo").status_code == 200
        again = api.post(f"/api/clean/{sid}/undo")
        assert again.status_code == 404 and "nothing to undo" in again.json()["detail"].lower()

    def test_a_clean_column_is_required_except_for_duplicates(self, api):
        sid = upload(api, NUMBERS)
        missing = _clean(api, sid, issue_type="negative_values", strategy="set_zero")
        assert missing.status_code == 400 and "'column' is required" in missing.json()["detail"]
        assert _clean(api, sid, issue_type="duplicates", strategy="drop_duplicates").status_code == 200

    def test_the_set_nan_and_flag_only_warnings_survive(self, api):
        sid = upload(api, NUMBERS)
        r = _clean(api, sid, issue_type="negative_values", column="Revenue", strategy="set_nan")
        assert "missing values created by set_nan" in r.json()["warning"]
        api.post(f"/api/clean/{sid}/undo")
        r = _clean(api, sid, issue_type="negative_values", column="Revenue", strategy="flag_only")
        assert "Revenue_negative_flag" in r.json()["warning"]

    def test_applying_clears_the_cached_profile_and_logs_rows(self, api, caplog):
        sid = upload(api, NUMBERS)
        cache.set_profile(sid, {"p": 1})
        with caplog.at_level(logging.INFO):
            _clean(api, sid, issue_type="missing_numerical", column="Age", strategy="median")
        assert cache.get_profile(sid) is None
        assert any(f"session_id={sid}" in r.getMessage() and "rows 5→5" in r.getMessage() for r in caplog.records)

    def test_two_simultaneous_applies_lose_nothing(self, api):
        from main import app
        sid = upload(api, NUMBERS)

        def apply(column):
            TestClient(app).post(
                f"/api/clean/{sid}/apply",
                json={"issue_type": "missing_numerical", "column": column, "strategy": "median"},
            )

        threads = [threading.Thread(target=apply, args=(c,)) for c in ("Age", "Income")]
        [t.start() for t in threads]
        [t.join() for t in threads]
        rows = _rows(api, sid)
        assert all(row["Age"] is not None and row["Income"] is not None for row in rows)
        assert api.get(f"/api/clean/{sid}/history").json()["total_operations"] == 2

    def test_transform_formula_with_a_spaced_column_and_a_blocked_looking_column(self, api):
        sid = upload(api, pd.DataFrame({"Total Sales": [1.0, 2.0, 3.0], "Open": [5.0, 6.0, 7.0], "Close": [6.0, 6.0, 9.0]}))
        r = _transform(api, sid, transform_type="formula", column="Open",
                       options={"formula": "`Total Sales` * 2", "new_column": "double"})
        assert r.status_code == 200 and r.json()["new_columns"] == ["double"]
        r = _transform(api, sid, transform_type="formula", column="Open",
                       options={"formula": "Open - Close", "new_column": "diff"})
        assert r.status_code == 200

    @pytest.mark.parametrize("formula, fragment", [
        ("import os", "disallowed keyword"), ("__import__('os')", "double underscore"),
        ("9**9**9", "chained"), ("Open.mean()", "attribute"), ("unknown + 1", "unknown column"),
    ])
    def test_a_malicious_formula_is_rejected_and_changes_nothing(self, api, formula, fragment):
        sid = upload(api, pd.DataFrame({"Open": [5.0, 6.0], "Close": [6.0, 9.0]}))
        before = _rows(api, sid)
        r = _transform(api, sid, transform_type="formula", column="Open", options={"formula": formula, "new_column": "x"})
        assert r.status_code == 400 and fragment in r.json()["detail"].lower()
        assert _rows(api, sid) == before
        assert api.get(f"/api/transform/{sid}/history").json()["total_operations"] == 0
        assert api.post(f"/api/transform/{sid}/undo").status_code == 404      # no snapshot was taken either

    def test_options_cannot_override_the_validated_transform_or_column(self, api):
        sid = upload(api, pd.DataFrame({"Age": [10.0, 20.0, 30.0]}))
        r = _transform(api, sid, transform_type="minmax", column="Age",
                       options={"transform_type": "formula", "column": "Nope"})
        assert r.status_code == 200 and r.json()["transform_type"] == "minmax"

    def test_unusable_options_give_a_400_not_a_500(self, api):
        sid = upload(api, pd.DataFrame({"Age": [10.0, 20.0, 30.0]}))
        before = _rows(api, sid)
        r = _transform(api, sid, transform_type="equal_width", column="Age", options={"bins": 0})
        assert r.status_code == 400 and r.json()["error"].startswith("The transform could not be applied")
        assert _rows(api, sid) == before

    def test_excluding_a_column_from_ml_changes_no_data_and_can_be_undone(self, api):
        sid = upload(api, pd.DataFrame({"ID": [1, 2, 3], "v": [1.0, 2.0, 3.0]}))
        before = _rows(api, sid)
        r = _transform(api, sid, transform_type="exclude_ml", column="ID")
        assert r.status_code == 200 and r.json()["new_columns"] == [] and "excluded from ML" in r.json()["note"]
        assert _rows(api, sid) == before
        assert api.get(f"/api/transform/{sid}/history").json()["total_operations"] == 1
        assert api.post(f"/api/transform/{sid}/undo").status_code == 200
        assert api.get(f"/api/transform/{sid}/history").json()["total_operations"] == 0

    def test_transform_log_line_has_rows_and_columns(self, api, caplog):
        sid = upload(api, pd.DataFrame({"G": ["a", "b", "a"], "v": [1.0, 2.0, 3.0]}))
        with caplog.at_level(logging.INFO):
            _transform(api, sid, transform_type="onehot", column="G")
        assert any("Transform applied" in r.getMessage() and "rows 3→3" in r.getMessage() and "cols 2→3" in r.getMessage()
                   for r in caplog.records)

    def test_reset_removes_a_generated_report(self, api):
        sid = upload(api, NUMBERS)
        assert api.post(f"/api/report/{sid}/generate", json={"sections": ["appendix"], "format": "html"}).status_code == 200
        assert api.get(f"/api/report/{sid}/download/html").status_code == 200
        api.post(f"/api/dataset/{sid}/reset")
        assert api.get(f"/api/report/{sid}/download/html").status_code == 404

    def test_a_wide_dataset_reports_the_correlation_cap(self, api):
        rng = np.random.default_rng(0)
        wide = pd.DataFrame(rng.normal(size=(40, 25)), columns=[f"c{i}" for i in range(25)])
        sid = upload(api, wide)
        r = api.post(f"/api/eda/{sid}/analyze", json={"analysis_type": "multivariate"})
        result = r.json()["result"]
        assert r.status_code == 200
        assert result["n_columns"] == 20 and result["columns_capped"] is True
        assert result["warning"] == "Showing the first 20 of 25 numerical columns."

    def test_eda_analyze_runs_in_the_thread_pool(self):
        from routers import eda
        assert not inspect.iscoroutinefunction(eda.analyze)

    async def test_profiling_does_not_block_the_event_loop(self, api, monkeypatch):
        from main import app
        from services import profile_service
        sid = upload(api, NUMBERS)
        original = profile_service.identifier_detection
        monkeypatch.setattr(profile_service, "identifier_detection", lambda df: (time.sleep(0.8), original(df))[1])
        response, ticks, total, health_after = await measure(app, lambda ac: ac.post(f"/api/profile/{sid}/run"))
        assert response.status_code == 200 and total >= 0.7
        assert loop_is_free(ticks, total, health_after), (ticks, total, health_after)


# ═══════════════════════════════════════════════════════════════
# J. Gemini call sites: the nine replaced helpers
# ═══════════════════════════════════════════════════════════════

def _eda_frame():
    return pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": ["x", "y", "x", "z"]})


_ML_PROBLEM = {"problem_type": "binary_classification", "target": "Churn", "n_classes": 2}
_ML_MODELS = [
    {"name": "A", "score": 90, "reasons": ["r1", "r2", "r3"], "complexity": "low"},
    {"name": "B", "score": 80, "reasons": [], "complexity": "medium"},
]
_ML_CONTEXT = dict(n_rows=100, n_features=5, n_numeric=3, n_categorical=2, missing_pct=1.0,
                   skewed_cols=[], outlier_cols=["x"], multicollinear=False)

CALL_SITES = [
    pytest.param(
        "services.profile_service", "_generate_gemini_insights",
        lambda: ({"dimensions": {"rows": 4}, "missing_values": {"A": {"pct": 25.0}}, "outlier_detection": {"A": {"outlier_count": 0}}},),
        {"insights": ["a", "b", "c"]}, lambda r: r == ["a", "b", "c"], lambda r: r is None, id="profile"),
    pytest.param(
        "services.clean_service", "_get_gemini_explanation",
        lambda: ("missing_numerical", "Age", "median", {"missing_pct": 12}),
        {"explanation": "Median is robust to the skew in Age."}, lambda r: r == "Median is robust to the skew in Age.",
        lambda r: r is None, id="clean"),
    pytest.param(
        "services.transform_service", "_get_gemini_explanation",
        lambda: ("log", "Income", {"skewness": 3.1}),
        {"explanation": "Log compresses the long right tail."}, lambda r: r == "Log compresses the long right tail.",
        lambda r: r is None, id="transform"),
    pytest.param(
        "services.eda_service", "generate_eda_insights", lambda: (_eda_frame(),),
        {"insights": ["i1", "i2", "i3"]}, lambda r: r == {"insights": ["i1", "i2", "i3"], "source": "gemini"},
        lambda r: r["source"] == "rule_based", id="eda-insights"),
    pytest.param(
        "services.eda_service", "answer_question", lambda: (_eda_frame(), "what is the trend?"),
        {"answer": "Sales rise.", "suggested_analysis": "time_series"},
        lambda r: r["source"] == "gemini" and r["answer"] == "Sales rise.",
        lambda r: r["source"] == "rule_based", id="eda-question"),
    pytest.param(
        "services.stats_service", "_get_gemini_explanation",
        lambda: ("independent_t", {"statistic": 2.1, "p_value": 0.03, "effect_size": {"label": "small"}}),
        {"plain_english": "The groups differ."}, lambda r: r == "The groups differ.", lambda r: r is None, id="stats"),
    pytest.param(
        "services.nlp_service", "_get_gemini_nlp_insights", lambda: ({"column": "Review"},),
        {"insights": ["a", "b"]}, lambda r: r == ["a", "b"], lambda r: r is None, id="nlp"),
    pytest.param(
        "services.ml_service", "generate_reasoning", lambda: (_ML_PROBLEM, _ML_MODELS, "predict churn", _ML_CONTEXT, {}),
        {"summary": "A long enough summary sentence for the check.", "model_reasons": {"A": "gem A"}},
        lambda r: r["source"] == "gemini" and r["model_reasons"]["A"] == "gem A" and "B" in r["model_reasons"],
        lambda r: r["source"] == "rule_based", id="ml"),
    pytest.param(
        "services.report_service", "_gemini_summary", lambda: ({"rows": 10},),
        {"summary": "A sufficiently long executive summary text here.", "highlights": ["h1"]},
        lambda r: r["source"] == "gemini" and r["highlights"] == ["h1"], lambda r: r is None, id="report"),
]


@pytest.mark.parametrize("module, name, args, reply, expect_gemini, expect_fallback", CALL_SITES)
class TestCallSites:
    async def test_uses_the_gemini_reply_and_fills_every_placeholder(
        self, monkeypatch, module, name, args, reply, expect_gemini, expect_fallback
    ):
        mod = importlib.import_module(module)
        seen = {}

        async def fake_generate_json(prompt):
            seen["prompt"] = prompt
            return reply

        monkeypatch.setattr(mod, "generate_json", fake_generate_json)    # fails if the import line is missing
        result = await getattr(mod, name)(*args())
        assert expect_gemini(result)
        assert "{{" not in seen["prompt"]          # every placeholder in the prompt file was filled

    async def test_falls_back_when_gemini_returns_nothing(
        self, monkeypatch, module, name, args, reply, expect_gemini, expect_fallback
    ):
        mod = importlib.import_module(module)

        async def nothing(prompt):
            return None

        monkeypatch.setattr(mod, "generate_json", nothing)
        assert expect_fallback(await getattr(mod, name)(*args()))

    async def test_falls_back_when_gemini_raises_unexpectedly(
        self, monkeypatch, module, name, args, reply, expect_gemini, expect_fallback
    ):
        mod = importlib.import_module(module)

        async def explode(prompt):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(mod, "generate_json", explode)
        assert expect_fallback(await getattr(mod, name)(*args()))

    async def test_works_without_an_api_key(
        self, monkeypatch, module, name, args, reply, expect_gemini, expect_fallback
    ):
        gc.reset_gemini_state()
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
        mod = importlib.import_module(module)
        started = time.perf_counter()
        assert expect_fallback(await getattr(mod, name)(*args()))
        assert time.perf_counter() - started < 5