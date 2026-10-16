"""
Middleware and response classes
-------------------------------
RequestContextMiddleware is a plain ASGI middleware that, for every request:

  - gives it an ID (or accepts a sane one from the X-Request-ID header) and
    makes it available as request.state.request_id
  - adds the request ID and security headers to the response
  - logs one line: method, path, status, duration, request ID
  - turns an unhandled exception into the standard 500 response. It does this
    here, inside the CORS layer, because Starlette's own last-resort handler
    runs outside it: that response would reach the browser without CORS
    headers, and the frontend would show "Network error" instead of our message.

NaNSafeJSONResponse is the default response class: pandas and numpy produce NaN
and Infinity, which are not valid JSON, so they are sent as null.
"""

import json
import logging
import math
import re
import time
import uuid
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from core.exceptions import error_response, generic_exception_handler

logger = logging.getLogger(__name__)

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{8,64}")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _incoming_request_id(scope) -> str | None:
    for key, value in scope.get("headers", []):
        if key == b"x-request-id":
            candidate = value.decode("latin-1")
            return candidate if _SAFE_REQUEST_ID.fullmatch(candidate) else None
    return None


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = request_id

        outcome = {"status": None}
        started_at = time.perf_counter()

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                outcome["status"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for name, value in SECURITY_HEADERS.items():
                    if name not in headers:
                        headers[name] = value
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception as exc:
            if outcome["status"] is not None:
                raise  # the response has already started; nothing sensible to send
            response = await generic_exception_handler(Request(scope, receive), exc)
            await response(scope, receive, send_with_headers)
        finally:
            self._log(scope, outcome["status"] or 500, started_at, request_id)

    @staticmethod
    def _log(scope, status: int, started_at: float, request_id: str) -> None:
        path = _CONTROL_CHARS.sub("?", scope.get("path", ""))[:200]   # no log forging
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if path == "/api/health":
            level = logging.DEBUG
        elif status >= 500:
            level = logging.WARNING
        else:
            level = logging.INFO
        logger.log(
            level, "%s %s -> %s | %.0fms | request_id=%s",
            scope.get("method", "?"), path, status, elapsed_ms, request_id,
        )


def _replace_nan(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _replace_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_replace_nan(v) for v in obj]
    return obj


class NaNSafeJSONResponse(JSONResponse):
    """Like JSONResponse, but NaN and Infinity become null instead of an error."""

    def render(self, content) -> bytes:
        return json.dumps(
            _replace_nan(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    
class BodyLimitMiddleware:
    """
    Reject oversized request bodies before they are parsed.

    FastAPI receives and spools a whole multipart upload to a temporary file
    before the endpoint runs, so checking the size inside the endpoint cannot
    stop an enormous upload from filling the disk first. This runs earlier:

      - if Content-Length already exceeds the limit, answer 413 without reading
        the body at all;
      - otherwise count the bytes as they arrive and raise a 413 the moment the
        limit is crossed (this also covers chunked uploads with no Content-Length).

    The upload path gets its own, larger limit; everything else is small JSON.
    """

    def __init__(self, app, default_limit: int, upload_path: str, upload_limit: int, upload_message: str):
        self.app = app
        self.default_limit = default_limit
        self.upload_path = upload_path
        self.upload_limit = upload_limit
        self.upload_message = upload_message

    @staticmethod
    def _declared_length(scope) -> int | None:
        for key, value in scope.get("headers", []):
            if key == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return

        if scope.get("path") == self.upload_path:
            limit, message = self.upload_limit, self.upload_message
        else:
            limit = self.default_limit
            message = f"The request is too large. The maximum size is {limit // (1024 * 1024)} MB."

        declared = self._declared_length(scope)
        if declared is not None and declared > limit:
            response = error_response(Request(scope, receive), 413, message)
            await response(scope, receive, send)
            return

        received = 0

        async def counting_receive():
            nonlocal received
            event = await receive()
            if event["type"] == "http.request":
                received += len(event.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=message)
            return event

        await self.app(scope, counting_receive, send)

