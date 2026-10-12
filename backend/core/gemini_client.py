"""
Gemini client
-------------
The only module that talks to Gemini. Services call generate_json() (or
generate_text()) and treat None as "use the rule-based answer".

What it guarantees
  - Never raises. A missing key, a missing SDK, a timeout, a rate limit or bad
    JSON all return None, so the app keeps working without AI.
  - Never blocks the event loop. The SDK call is synchronous, so it runs on a
    small dedicated thread pool.
  - Bounded waiting. Every call has a timeout. Rate limits and server errors
    are retried with backoff (GEMINI_MAX_RETRIES); timeouts are not retried.
  - Fails fast afterwards. After a timeout or exhausted retries the client
    stays quiet for a minute, so later requests fall back instantly instead of
    each waiting on a struggling API.
  - Never logs prompts, and redacts keys from anything it does log.
"""

import asyncio
import importlib.util
import json
import logging
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.config import settings
from core.security import scrub_secrets

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

DEFAULT_TIMEOUT_SECONDS = 15.0     # per attempt; override with GEMINI_TIMEOUT_SECONDS
RETRY_BUDGET_SECONDS = 10.0        # extra time allowed for retries beyond one timeout
TIMEOUT_MARGIN_SECONDS = 2.0       # outer limit = SDK timeout + margin
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 8.0
COOLDOWN_SECONDS = 60
MAX_CONCURRENT_CALLS = 4

_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS, thread_name_prefix="gemini")
_cooldown_until = 0.0
_sdk_missing = False


# Settings & state

def _timeout() -> float:
    try:
        value = float(getattr(settings, "GEMINI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


def _max_retries() -> int:
    try:
        return max(0, int(getattr(settings, "GEMINI_MAX_RETRIES", 3)))
    except (TypeError, ValueError):
        return 3


def _start_cooldown(seconds: float) -> None:
    global _cooldown_until
    _cooldown_until = max(_cooldown_until, time.monotonic() + seconds)


def cooldown_remaining() -> int:
    """Whole seconds until Gemini will be tried again; 0 when available."""
    return max(0, int(math.ceil(_cooldown_until - time.monotonic())))


def reset_gemini_state() -> None:
    """Clear the cooldown and the missing-SDK flag. Used by tests."""
    global _cooldown_until, _sdk_missing
    _cooldown_until = 0.0
    _sdk_missing = False


def _sdk_installed() -> bool:
    try:
        return importlib.util.find_spec("google.genai") is not None
    except (ImportError, ValueError):
        return False


def gemini_status() -> dict:
    """Safe-to-expose status for the UI badge. Never includes the key."""
    remaining = cooldown_remaining()
    return {
        "configured": bool(getattr(settings, "GEMINI_API_KEY", "")),
        "model": getattr(settings, "GEMINI_MODEL", None),
        "sdk_installed": _sdk_installed(),
        "cooling_down": remaining > 0,
        "retry_after_seconds": remaining,
    }


# Prompts

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render_prompt(name: str, **values) -> str | None:
    """
    Load prompts/<name> and fill its {{PLACEHOLDER}} fields. Substitution is a
    single pass, so text inside a value (for example a user's question) is never
    treated as another placeholder. Returns None if the template is missing.
    The path is anchored to the backend folder, not the working directory.
    """
    if "/" in name or "\\" in name:
        return None
    try:
        template = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    except OSError:
        logger.warning("Prompt template not found: %s", name)
        return None
    return _PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), m.group(0))), template)


# The call

def _sync_generate(prompt: str) -> str:
    """Blocking SDK call. Runs on the Gemini thread pool, never on the event loop."""
    from google import genai

    api_key = settings.GEMINI_API_KEY
    try:
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": int(_timeout() * 1000)},   # milliseconds
        )
    except Exception:  # an older SDK without http_options
        client = genai.Client(api_key=api_key)

    response = client.models.generate_content(model=settings.GEMINI_MODEL, contents=prompt)
    return response.text or ""


def _classify(exc: BaseException) -> str:
    """'retry' for rate limits, server errors and network blips; otherwise 'fatal'."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return "retry" if code in (408, 429) or code >= 500 else "fatal"

    text = f"{type(exc).__name__} {exc}".lower()
    markers = ("429", "rate limit", "rate_limit", "resource_exhausted", "quota",
               "overloaded", "unavailable", "502", "503", "504")
    if any(m in text for m in markers) or isinstance(exc, (ConnectionError, OSError)):
        return "retry"
    return "fatal"


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {scrub_secrets(str(exc))[:200]}"


async def generate_text(prompt: str) -> str | None:
    """Return Gemini's text, or None for any reason it could not be obtained."""
    global _sdk_missing

    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if not getattr(settings, "GEMINI_API_KEY", ""):
        return None
    if _sdk_missing:
        return None
    if cooldown_remaining() > 0:
        logger.debug("Gemini is cooling down; using the rule-based fallback.")
        return None

    timeout = _timeout()
    retries = _max_retries()
    deadline = time.monotonic() + timeout + RETRY_BUDGET_SECONDS
    loop = asyncio.get_running_loop()
    attempt = 0

    while True:
        attempt += 1
        left = deadline - time.monotonic()
        if left <= 0:
            break

        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(_EXECUTOR, _sync_generate, prompt),
                timeout=min(timeout + TIMEOUT_MARGIN_SECONDS, left),
            )
            text = (text or "").strip()
            if not text:
                logger.warning("Gemini returned an empty response.")
                return None
            return text

        except asyncio.TimeoutError:
            logger.warning("Gemini call timed out after %.0fs; falling back.", timeout)
            _start_cooldown(COOLDOWN_SECONDS)
            return None

        except ImportError:
            _sdk_missing = True
            logger.error("The google-genai package is not installed. Run: pip install google-genai")
            return None

        except Exception as e:
            if _classify(e) == "retry" and attempt <= retries:
                delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)
                delay += random.uniform(0, delay * 0.25)
                if time.monotonic() + delay < deadline:
                    logger.warning(
                        "Gemini rate-limited or unavailable (attempt %d of %d); retrying in %.1fs. %s",
                        attempt, retries + 1, delay, _describe(e),
                    )
                    await asyncio.sleep(delay)
                    continue

            logger.warning("Gemini call failed: %s", _describe(e))
            if _classify(e) == "retry":
                _start_cooldown(COOLDOWN_SECONDS)
            return None

    _start_cooldown(COOLDOWN_SECONDS)
    return None


# JSON

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def parse_json_response(raw: str) -> dict | None:
    """
    Parse a JSON object out of a model reply. Handles markdown fences and chatter
    before or after the object. Returns None if no object can be parsed.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()

    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None

    return data if isinstance(data, dict) else None


async def generate_json(prompt: str) -> dict | None:
    """Return Gemini's reply parsed as a JSON object, or None."""
    raw = await generate_text(prompt)
    if raw is None:
        return None
    parsed = parse_json_response(raw)
    if parsed is None:
        logger.warning("Gemini returned a reply that is not a valid JSON object.")
    return parsed