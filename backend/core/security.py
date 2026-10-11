"""
Security helpers
----------------
Everything that decides whether outside input is safe lives here, so it can be
audited in one place.

  Session IDs   validate_session_id
  Uploads       get_extension, sanitize_filename, validate_mime_type,
                validate_upload_content, check_xlsx_zip
  Formulas      validate_formula
  Error output  scrub_paths, scrub_secrets, safe_detail

Nothing in this module executes user input.
"""

import ast
import io
import logging
import re
import zipfile

from core.config import settings

logger = logging.getLogger(__name__)


# Constants

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# The file signature (see validate_upload_content) is the real guard. The MIME
# type is only a first filter, so it has to accept what real clients send:
# Chrome on Windows with Excel installed reports .csv files as
# application/vnd.ms-excel, and some clients send application/octet-stream.
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",                                           # .xls, and CSVs from Windows
    "application/octet-stream",                                           # generic fallback
}

SNIFF_BYTES = 8192                      # how much of a file to inspect
UPLOAD_CHUNK_BYTES = 1024 * 1024        # chunk size when reading an upload
MAX_FILENAME_LENGTH = 120
MAX_XLSX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024   # zip-bomb guard
MAX_XLSX_ENTRIES = 2000

GENERIC_ERROR_DETAIL = "An unexpected error occurred. Please try again."

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# Session IDs

def validate_session_id(session_id) -> bool:
    """
    True only for a canonical UUID v4 string. Session IDs become file names, so
    this is also what prevents path traversal. fullmatch (unlike `$`) rejects
    a trailing newline.
    """
    return isinstance(session_id, str) and bool(UUID_PATTERN.fullmatch(session_id))


# Uploads

_FILENAME_JUNK = re.compile(r"[\x00-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def get_extension(filename: str | None) -> str:
    """
    Lower-cased extension including the dot, or '' if there is none.
    """
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def sanitize_filename(name, default: str = "upload") -> str:
    """
    Reduce a client-supplied file name to a safe base name: no directories
    (Windows or POSIX), no drive letter, no control or direction-override
    characters, bounded length. The name is only ever echoed back or logged;
    it is never used as a path.
    """
    if not isinstance(name, str) or not name:
        return default

    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"^[A-Za-z]:", "", name)
    name = _FILENAME_JUNK.sub("", name).strip().strip(".").strip()

    if len(name) > MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) <= 10: name = stem[: MAX_FILENAME_LENGTH - len(ext) - 1] + "." + ext
        else: name = name[:MAX_FILENAME_LENGTH]

    return name or default


def normalise_mime(content_type: str | None) -> str:
    """
    'text/csv; charset=utf-8' -> 'text/csv'.
    """
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def validate_mime_type(content_type: str | None, filename: str) -> bool:
    """
    First-line upload filter: the extension must be supported and, when the
    client sends a MIME type, it must be on the allowlist. A missing MIME type
    is accepted because some clients omit it; validate_upload_content decides.
    """
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return False

    mime = normalise_mime(content_type)
    if not mime:
        return True
    if mime in ALLOWED_MIME_TYPES:
        return True

    logger.warning("Upload rejected: MIME type %r is not allowed for %r files", mime[:100], ext)
    return False


_XLSX_SIGNATURE = b"PK\x03\x04"                  # zip container
_XLS_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")   # OLE2 compound file


def validate_upload_content(head: bytes, ext: str) -> tuple[str | None, str]:
    """
    Decide what a file really is from its first bytes and check that it fits
    its extension. Returns (kind, error): kind is "csv", "xlsx" or "xls", and
    error is "" when the file is acceptable.

    An .xls that is really an .xlsx is accepted and reported as "xlsx", because
    Excel exports produce these often; the caller should parse by kind.
    """
    if not head:
        return None, "The uploaded file is empty."

    if head.startswith(_XLSX_SIGNATURE):
        if ext in (".xlsx", ".xls"):
            return "xlsx", ""
        return None, "This file is an Excel workbook but is named .csv. Rename it to .xlsx."

    if head.startswith(_XLS_SIGNATURE):
        if ext == ".xls":
            return "xls", ""
        if ext == ".xlsx":
            return None, (
                "This workbook is in the old .xls format or is password-protected. "
                "Re-save it as .xlsx without a password."
            )
        return None, "This file is an Excel workbook but is named .csv. Rename it to .xls."

    # No workbook signature, so it can only be a CSV.
    if ext != ".csv":
        return None, f"The file does not look like a real {ext} workbook."
    if b"\x00" in head:
        return None, "The file looks binary or UTF-16 encoded. Save it as a UTF-8 CSV and try again."
    return "csv", ""


def check_xlsx_zip(data: bytes) -> tuple[bool, str]:
    """
    Zip-bomb and sanity check for .xlsx files, run before the workbook is
    parsed. A small upload can expand to gigabytes of XML. Python's zipfile
    never returns more bytes than an entry's declared size, so checking the
    declared sizes is an effective bound for the parser too.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError):
        return False, "The workbook is corrupted or is not a valid .xlsx file."

    if len(infos) > MAX_XLSX_ENTRIES:
        return False, "The workbook contains too many internal files."
    if any(i.flag_bits & 0x1 for i in infos):
        return False, "The workbook is password-protected. Remove the password and try again."

    total = sum(i.file_size for i in infos)
    if total > MAX_XLSX_UNCOMPRESSED_BYTES:
        limit_mb = MAX_XLSX_UNCOMPRESSED_BYTES // (1024 * 1024)
        return False, f"The workbook expands to more than {limit_mb} MB, which is too large to process."

    if "xl/workbook.xml" not in {i.filename for i in infos}:
        return False, "The file is a zip archive but not an Excel workbook."

    return True, ""


# Formulas  (used by the Transform stage's custom-formula column)

MAX_FORMULA_LENGTH = 200
MAX_POWER_OPERATORS = 5
MAX_EXPONENT = 10

ALLOWED_FORMULA_FUNCTIONS = frozenset({
    "abs", "sqrt", "log", "log10", "log1p", "exp", "sin", "cos", "tan", "floor", "ceil",
})
ALLOWED_FORMULA_KEYWORDS = frozenset({"and", "or", "not"})

# Used only to give a clearer message; anything that is not a column, a keyword
# or an allowed function is rejected anyway.
BLOCKED_FORMULA_NAMES = frozenset({
    "import", "os", "sys", "open", "exec", "eval", "compile", "input", "breakpoint",
    "subprocess", "shutil", "pathlib", "glob", "socket", "urllib", "requests", "http",
    "getattr", "setattr", "delattr", "vars", "dir", "locals", "globals", "lambda",
    "builtins",
})

_FORMULA_CHARS = re.compile(r"[A-Za-z0-9_\s+\-*/%().,<>=!&|^]+")
_BACKTICKED = re.compile(r"`([^`]+)`")
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*")
_ATTRIBUTE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|\))\s*\.\s*[A-Za-z_]")
_EXPONENT = re.compile(
    r"\s*[+-]?\s*([A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def validate_formula(formula: str, existing_columns: list[str]) -> tuple[bool, str]:
    """
    Validate a custom-formula expression before it reaches pandas' df.eval().
    Returns (is_valid, error_message).

    This is an allowlist, not a blocklist. A formula may contain only: existing
    column names (bare, or in backticks when they contain spaces), numbers,
    arithmetic / comparison / boolean operators, parentheses, and a few math
    functions. Everything else is rejected, including double underscores,
    assignments, attribute access, '@' references, multiple statements and
    runaway powers such as 9**9**9.
    """
    if not isinstance(formula, str):
        return False, "Formula must be text."
    formula = formula.strip()
    if not formula:
        return False, "Formula is empty."
    if len(formula) > MAX_FORMULA_LENGTH:
        return False, f"Formula is too long (maximum {MAX_FORMULA_LENGTH} characters)."
    if "\n" in formula or "\r" in formula:
        return False, "Formula must be a single expression."
    if "__" in formula:
        return False, "Formula contains disallowed content (double underscore)."

    columns = {str(c) for c in existing_columns}

    # Backticked names must be existing columns; then hide them from the scans
    # below by replacing each with a plain number.
    if formula.count("`") % 2:
        return False, "Formula has an unbalanced backtick."
    for name in _BACKTICKED.findall(formula):
        if name not in columns:
            return False, f"Formula references an unknown column: '{name}'."
    scan = _BACKTICKED.sub(" 0 ", formula)

    if not _FORMULA_CHARS.fullmatch(scan):
        return False, "Formula contains disallowed characters."

    # '==', '<=', '>=', '!=' are fine; a lone '=' would be an assignment.
    if "=" in re.sub(r"[<>!=]=", "", scan):
        return False, "Assignments are not allowed in a formula."

    if _ATTRIBUTE.search(scan):
        return False, "Attribute access is not allowed in a formula."

    # Powers: limit how many, require a plain number or column as the exponent,
    # keep numeric exponents small, and reject chains such as a**b**c.
    if scan.count("**") > MAX_POWER_OPERATORS:
        return False, f"Formula uses too many powers (maximum {MAX_POWER_OPERATORS})."
    for m in re.finditer(r"\*\*", scan):
        rest = scan[m.end():]
        exp = _EXPONENT.match(rest)
        if not exp:
            return False, "A power's exponent must be a number or a column name."
        token = exp.group(1)
        if not (token[0].isalpha() or token[0] == "_"):
            if float(token) > MAX_EXPONENT:
                return False, f"A power's exponent is too large (maximum {MAX_EXPONENT})."
        if re.match(r"\s*\*\*", rest[exp.end():]):
            return False, "Chained powers (a**b**c) are not allowed."

    # Every name must be a column, a boolean keyword, or an allowed function call.
    for m in _IDENTIFIER.finditer(scan):
        token = m.group(0)
        if token in columns or token in ALLOWED_FORMULA_KEYWORDS:
            continue
        if token in ALLOWED_FORMULA_FUNCTIONS and scan[m.end():].lstrip().startswith("("):
            continue
        if token.lower() in BLOCKED_FORMULA_NAMES:
            return False, f"Formula contains a disallowed keyword: '{token}'."
        return False, f"Formula references an unknown column or function: '{token}'."

    # Last: what remains must be one syntactically valid expression. This runs
    # after the checks above so their more specific messages win.
    try:
        ast.parse(scan.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False, "Formula is not a valid expression."

    return True, ""


# Error output: never leak paths, secrets or tracebacks

_SEG = r"[^\\/\s'\"<>|*?:]"   # one path component character

_PATH_PATTERNS = [
    # Windows drive paths. Directory names may contain single spaces
    # ("C:\Program Files\..."); the final component may not.
    re.compile(r"(?<!\w)[A-Za-z]:[\\/](?:" + _SEG + r"+(?: " + _SEG + r"+)*[\\/])*" + _SEG + r"*"),
    # UNC paths
    re.compile(r"(?<![\w\\])\\\\" + _SEG + r"+(?:[\\/]" + _SEG + r"+)*"),
    # POSIX absolute paths under well-known roots (so URL paths like /api/... survive)
    re.compile(
        r"(?<![\w.~:/-])/(?:home|Users|usr|var|tmp|etc|opt|mnt|root|srv|app|proc|"
        r"lib|lib64|bin|sbin|private|Library|Volumes|snap|workspace)"
        r"(?:/[^\s'\"<>:|*?]*)?(?!\w)"
    ),
    # Our own relative storage paths
    re.compile(r"(?<![\w/\\.-])(?:\.{1,2}[\\/])*(?:uploads|prompts|templates)[\\/][^\s'\"<>:|*?]*"),
    # Home-relative paths
    re.compile(r"(?<![\w.])~[\\/][^\s'\"<>:|*?]*"),
]


def scrub_paths(text: str) -> str:
    """Replace filesystem paths with '[path]'. For client-facing text only."""
    if not text:
        return text
    for pattern in _PATH_PATTERNS:
        text = pattern.sub("[path]", text)
    return text


_SECRET_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[redacted]"),                      # Google API keys
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*"), "Bearer [redacted]"),
    (re.compile(r"(?i)([?&]key=)[^&\s\"']+"), r"\1[redacted]"),
    (
        re.compile(
            r"(?i)([A-Za-z0-9_\-]*(?:api[_-]?key|token|secret|password|authorization)"
            r"\s*[=:]\s*)[\"']?[^\s\"'&,;]+"
        ),
        r"\1[redacted]",
    ),
]


def scrub_secrets(text: str) -> str:
    """Redact API keys and tokens, including the configured Gemini key."""
    if not text:
        return text
    key = getattr(settings, "GEMINI_API_KEY", "") or ""
    if len(key) >= 8:
        text = text.replace(key, "[redacted]")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def safe_detail(message, max_len: int = 500) -> str:
    """
    Make an error message safe to send to a client: no tracebacks, no
    filesystem paths, no secrets, one line, bounded length.
    """
    text = "" if message is None else str(message)
    if "Traceback (most recent call last)" in text:
        return GENERIC_ERROR_DETAIL

    text = scrub_secrets(scrub_paths(text[:2000]))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text or GENERIC_ERROR_DETAIL












