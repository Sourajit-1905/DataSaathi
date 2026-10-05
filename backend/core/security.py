import re
import logging

logger = logging.getLogger(__name__)

# Allowed MIME types for upload
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
}

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# UUID v4 pattern
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Disallowed keywords in formula expressions
FORMULA_BLOCKLIST = re.compile(
    r"\b(import|os|sys|open|exec|eval|__import__|__builtins__|subprocess|shutil|"
    r"pathlib|glob|socket|urllib|requests|http|__class__|__dict__|__module__|"
    r"__bases__|getattr|setattr|delattr|vars|dir|locals|globals)\b",
    re.IGNORECASE,
)

# Only allow formula expressions that reference column names + basic math
# (digits, operators, column names with spaces in backticks, parens, whitespace)
FORMULA_SAFE_PATTERN = re.compile(
    r"^[@A-Za-z0-9_`\s\+\-\*\/\(\)\.\,\>\<\=\!\&\|\^%]+$"
)


def validate_session_id(session_id: str) -> bool:
    """Return True if session_id is a valid UUID v4."""
    return bool(UUID_PATTERN.match(session_id))


def validate_mime_type(content_type: str | None, filename: str) -> bool:
    """
    Return True if the file is allowed.
    Checks both MIME type (when provided) and file extension.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False
    # If content_type is provided, also check it (but don't fail on None —
    # some clients send octet-stream for CSV)
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        # Allow application/octet-stream as a fallback; extension check is the real guard
        if content_type != "application/octet-stream":
            logger.warning(f"Unexpected MIME type '{content_type}' for file '{filename}'")
    return True


def validate_formula(formula: str, existing_columns: list[str]) -> tuple[bool, str]:
    """
    Validate a pandas df.eval() formula for safety.
    Returns (is_valid, error_message).
    """
    if FORMULA_BLOCKLIST.search(formula):
        return False, "Formula contains disallowed keywords."

    if not FORMULA_SAFE_PATTERN.match(formula):
        return False, "Formula contains disallowed characters."

    # Check that formula only references known column names
    # Extract potential column references (backtick-wrapped or bare words)
    backtick_cols = re.findall(r"`([^`]+)`", formula)
    bare_words = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", formula)

    referenced = set(backtick_cols + bare_words)
    # Remove math function names that pandas eval supports
    math_funcs = {"abs", "sqrt", "log", "exp", "sin", "cos", "tan"}
    referenced -= math_funcs

    unknown = [r for r in referenced if r not in existing_columns]
    if unknown:
        return False, f"Formula references unknown columns: {unknown}"

    return True, ""
