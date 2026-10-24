"""
Security tests (Stage 10).

Covers the helpers in core/security.py (session IDs, file names, upload content,
zip safety, formula validation, path and secret scrubbing) and checks that error
responses never expose paths, secrets or tracebacks.

"""

import io
import time
import uuid
import zipfile

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from core import security
from core.config import settings
from core.exceptions import register_exception_handlers
from core.middleware import RequestContextMiddleware


# Session IDs

def test_session_id_accepts_uuid4():
    assert security.validate_session_id(str(uuid.uuid4()))


def test_session_id_accepts_uppercase_uuid4():
    assert security.validate_session_id(str(uuid.uuid4()).upper())


@pytest.mark.parametrize("bad", [
    "not-a-uuid",
    "",
    None,
    123,
    str(uuid.uuid1()),                       # wrong version
    "../../etc/passwd",
    "..\\windows\\system32",
    str(uuid.uuid4()) + "\n",                # `$` would accept this; fullmatch must not
    str(uuid.uuid4()) + "/../x",
    " " + str(uuid.uuid4()),
])
def test_session_id_rejects(bad):
    assert security.validate_session_id(bad) is False


# File names

@pytest.mark.parametrize("raw, expected", [
    ("report 2024.csv", "report 2024.csv"),
    ("C:\\Users\\Biswajit\\Desktop\\data.csv", "data.csv"),
    ("C:data.csv", "data.csv"),
    ("../../etc/passwd", "passwd"),
    ("a/b/c.csv", "c.csv"),
    ("evil\u202efdp.csv", "evilfdp.csv"),     # right-to-left override removed
    ("a\x00b\nc.csv", "abc.csv"),             # control characters removed
    ("", "upload"),
    (None, "upload"),
    ("...", "upload"),
    ("/", "upload"),
])
def test_sanitize_filename(raw, expected):
    assert security.sanitize_filename(raw) == expected


def test_sanitize_filename_bounds_length_and_keeps_extension():
    name = security.sanitize_filename("a" * 300 + ".csv")
    assert len(name) <= security.MAX_FILENAME_LENGTH
    assert name.endswith(".csv")


@pytest.mark.parametrize("filename, ext", [
    ("A.XLSX", ".xlsx"), ("data.csv", ".csv"), ("archive.tar.gz", ".gz"), ("noext", ""), (None, ""), ("", ""),
])
def test_get_extension(filename, ext):
    assert security.get_extension(filename) == ext


# MIME types

@pytest.mark.parametrize("mime, filename", [
    ("text/csv", "a.csv"),
    ("application/vnd.ms-excel", "a.csv"),             # Chrome on Windows with Excel installed
    ("application/octet-stream", "a.csv"),
    ("text/csv; charset=utf-8", "a.csv"),               # parameters are ignored
    (None, "a.csv"),                                    # some clients omit it
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "a.xlsx"),
    ("application/vnd.ms-excel", "a.xls"),
])
def test_mime_accepted(mime, filename):
    assert security.validate_mime_type(mime, filename) is True


@pytest.mark.parametrize("mime, filename", [
    ("application/json", "a.csv"),
    ("image/png", "a.xlsx"),
    ("text/html", "a.csv"),
    ("text/csv", "a.json"),                             # extension not supported
    ("text/csv", "noext"),
])
def test_mime_rejected(mime, filename):
    assert security.validate_mime_type(mime, filename) is False


# Upload content (file signatures)

def make_xlsx(names=("[Content_Types].xml", "xl/workbook.xml")) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, "<x/>")
    return buf.getvalue()


XLS = bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 100


def test_content_csv_ok():
    assert security.validate_upload_content(b"a,b\n1,2\n", ".csv") == ("csv", "")


def test_content_xlsx_ok():
    assert security.validate_upload_content(make_xlsx()[:8192], ".xlsx") == ("xlsx", "")


def test_content_xls_ok():
    assert security.validate_upload_content(XLS, ".xls") == ("xls", "")


def test_content_xlsx_saved_with_xls_extension_is_parsed_as_xlsx():
    kind, error = security.validate_upload_content(make_xlsx()[:8192], ".xls")
    assert kind == "xlsx" and error == ""


@pytest.mark.parametrize("head, ext, fragment", [
    (b"", ".csv", "empty"),
    (make_xlsx()[:8192], ".csv", "named .csv"),
    (XLS, ".csv", "named .csv"),
    (XLS, ".xlsx", "password"),                         # encrypted xlsx is an OLE container too
    (b"a,b\n1,2\n", ".xlsx", "real .xlsx"),
    (b"a,b\n1,2\n", ".xls", "real .xls"),
    (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", ".csv", "binary"),
    ("a,b\n1,2\n".encode("utf-16"), ".csv", "utf-16"),
])
def test_content_rejected(head, ext, fragment):
    kind, error = security.validate_upload_content(head, ext)
    assert kind is None
    assert fragment in error.lower()


# .xlsx zip safety

def test_zip_ok():
    assert security.check_xlsx_zip(make_xlsx()) == (True, "")


def test_zip_without_workbook_is_rejected():
    ok, message = security.check_xlsx_zip(make_xlsx(("word/document.xml",)))
    assert not ok and "not an excel workbook" in message.lower()


def test_zip_corrupt_is_rejected():
    ok, message = security.check_xlsx_zip(b"PK\x03\x04" + b"junk" * 50)
    assert not ok and "corrupt" in message.lower()


def test_zip_bomb_is_rejected(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", b"0" * (2 * 1024 * 1024))   # tiny when compressed
    monkeypatch.setattr(security, "MAX_XLSX_UNCOMPRESSED_BYTES", 1024 * 1024)
    ok, message = security.check_xlsx_zip(buf.getvalue())
    assert not ok and "expands" in message


def test_zip_with_too_many_entries_is_rejected(monkeypatch):
    monkeypatch.setattr(security, "MAX_XLSX_ENTRIES", 2)
    ok, message = security.check_xlsx_zip(make_xlsx(("a", "b", "xl/workbook.xml")))
    assert not ok and "too many" in message


def test_zip_encrypted_is_rejected():
    raw = bytearray(make_xlsx())
    raw[raw.index(b"PK\x01\x02") + 8] |= 0x1        # set the "encrypted" flag in the central directory
    ok, message = security.check_xlsx_zip(bytes(raw))
    assert not ok and "password" in message


# Formula validation

COLUMNS = ["Age", "Score", "Open", "Close", "Total Sales", "price", "log"]


@pytest.mark.parametrize("formula", [
    "Age + Score",
    "`Total Sales` * 2",                 # a column name with a space
    "Open - Close",                      # `open` is only blocked when it is not a column
    "Age > 30 and Score < 5",
    "not (Age > 30)",
    "abs(Age - 30)",
    "sqrt(price)",
    "log1p(price)",
    "Age ** 2 + Score ** 2",
    "Age ** Score",
    "Age / (Score + 1)",
    "Age == 30",
    "Age % 7",
    "1.5e3 * Age",
    "Age * -1",
    "log",                               # a column that shares a function's name
])
def test_formula_accepted(formula):
    ok, message = security.validate_formula(formula, COLUMNS)
    assert ok, message


@pytest.mark.parametrize("formula, fragment", [
    ("import os", "disallowed keyword"),
    ("eval(Age)", "disallowed keyword"),
    ("open(Age)", "disallowed keyword"),
    ("__import__('os')", "double underscore"),
    ("__class__", "double underscore"),
    ("Age; DROP", "disallowed characters"),
    ("@x", "disallowed characters"),
    ("os.system('x')", "disallowed characters"),
    ("lambda x: x", "disallowed characters"),
    ("Age[0]", "disallowed characters"),
    ("Age = 5", "assignment"),
    ("Age => 5", "assignment"),
    ("9**9**9", "chained"),
    ("Age ** 2 ** 2", "chained"),
    ("(((9**9)**9)**9)**9**9", "chained"),
    ("Age ** 11", "too large"),
    ("Age ** (2)", "exponent must be"),
    ("Age ** 1 ** 1 ** 1 ** 1 ** 1 ** 1", "too many powers"),
    ("Age.mean()", "attribute"),
    ("Age.T", "attribute"),
    ("unknown + 1", "unknown column or function"),
    ("Age if Score else 1", "unknown column or function"),
    ("`Missing` + 1", "unknown column"),
    ("`Age + 1", "unbalanced"),
    ("Age +\nScore", "single expression"),
    ("", "empty"),
    ("x" * 300, "too long"),
    ("Age == == 5", "not a valid expression"),
])
def test_formula_rejected_with_a_reason(formula, fragment):
    ok, message = security.validate_formula(formula, COLUMNS)
    assert not ok
    assert fragment in message.lower()


def test_formula_must_be_text():
    assert security.validate_formula(None, COLUMNS)[0] is False
    assert security.validate_formula(123, COLUMNS)[0] is False


def test_formula_blocked_word_allowed_when_it_is_a_column():
    assert security.validate_formula("open + 1", ["open"])[0] is True
    assert security.validate_formula("open + 1", ["Age"])[0] is False


# Path scrubbing

@pytest.mark.parametrize("raw, expected", [
    (r"Cannot read C:\Users\Biswajit\Desktop\SP49\models\x.bin now", "Cannot read [path] now"),
    (r"Missing C:\Program Files\App\lib\y.py: boom", "Missing [path]: boom"),
    ("No such file: '/home/user/app/uploads/x.csv'", "No such file: '[path]'"),
    ("open uploads/working/abc.csv failed", "open [path] failed"),
    ("see ./uploads/original/abc.csv", "see [path]"),
    ("cache at ~/.cache/huggingface/hub", "cache at [path]"),
    (r"share \\server\share\folder\f.txt used", "share [path] used"),
    ('File "/usr/lib/python3/site-packages/a.py", line 3', 'File "[path]", line 3'),
])
def test_scrub_paths_removes_paths(raw, expected):
    assert security.scrub_paths(raw) == expected


@pytest.mark.parametrize("text", [
    "https://example.com/home/page",
    "http://localhost:8000/api/dataset/x",
    "/api/dataset/abc/upload",
    "Train/test split",
    "and/or",
    "cardinality/type",
    "Valid types: ['a', 'b']",
    "/homework is due",
])
def test_scrub_paths_leaves_ordinary_text_alone(text):
    assert security.scrub_paths(text) == text


def test_scrub_paths_has_no_catastrophic_backtracking():
    started = time.time()
    security.scrub_paths("C:\\" + "a " * 1500)
    security.safe_detail("C:\\" + "word " * 2000 + "/")
    assert time.time() - started < 2


# Secret scrubbing

GOOGLE_KEY = "AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"


def test_scrub_secrets_google_key():
    assert "AIzaSy" not in security.scrub_secrets(f"key {GOOGLE_KEY} leaked")


def test_scrub_secrets_url_parameter():
    assert "SECRET" not in security.scrub_secrets("GET /v1/models?key=SECRET&alt=json")


def test_scrub_secrets_env_style_and_bearer():
    assert "abc123" not in security.scrub_secrets("GEMINI_API_KEY=abc123 loaded")
    assert "tok123" not in security.scrub_secrets("Authorization: Bearer tok123")


def test_scrub_secrets_configured_key(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "my-very-secret-key-123")
    assert "my-very-secret" not in security.scrub_secrets("the SDK said: bad key my-very-secret-key-123")


def test_scrub_secrets_leaves_ordinary_text_alone():
    assert security.scrub_secrets("Column 'Age' not found") == "Column 'Age' not found"


# safe_detail

def test_safe_detail_hides_tracebacks():
    assert security.safe_detail("Traceback (most recent call last):\n File x") == security.GENERIC_ERROR_DETAIL


def test_safe_detail_is_one_line_and_bounded():
    assert "\n" not in security.safe_detail("line1\n\n  line2")
    assert len(security.safe_detail("x " * 2000)) <= 500


def test_safe_detail_empty_input_gets_a_generic_message():
    assert security.safe_detail("") == security.GENERIC_ERROR_DETAIL
    assert security.safe_detail(None) == security.GENERIC_ERROR_DETAIL


def test_safe_detail_keeps_long_lists_of_valid_values():
    message = "Unknown. Valid types: " + ", ".join(f"t{i}" for i in range(40))
    assert "Valid types" in security.safe_detail(message)


def test_safe_detail_removes_paths_and_keys():
    out = security.safe_detail(rf"Failed: C:\Users\Biswajit\.cache\hf\model.bin with key={GOOGLE_KEY}")
    assert "Biswajit" not in out and "AIzaSy" not in out


# Error responses never expose paths, secrets or tracebacks

LEAKS = ["Biswajit", "AIza", "Traceback", "kaboom", "/home/app", "site-packages", "RuntimeError"]


class _Body(BaseModel):
    column: str
    top_n: int = 5


def _error_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    @app.get("/path-in-http-error")
    def path_in_http_error():
        raise HTTPException(status_code=400, detail=r"Cannot open C:\Users\Biswajit\Desktop\uploads\working\x.csv")

    @app.get("/boom")
    def boom():
        raise RuntimeError(rf"kaboom at C:\Users\Biswajit\x.py and /home/app/y.py with key={GOOGLE_KEY}")

    @app.post("/body")
    def body(b: _Body):
        return {"ok": True}

    return app


@pytest.fixture(scope="module")
def error_client():
    return TestClient(_error_app(), raise_server_exceptions=False)


def test_http_error_detail_is_scrubbed(error_client):
    r = error_client.get("/path-in-http-error")
    assert r.status_code == 400
    assert "Biswajit" not in r.text and "[path]" in r.text


def test_unhandled_error_is_generic_and_leaks_nothing(error_client):
    r = error_client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "An unexpected error occurred. Please try again."
    assert body["detail"] == f"Reference: {body['request_id']}"
    assert not any(leak in r.text for leak in LEAKS)


def test_validation_error_does_not_echo_the_input(error_client):
    r = error_client.post("/body", json={"top_n": "SECRET-VALUE-123"})
    assert r.status_code == 422
    assert "SECRET-VALUE-123" not in r.text
    assert {f["field"] for f in r.json()["fields"]} >= {"column", "top_n"}


@pytest.mark.parametrize("method, path", [("get", "/no/such/route"), ("delete", "/body")])
def test_framework_errors_use_the_standard_shape(error_client, method, path):
    r = getattr(error_client, method)(path)
    assert r.status_code in (404, 405)
    assert {"error", "detail", "code", "request_id"} <= set(r.json())