from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Server
    BACKEND_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS
    CORS_ORIGIN: str = "http://localhost:5173"

    # File Upload
    MAX_UPLOAD_MB: int = 50
    UPLOAD_DIR: str = "uploads"

    # Gemini AI

    GEMINI_TIMEOUT_SECONDS: int = 15
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_MAX_RETRIES: int = 3


    SESSION_TTL_HOURS: int = 24

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        return Path(self.UPLOAD_DIR)

    @property
    def original_path(self) -> Path:
        return self.upload_path / "original"

    @property
    def working_path(self) -> Path:
        return self.upload_path / "working"

    @property
    def history_path(self) -> Path:
        return self.upload_path / "history"

    @property
    def snapshots_path(self) -> Path:
        return self.upload_path / "snapshots"

    @property
    def reports_path(self) -> Path:
        return self.upload_path / "reports"

    @property
    def chat_path(self) -> Path:
        return self.upload_path / "chat"



settings = Settings()
