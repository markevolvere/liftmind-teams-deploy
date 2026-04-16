"""
LiftMind configuration — Teams-adapted version.
Removes Telegram-specific settings; all others kept intact.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Base directory is the parent of the liftmind/ package (i.e. the app root)
_LIFTMIND_DIR = Path(__file__).parent
_APP_ROOT = _LIFTMIND_DIR.parent


class Settings:
    # ==========================================================================
    # CORE SETTINGS
    # ==========================================================================
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    PRODUCTION_MODE: bool = os.getenv("PRODUCTION_MODE", "false").lower() == "true"

    # ==========================================================================
    # PATHS
    # ==========================================================================
    BASE_DIR: str = str(_APP_ROOT)
    MANUALS_DIR: str = str(_APP_ROOT / "data" / "manuals")
    IMAGES_DIR: str = str(_APP_ROOT / "data" / "images")
    PROMPTS_DIR: str = str(_APP_ROOT / "prompts")

    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "liftmind.log")

    # ==========================================================================
    # CLAUDE MODEL SETTINGS
    # ==========================================================================
    CLAUDE_TIMEOUT: int = 120
    CLAUDE_CLI_INTERCEPTOR_TIMEOUT: int = 35

    # Full model identifiers used by the Anthropic SDK
    CLAUDE_MODEL_HAIKU: str = "claude-haiku-4-5-20251001"
    CLAUDE_MODEL_SONNET: str = "claude-sonnet-4-6"

    # Model routing: map query types to speed/quality trade-off
    CLAUDE_MODEL_ROUTING: dict = {
        "fault_code":    "haiku",   # Fast lookups
        "specification": "haiku",   # Fast spec retrieval
        "procedure":     "sonnet",  # Needs reasoning
        "general":       "sonnet",  # Balanced
    }

    # ==========================================================================
    # RESPONSE CACHE
    # ==========================================================================
    RESPONSE_CACHE_TTL: int = int(os.getenv("RESPONSE_CACHE_TTL", "3600"))
    RESPONSE_CACHE_MAX_SIZE: int = int(os.getenv("RESPONSE_CACHE_MAX_SIZE", "200"))

    # ==========================================================================
    # ML MODEL SETTINGS
    # ==========================================================================
    USE_EMBEDDINGS: bool = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "false").lower() == "true"  # Off by default (large model)

    # ==========================================================================
    # DATABASE POOL SETTINGS
    # ==========================================================================
    DB_POOL_MIN_CONN: int = int(os.getenv("DB_POOL_MIN_CONN", "2"))
    DB_POOL_MAX_CONN: int = int(os.getenv("DB_POOL_MAX_CONN", "10"))
    DB_POOL_RECONNECT_RETRIES: int = int(os.getenv("DB_POOL_RECONNECT_RETRIES", "3"))

    # ==========================================================================
    # RAG PIPELINE SETTINGS
    # ==========================================================================
    RRF_K_PARAMETER: int = int(os.getenv("RRF_K_PARAMETER", "60"))

    RRF_K_BY_QUERY_TYPE: dict = {
        "fault_code":    25,
        "specification": 35,
        "procedure":     50,
        "general":       60,
    }

    MAX_RERANK_CANDIDATES: int = int(os.getenv("MAX_RERANK_CANDIDATES", "50"))
    RERANK_LATENCY_WARNING_MS: int = int(os.getenv("RERANK_LATENCY_WARNING_MS", "500"))

    HYDE_ENABLED: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
    HYDE_ACTIVATION_THRESHOLD: int = int(os.getenv("HYDE_ACTIVATION_THRESHOLD", "1"))

    # ==========================================================================
    # SLANG INTERCEPTOR SETTINGS
    # ==========================================================================
    SLANG_INTERCEPTOR_TIMEOUT: int = int(os.getenv("SLANG_INTERCEPTOR_TIMEOUT", "20"))

    # ==========================================================================
    # PDF/DOCUMENT SETTINGS
    # ==========================================================================
    PDF_READ_TIMEOUT: int = int(os.getenv("PDF_READ_TIMEOUT", "15"))

    # ==========================================================================
    # RATE LIMITING
    # ==========================================================================
    API_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "30"))

    def __init__(self):
        if not self.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set — Slang Interceptor will use fallback mode")

        if self.PRODUCTION_MODE:
            self._validate()

    def get_claude_model(self, shortname: str) -> str:
        """Resolve 'haiku' or 'sonnet' to a full model identifier."""
        if shortname == "haiku":
            return self.CLAUDE_MODEL_HAIKU
        if shortname == "sonnet":
            return self.CLAUDE_MODEL_SONNET
        # If already a full model string, return as-is
        return shortname or self.CLAUDE_MODEL_SONNET

    def _validate(self):
        missing = []
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        if not self.ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise ValueError(
                f"PRODUCTION_MODE is enabled but required env vars are missing: {', '.join(missing)}"
            )


settings = Settings()
