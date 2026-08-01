"""Runtime configuration."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    redis_url: str = "redis://localhost:6379/0"
    object_storage_path: str = "data/raw"
    s3_endpoint_url: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "rag-documents"
    s3_region: str = "us-east-1"
    max_upload_bytes: int = 50 * 1024 * 1024
    requests_per_minute: int = 120
    weaviate_url: str = ""
    weaviate_api_key: str = ""
    weaviate_collection: str = "FilingSection"
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "agentic-rag-secret"
    oidc_issuer: str = ""
    oidc_audience: str = "agentic-rag"
    oidc_jwks_url: str = ""
    dev_auth: bool = True
    max_retrieval_candidates: int = 100
    max_reranked_candidates: int = 20
    max_evidence_per_leaf: int = 8
    max_leaf_retries: int = 2
    max_total_retrieval_calls: int = 30
    max_total_model_calls: int = 30
    direct_confidence_threshold: float = 0.55
    # LiteLLM-prefixed model strings (provider/model) -- swapping the provider
    # (e.g. to "deepseek/deepseek-v4-flash") is a config-only change, no code
    # touched. The provider's own credential env var (MINIMAX_API_KEY,
    # DEEPSEEK_API_KEY, ...) is auto-discovered by LiteLLM; llm_api_key/
    # llm_base_url are optional explicit overrides, e.g. for a custom
    # gateway/proxy.
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_flash_model: str = "minimax/MiniMax-M2.7-highspeed"
    llm_pro_model: str = "minimax/MiniMax-M3"
    # Both models' visible <think>...</think> reasoning shares one token
    # budget with the actual answer -- observed live, a 32-source multihop
    # grounded-claims call produced 8000+ characters of reasoning alone and
    # was cut off mid-sentence before any JSON at a 2000 budget, deterministically
    # failing to parse every time. Calibrated headroom, not a guarantee -- raise
    # further if truncation persists on heavier queries.
    llm_flash_max_tokens: int = 8000
    llm_pro_max_tokens: int = 15000
    # "Highspeed" is a premium low-latency M2.7 variant, not a cheap one -- it
    # costs MORE per token than M3 ($0.60/$2.40 vs $0.30/$1.20). Flash still
    # gets the volume-cost benefit from fewer output tokens on ordinary calls,
    # but per-token this inverts the usual cheap-flash/pricey-pro assumption;
    # rates below are MiniMax's official pay-as-you-go prices, not a guess.
    llm_flash_input_usd_per_million: float = 0.60
    llm_flash_output_usd_per_million: float = 2.40
    llm_pro_input_usd_per_million: float = 0.30
    llm_pro_output_usd_per_million: float = 1.20
    voyage_api_key: str = ""
    voyage_base_url: str = "https://api.voyageai.com/v1"
    voyage_embedding_model: str = "voyage-4-lite"
    voyage_rerank_model: str = "rerank-2.5-lite"
    index_version: str = "v1"
    langsmith_project: str = "agentic-rag"
    allow_sensitive_tracing: bool = False
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "agentic-rag"

    @field_validator("weaviate_url")
    @classmethod
    def normalize_weaviate_url(cls, value: str) -> str:
        return f"https://{value}" if value and "://" not in value else value


def assert_safe_for_environment(settings: Settings) -> None:
    """Refuse to boot with dev-only defaults outside development."""
    if settings.app_env == "development":
        return
    if settings.dev_auth:
        raise RuntimeError("dev_auth must be false outside development")
    if settings.neo4j_password == "agentic-rag-secret":
        raise RuntimeError("neo4j_password must be overridden outside development")


@lru_cache
def get_settings() -> Settings:
    return Settings()
