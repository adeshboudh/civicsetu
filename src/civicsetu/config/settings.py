from typing import Annotated
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Providers
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    # LLM Routing
    primary_model: str = Field(
        default="gemini/gemini-2.5-flash-lite", alias="PRIMARY_MODEL"
    )
    fallback_model_1: str = Field(
        default="groq/llama-3.3-70b-versatile", alias="FALLBACK_MODEL_1"
    )
    fallback_model_2: str = Field(
        default="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        alias="FALLBACK_MODEL_2",
    )
    fallback_model_3: str = Field(
        default="openrouter/qwen/qwen3.6-plus:free",
        alias="FALLBACK_MODEL_3",
    )
    local_model: str = Field(default="ollama/mistral", alias="LOCAL_MODEL")

    # Embeddings
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # PostgreSQL
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="civicsetu", alias="POSTGRES_DB")
    postgres_user: str = Field(default="civicsetu", alias="POSTGRES_USER")
    postgres_password: str = Field(default="civicsetu_dev", alias="POSTGRES_PASSWORD")

    # Neo4j (Phase 1)
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")

    # FastAPI
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_env: str = Field(default="development", alias="API_ENV")
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=["http://localhost:3000"],
        alias="ALLOWED_ORIGINS",
    )

    # Observability
    phoenix_host: str = Field(default="localhost", alias="PHOENIX_HOST")
    phoenix_port: int = Field(default=6006, alias="PHOENIX_PORT")

    # Ingestion
    data_raw_dir: str = Field(default="data/raw", alias="DATA_RAW_DIR")
    data_processed_dir: str = Field(
        default="data/processed", alias="DATA_PROCESSED_DIR"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_conninfo(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        return self.api_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
