from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    OPENAI_API_KEY: SecretStr | None = Field(None, description="OpenAI API Key")
    OPENAI_MODEL_AGENT: str = Field("gpt-5", description="Model for agentic reasoning")
    OPENAI_MODEL_VISION: str = Field(
        "gpt-5-mini", 
        description="Cost-effective vision capable model (e.g. gpt-4o-mini or gpt-5-mini)"
    )
    OPENAI_MODEL_STRUCTURED: str = Field(
        "gpt-4o-2024-08-06", 
        description="Model optimized for structured JSON output"
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        "text-embedding-3-large", 
        description="Model for embeddings"
    )
    PAYROLL_MISMATCH_THRESHOLD: float = Field(
        0.05, 
        description="Threshold for payroll mismatch warnings"
    )

# Singleton instance
settings = Settings()
