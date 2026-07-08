"""Central configuration, loaded once from environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embed_model: str = "mxbai-embed-large"

    qdrant_path: str = "./qdrant_db"
    collection_name: str = "kb_docs"

    top_k: int = 5


settings = Settings()
