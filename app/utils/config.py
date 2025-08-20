"""
Configuration management for the news research system.
Handles environment variables, database settings, and AI model configurations.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:password@postgres:5432/news_research",
        description="PostgreSQL connection URL"
    )
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = Field(default="faiss", description="Vector store type (faiss/pinecone)")
    FAISS_INDEX_PATH: str = Field(default="./data/faiss_index", description="FAISS index storage path")
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="Pinecone API key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Pinecone environment")
    PINECONE_INDEX_NAME: Optional[str] = Field(default="news-research", description="Pinecone index name")
    
    # AI Model Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4", description="OpenAI model to use")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Embedding model")
    MAX_TOKENS: int = Field(default=2000, description="Maximum tokens for AI responses")
    
    # News API Configuration
    NEWS_API_KEY: Optional[str] = Field(default=None, description="News API key")
    GOOGLE_NEWS_RSS_URL: str = Field(
        default="https://news.google.com/rss",
        description="Google News RSS URL"
    )
    
    # Scraping Configuration
    MAX_ARTICLES_PER_QUERY: int = Field(default=50, description="Maximum articles to fetch per query")
    SCRAPING_TIMEOUT: int = Field(default=30, description="Timeout for web scraping requests")
    USER_AGENT: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        description="User agent for web scraping"
    )
    
    # Cache Configuration
    REDIS_URL: Optional[str] = Field(default=None, description="Redis URL for caching")
    CACHE_TTL: int = Field(default=3600, description="Cache time-to-live in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """Get the database URL for SQLAlchemy."""
    return settings.DATABASE_URL


def get_openai_config() -> dict:
    """Get OpenAI configuration."""
    return {
        "api_key": settings.OPENAI_API_KEY,
        "model": settings.OPENAI_MODEL,
        "max_tokens": settings.MAX_TOKENS,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


def get_vector_store_config() -> dict:
    """Get vector store configuration."""
    if settings.VECTOR_STORE_TYPE == "pinecone":
        return {
            "type": "pinecone",
            "api_key": settings.PINECONE_API_KEY,
            "environment": settings.PINECONE_ENVIRONMENT,
            "index_name": settings.PINECONE_INDEX_NAME,
        }
    else:
        return {
            "type": "faiss",
            "index_path": settings.FAISS_INDEX_PATH,
        }


def get_scraping_config() -> dict:
    """Get web scraping configuration."""
    return {
        "max_articles": settings.MAX_ARTICLES_PER_QUERY,
        "timeout": settings.SCRAPING_TIMEOUT,
        "user_agent": settings.USER_AGENT,
        "news_api_key": settings.NEWS_API_KEY,
        "google_news_rss": settings.GOOGLE_NEWS_RSS_URL,
    }