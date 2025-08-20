"""
Database models for the news research system.
Defines SQLAlchemy models for articles, queries, and summaries.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean,
    ForeignKey, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.sql import func

Base = declarative_base()


class Article(Base):
    """
    Model for storing news articles with metadata and content.
    """
    __tablename__ = "articles"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    title: Mapped[str] = Column(String(500), nullable=False, index=True)
    content: Mapped[str] = Column(Text, nullable=False)
    summary: Mapped[Optional[str]] = Column(Text, nullable=True)
    url: Mapped[str] = Column(String(1000), unique=True, nullable=False, index=True)
    source: Mapped[str] = Column(String(200), nullable=False, index=True)
    author: Mapped[Optional[str]] = Column(String(200), nullable=True)
    published_at: Mapped[Optional[datetime]] = Column(DateTime, nullable=True, index=True)
    scraped_at: Mapped[datetime] = Column(DateTime, default=func.now(), nullable=False)
    
    # Content analysis fields
    word_count: Mapped[Optional[int]] = Column(Integer, nullable=True)
    language: Mapped[Optional[str]] = Column(String(10), nullable=True, default="en")
    
    # Vector embedding metadata
    embedding_id: Mapped[Optional[str]] = Column(String(100), nullable=True, index=True)
    embedding_model: Mapped[Optional[str]] = Column(String(100), nullable=True)
    
    # Quality and relevance scores
    quality_score: Mapped[Optional[float]] = Column(Float, nullable=True)
    relevance_score: Mapped[Optional[float]] = Column(Float, nullable=True)
    
    # Processing status
    is_processed: Mapped[bool] = Column(Boolean, default=False, nullable=False)
    processing_error: Mapped[Optional[str]] = Column(Text, nullable=True)
    
    # Relationships
    query_articles: Mapped[List["QueryArticle"]] = relationship(
        "QueryArticle", back_populates="article"
    )
    
    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:50]}...', source='{self.source}')>"


class ResearchQuery(Base):
    """
    Model for storing user research queries and their metadata.
    """
    __tablename__ = "research_queries"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    query_text: Mapped[str] = Column(String(1000), nullable=False, index=True)
    query_hash: Mapped[str] = Column(String(64), unique=True, nullable=False, index=True)
    
    # Query parameters
    max_articles: Mapped[int] = Column(Integer, default=50, nullable=False)
    date_from: Mapped[Optional[datetime]] = Column(DateTime, nullable=True)
    date_to: Mapped[Optional[datetime]] = Column(DateTime, nullable=True)
    sources_filter: Mapped[Optional[List[str]]] = Column(JSON, nullable=True)
    language_filter: Mapped[str] = Column(String(10), default="en", nullable=False)
    
    # Execution metadata
    created_at: Mapped[datetime] = Column(DateTime, default=func.now(), nullable=False)
    executed_at: Mapped[Optional[datetime]] = Column(DateTime, nullable=True)
    execution_time: Mapped[Optional[float]] = Column(Float, nullable=True)  # in seconds
    
    # Results metadata
    articles_found: Mapped[int] = Column(Integer, default=0, nullable=False)
    articles_processed: Mapped[int] = Column(Integer, default=0, nullable=False)
    
    # Status tracking
    status: Mapped[str] = Column(
        String(20), 
        default="pending", 
        nullable=False,
        index=True
    )  # pending, processing, completed, failed
    error_message: Mapped[Optional[str]] = Column(Text, nullable=True)
    
    # Relationships
    query_articles: Mapped[List["QueryArticle"]] = relationship(
        "QueryArticle", back_populates="query"
    )
    summaries: Mapped[List["Summary"]] = relationship(
        "Summary", back_populates="query"
    )
    
    def __repr__(self) -> str:
        return f"<ResearchQuery(id={self.id}, query='{self.query_text[:50]}...', status='{self.status}')>"


class QueryArticle(Base):
    """
    Association table linking queries to articles with relevance scores.
    """
    __tablename__ = "query_articles"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    query_id: Mapped[int] = Column(Integer, ForeignKey("research_queries.id"), nullable=False)
    article_id: Mapped[int] = Column(Integer, ForeignKey("articles.id"), nullable=False)
    
    # Relevance and ranking
    relevance_score: Mapped[float] = Column(Float, nullable=False, default=0.0)
    rank: Mapped[int] = Column(Integer, nullable=False)
    
    # Retrieval metadata
    retrieved_at: Mapped[datetime] = Column(DateTime, default=func.now(), nullable=False)
    retrieval_method: Mapped[str] = Column(String(50), nullable=False)  # vector, keyword, hybrid
    
    # Relationships
    query: Mapped["ResearchQuery"] = relationship(
        "ResearchQuery", back_populates="query_articles"
    )
    article: Mapped["Article"] = relationship(
        "Article", back_populates="query_articles"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_query_relevance", "query_id", "relevance_score"),
        Index("idx_query_rank", "query_id", "rank"),
    )
    
    def __repr__(self) -> str:
        return f"<QueryArticle(query_id={self.query_id}, article_id={self.article_id}, score={self.relevance_score})>"


class Summary(Base):
    """
    Model for storing generated summaries and reports.
    """
    __tablename__ = "summaries"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    query_id: Mapped[int] = Column(Integer, ForeignKey("research_queries.id"), nullable=False)
    
    # Summary content
    title: Mapped[str] = Column(String(500), nullable=False)
    content: Mapped[str] = Column(Text, nullable=False)
    key_points: Mapped[List[str]] = Column(JSON, nullable=True)
    
    # Summary metadata
    summary_type: Mapped[str] = Column(String(50), default="general", nullable=False)  # general, executive, detailed
    format: Mapped[str] = Column(String(20), default="text", nullable=False)  # text, markdown, html, json
    word_count: Mapped[int] = Column(Integer, nullable=False)
    
    # AI model information
    model_used: Mapped[str] = Column(String(100), nullable=False)
    model_version: Mapped[Optional[str]] = Column(String(50), nullable=True)
    temperature: Mapped[Optional[float]] = Column(Float, nullable=True)
    max_tokens: Mapped[Optional[int]] = Column(Integer, nullable=True)
    
    # Quality metrics
    coherence_score: Mapped[Optional[float]] = Column(Float, nullable=True)
    factual_accuracy_score: Mapped[Optional[float]] = Column(Float, nullable=True)
    
    # References and sources
    source_articles_count: Mapped[int] = Column(Integer, nullable=False)
    primary_sources: Mapped[List[str]] = Column(JSON, nullable=True)  # List of source URLs
    
    # Generation metadata
    created_at: Mapped[datetime] = Column(DateTime, default=func.now(), nullable=False)
    generation_time: Mapped[Optional[float]] = Column(Float, nullable=True)  # in seconds
    
    # Relationships
    query: Mapped["ResearchQuery"] = relationship(
        "ResearchQuery", back_populates="summaries"
    )
    
    def __repr__(self) -> str:
        return f"<Summary(id={self.id}, query_id={self.query_id}, type='{self.summary_type}')>"


class VectorMetadata(Base):
    """
    Model for tracking vector embeddings metadata.
    """
    __tablename__ = "vector_metadata"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    article_id: Mapped[int] = Column(Integer, ForeignKey("articles.id"), nullable=False)
    
    # Vector store information
    vector_id: Mapped[str] = Column(String(100), unique=True, nullable=False, index=True)
    vector_store_type: Mapped[str] = Column(String(20), nullable=False)  # faiss, pinecone
    index_name: Mapped[Optional[str]] = Column(String(100), nullable=True)
    
    # Embedding information
    embedding_model: Mapped[str] = Column(String(100), nullable=False)
    embedding_dimension: Mapped[int] = Column(Integer, nullable=False)
    chunk_index: Mapped[int] = Column(Integer, default=0, nullable=False)  # For chunked articles
    
    # Processing metadata
    created_at: Mapped[datetime] = Column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    def __repr__(self) -> str:
        return f"<VectorMetadata(id={self.id}, article_id={self.article_id}, vector_id='{self.vector_id}')>"