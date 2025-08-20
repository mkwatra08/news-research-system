"""
FastAPI main application for the news research system.
Provides REST API endpoints for news research and summarization.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from app.utils.config import settings
from app.db.postgres import init_database, close_database, health_check_db
from app.orchestrator import get_orchestrator, ResearchRequest, ResearchResult
from app.ai.rag_engine import get_rag_engine, RAGQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class ResearchRequestModel(BaseModel):
    """Request model for news research."""
    query: str = Field(..., description="Search query for news research")
    max_articles: int = Field(default=50, ge=1, le=200, description="Maximum articles to collect")
    date_from: Optional[datetime] = Field(default=None, description="Only articles newer than this date")
    date_to: Optional[datetime] = Field(default=None, description="Only articles older than this date")
    sources_filter: Optional[List[str]] = Field(default=None, description="List of sources to include")
    language_filter: str = Field(default="en", description="Language filter")
    summary_type: str = Field(default="general", description="Summary type: general, executive, detailed, bullet_points")
    include_rag_analysis: bool = Field(default=True, description="Whether to include RAG analysis")
    export_format: str = Field(default="json", description="Export format: json, markdown, html, pdf")
    cache_results: bool = Field(default=True, description="Whether to use cached results")


class RAGQueryModel(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="Question to ask")
    max_context_articles: int = Field(default=10, ge=1, le=50, description="Maximum context articles")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    response_style: str = Field(default="informative", description="Response style: informative, analytical, conversational, brief")
    context_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for context")


class ResearchResultModel(BaseModel):
    """Response model for research results."""
    query_id: int
    query: str
    status: str
    articles_count: int
    execution_time: float
    summary: Optional[Dict[str, Any]] = None
    rag_responses: Optional[List[Dict[str, Any]]] = None
    report_url: Optional[str] = None
    error_message: Optional[str] = None


class RAGResponseModel(BaseModel):
    """Response model for RAG queries."""
    answer: str
    confidence_score: float
    context_articles_count: int
    sources: List[str]
    generation_time: float
    tokens_used: int


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    database: bool
    vector_store: bool
    timestamp: datetime


class SummaryRequestModel(BaseModel):
    """Request model for topic summaries."""
    topic: str = Field(..., description="Topic to summarize")
    summary_type: str = Field(default="general", description="Summary type")
    max_length: int = Field(default=500, ge=100, le=2000, description="Maximum summary length in words")
    include_sources: bool = Field(default=True, description="Whether to include source references")


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting news research API...")
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down news research API...")
    try:
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")


# Create FastAPI application
app = FastAPI(
    title="News Research & Summarization API",
    description="AI-powered news research and summarization system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for background tasks (in production, use Redis or similar)
background_tasks_storage: Dict[str, ResearchResult] = {}


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "News Research & Summarization API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connectivity
        db_healthy = await health_check_db()
        
        # Check vector store (simplified check)
        vector_store_healthy = True
        try:
            vector_store = get_rag_engine().vector_store
            await vector_store.get_document_count()
        except Exception as e:
            logger.warning(f"Vector store health check failed: {e}")
            vector_store_healthy = False
        
        overall_status = "healthy" if db_healthy and vector_store_healthy else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            database=db_healthy,
            vector_store=vector_store_healthy,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/research", response_model=ResearchResultModel)
async def research_news(
    request: ResearchRequestModel,
    background_tasks: BackgroundTasks
) -> ResearchResultModel:
    """
    Start a news research task.
    Returns immediately with task ID for async processing.
    """
    try:
        # Validate request
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Create research request
        research_request = ResearchRequest(
            query=request.query.strip(),
            max_articles=request.max_articles,
            date_from=request.date_from,
            date_to=request.date_to,
            sources_filter=request.sources_filter,
            language_filter=request.language_filter,
            summary_type=request.summary_type,
            include_rag_analysis=request.include_rag_analysis,
            export_format=request.export_format,
            cache_results=request.cache_results
        )
        
        # Start background task
        orchestrator = get_orchestrator()
        
        # For demo purposes, we'll execute synchronously
        # In production, use Celery or similar for true background processing
        result = await orchestrator.execute_research(research_request)
        
        # Prepare response
        response = ResearchResultModel(
            query_id=result.query_id,
            query=result.query,
            status=result.status,
            articles_count=len(result.articles),
            execution_time=result.execution_time,
            error_message=result.error_message
        )
        
        # Add summary if available
        if result.summary:
            response.summary = {
                "title": result.summary.title,
                "content": result.summary.content,
                "key_points": result.summary.key_points,
                "word_count": result.summary.word_count,
                "confidence_score": result.summary.confidence_score,
                "sources_count": len(result.summary.sources)
            }
        
        # Add RAG responses if available
        if result.rag_responses:
            response.rag_responses = [
                {
                    "answer": rag.answer,
                    "confidence_score": rag.confidence_score,
                    "sources_count": len(rag.sources),
                    "generation_time": rag.generation_time
                }
                for rag in result.rag_responses
            ]
        
        # Add report URL if available
        if result.report_path:
            response.report_url = f"/reports/{result.query_id}"
        
        return response
        
    except Exception as e:
        logger.error(f"Research request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=RAGResponseModel)
async def ask_question(request: RAGQueryModel) -> RAGResponseModel:
    """
    Ask a question using RAG (Retrieval-Augmented Generation).
    """
    try:
        # Validate request
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Create RAG query
        rag_query = RAGQuery(
            question=request.question.strip(),
            max_context_articles=request.max_context_articles,
            similarity_threshold=request.similarity_threshold,
            response_style=request.response_style,
            context_filter=request.context_filter,
            include_sources=True
        )
        
        # Execute RAG query
        rag_engine = get_rag_engine()
        response = await rag_engine.query(rag_query)
        
        return RAGResponseModel(
            answer=response.answer,
            confidence_score=response.confidence_score,
            context_articles_count=len(response.context_articles),
            sources=response.sources,
            generation_time=response.generation_time,
            tokens_used=response.tokens_used
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", response_model=Dict[str, Any])
async def get_topic_summary(
    topic: str = Query(..., description="Topic to summarize"),
    summary_type: str = Query(default="general", description="Summary type"),
    max_articles: int = Query(default=20, ge=5, le=100, description="Maximum articles to use")
) -> Dict[str, Any]:
    """
    Get a quick summary of a topic using existing data.
    """
    try:
        if not topic.strip():
            raise HTTPException(status_code=400, detail="Topic cannot be empty")
        
        # Search for similar articles in vector store
        rag_engine = get_rag_engine()
        similar_articles = await rag_engine.search_similar_articles(
            query=topic.strip(),
            max_results=max_articles
        )
        
        if not similar_articles:
            raise HTTPException(status_code=404, detail="No relevant articles found for this topic")
        
        # Generate a quick summary using RAG
        rag_query = RAGQuery(
            question=f"Provide a {summary_type} summary of {topic}",
            max_context_articles=min(max_articles, 10),
            response_style="informative",
            include_sources=True
        )
        
        response = await rag_engine.query(rag_query)
        
        return {
            "topic": topic,
            "summary": response.answer,
            "confidence_score": response.confidence_score,
            "articles_used": len(response.context_articles),
            "sources": response.sources[:5],  # Limit sources in response
            "generation_time": response.generation_time,
            "related_articles": similar_articles[:5]  # Top 5 related articles
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Topic summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=Dict[str, Any])
async def search_articles(
    query: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
    similarity_threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
) -> Dict[str, Any]:
    """
    Search for articles in the vector store.
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Search vector store
        rag_engine = get_rag_engine()
        articles = await rag_engine.search_similar_articles(
            query=query.strip(),
            max_results=limit
        )
        
        # Filter by similarity threshold
        filtered_articles = [
            article for article in articles
            if article.get("relevance_score", 0) >= similarity_threshold
        ]
        
        return {
            "query": query,
            "total_results": len(filtered_articles),
            "articles": filtered_articles,
            "search_time": 0.1  # Placeholder
        }
        
    except Exception as e:
        logger.error(f"Article search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{query_id}")
async def download_report(query_id: int):
    """
    Download a generated report.
    """
    try:
        # In a real implementation, you would:
        # 1. Look up the report file path from database
        # 2. Verify user permissions
        # 3. Return the file
        
        # For now, return a placeholder response
        return JSONResponse(
            content={
                "message": f"Report download for query {query_id}",
                "note": "Report download functionality would be implemented here"
            }
        )
        
    except Exception as e:
        logger.error(f"Report download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """
    Get system statistics.
    """
    try:
        # Get vector store stats
        rag_engine = get_rag_engine()
        document_count = await rag_engine.vector_store.get_document_count()
        
        # In a real implementation, you would query the database for more stats
        return {
            "total_articles": document_count,
            "total_queries": 0,  # Would query database
            "total_summaries": 0,  # Would query database
            "system_uptime": "N/A",  # Would calculate actual uptime
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )