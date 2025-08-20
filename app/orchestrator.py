"""
Workflow orchestrator for the news research system.
Manages the complete pipeline: Collect → Clean → Store → Summarize → Respond.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.utils.config import settings
from app.db.postgres import get_db_session, db_manager
from app.db.models import ResearchQuery, Article, QueryArticle, Summary, VectorMetadata
from app.scraper.news_scraper import scrape_news_articles, NewsArticle
from app.scraper.api_connector import fetch_news_from_apis
from app.scraper.cleaner import clean_articles, CleanedArticle
from app.ai.summarizer import summarize_articles, Summary as AISummary
from app.ai.rag_engine import get_rag_engine, RAGQuery, RAGResponse
from app.ai.report_generator import generate_report, ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class ResearchRequest:
    """
    Request configuration for news research.
    """
    query: str
    max_articles: int = 50
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sources_filter: Optional[List[str]] = None
    language_filter: str = "en"
    summary_type: str = "general"
    include_rag_analysis: bool = True
    export_format: str = "json"
    cache_results: bool = True


@dataclass
class ResearchResult:
    """
    Complete research result with all generated content.
    """
    query_id: int
    query: str
    articles: List[CleanedArticle]
    summary: Optional[AISummary] = None
    rag_responses: List[RAGResponse] = None
    report_path: Optional[str] = None
    execution_time: float = 0.0
    status: str = "completed"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.rag_responses is None:
            self.rag_responses = []


class NewsResearchOrchestrator:
    """
    Orchestrates the complete news research workflow.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.rag_engine = get_rag_engine()
        self.report_generator = ReportGenerator()
    
    async def execute_research(self, request: ResearchRequest) -> ResearchResult:
        """
        Execute the complete news research workflow.
        
        Args:
            request: Research request configuration
            
        Returns:
            Complete research result
        """
        start_time = datetime.now()
        query_id = None
        
        try:
            async with db_manager.get_session() as session:
                # Step 1: Check cache and create research query record
                query_id, cached_result = await self._check_cache_and_create_query(
                    session, request
                )
                
                if cached_result and request.cache_results:
                    logger.info(f"Returning cached result for query: {request.query}")
                    return cached_result
                
                # Step 2: Update query status to processing
                await self._update_query_status(session, query_id, "processing")
                
                # Step 3: Collect articles
                logger.info(f"Starting article collection for query: {request.query}")
                raw_articles = await self._collect_articles(request)
                
                if not raw_articles:
                    await self._update_query_status(session, query_id, "failed", "No articles found")
                    return ResearchResult(
                        query_id=query_id,
                        query=request.query,
                        articles=[],
                        status="failed",
                        error_message="No articles found for the given query"
                    )
                
                # Step 4: Clean and validate articles
                logger.info(f"Cleaning {len(raw_articles)} articles")
                cleaned_articles = await self._clean_articles(raw_articles)
                
                if not cleaned_articles:
                    await self._update_query_status(session, query_id, "failed", "No valid articles after cleaning")
                    return ResearchResult(
                        query_id=query_id,
                        query=request.query,
                        articles=[],
                        status="failed",
                        error_message="No valid articles after cleaning and validation"
                    )
                
                # Step 5: Store articles in database
                logger.info(f"Storing {len(cleaned_articles)} articles in database")
                stored_articles = await self._store_articles(session, cleaned_articles, query_id)
                
                # Step 6: Add articles to vector store
                logger.info("Adding articles to vector store")
                await self._add_to_vector_store(session, stored_articles)
                
                # Step 7: Generate summary
                logger.info("Generating summary")
                summary = await self._generate_summary(cleaned_articles, request)
                
                # Step 8: Store summary in database
                if summary:
                    await self._store_summary(session, query_id, summary)
                
                # Step 9: Generate RAG responses (if requested)
                rag_responses = []
                if request.include_rag_analysis:
                    logger.info("Generating RAG analysis")
                    rag_responses = await self._generate_rag_analysis(request.query)
                
                # Step 10: Generate report
                report_path = None
                if request.export_format != "none":
                    logger.info(f"Generating {request.export_format} report")
                    report_path = await self._generate_report(
                        request, cleaned_articles, summary, rag_responses
                    )
                
                # Step 11: Update query status to completed
                execution_time = (datetime.now() - start_time).total_seconds()
                await self._update_query_completion(
                    session, query_id, len(cleaned_articles), execution_time
                )
                
                logger.info(f"Research completed in {execution_time:.2f} seconds")
                
                return ResearchResult(
                    query_id=query_id,
                    query=request.query,
                    articles=cleaned_articles,
                    summary=summary,
                    rag_responses=rag_responses,
                    report_path=report_path,
                    execution_time=execution_time,
                    status="completed"
                )
                
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            
            # Update query status to failed
            if query_id:
                try:
                    async with db_manager.get_session() as session:
                        await self._update_query_status(session, query_id, "failed", str(e))
                except:
                    pass  # Don't fail on status update failure
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ResearchResult(
                query_id=query_id or 0,
                query=request.query,
                articles=[],
                status="failed",
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _check_cache_and_create_query(
        self,
        session: AsyncSession,
        request: ResearchRequest
    ) -> tuple[int, Optional[ResearchResult]]:
        """
        Check for cached results and create new query record.
        
        Args:
            session: Database session
            request: Research request
            
        Returns:
            Tuple of (query_id, cached_result)
        """
        try:
            # Create query hash for caching
            query_hash = self._create_query_hash(request)
            
            # Check for recent cached results (within 1 hour)
            if request.cache_results:
                cache_cutoff = datetime.now() - timedelta(hours=1)
                
                cached_query = await session.execute(
                    select(ResearchQuery).where(
                        and_(
                            ResearchQuery.query_hash == query_hash,
                            ResearchQuery.status == "completed",
                            ResearchQuery.executed_at >= cache_cutoff
                        )
                    )
                )
                cached_query = cached_query.scalar_one_or_none()
                
                if cached_query:
                    # Load cached result
                    cached_result = await self._load_cached_result(session, cached_query)
                    return cached_query.id, cached_result
            
            # Create new query record
            new_query = ResearchQuery(
                query_text=request.query,
                query_hash=query_hash,
                max_articles=request.max_articles,
                date_from=request.date_from,
                date_to=request.date_to,
                sources_filter=request.sources_filter,
                language_filter=request.language_filter,
                status="pending"
            )
            
            session.add(new_query)
            await session.commit()
            await session.refresh(new_query)
            
            return new_query.id, None
            
        except Exception as e:
            logger.error(f"Failed to check cache and create query: {e}")
            raise
    
    def _create_query_hash(self, request: ResearchRequest) -> str:
        """Create a hash for the query request for caching."""
        hash_input = f"{request.query}:{request.max_articles}:{request.date_from}:{request.date_to}:{request.sources_filter}:{request.language_filter}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    async def _load_cached_result(
        self,
        session: AsyncSession,
        cached_query: ResearchQuery
    ) -> ResearchResult:
        """Load a cached research result."""
        try:
            # Load articles
            query_articles = await session.execute(
                select(QueryArticle, Article)
                .join(Article)
                .where(QueryArticle.query_id == cached_query.id)
                .order_by(QueryArticle.rank)
            )
            
            articles = []
            for qa, article in query_articles:
                # Convert database article to CleanedArticle
                cleaned_article = CleanedArticle(
                    title=article.title,
                    content=article.content,
                    url=article.url,
                    source=article.source,
                    author=article.author,
                    published_at=article.published_at,
                    summary=article.summary,
                    word_count=article.word_count or 0,
                    language=article.language or "en",
                    quality_score=article.quality_score or 0.0
                )
                articles.append(cleaned_article)
            
            # Load summary
            summary_record = await session.execute(
                select(Summary).where(Summary.query_id == cached_query.id)
            )
            summary_record = summary_record.scalar_one_or_none()
            
            summary = None
            if summary_record:
                from app.ai.summarizer import Summary as AISummary
                summary = AISummary(
                    title=summary_record.title,
                    content=summary_record.content,
                    key_points=summary_record.key_points or [],
                    word_count=summary_record.word_count,
                    summary_type=summary_record.summary_type,
                    sources=summary_record.primary_sources or [],
                    confidence_score=summary_record.factual_accuracy_score or 0.0,
                    generation_time=summary_record.generation_time,
                    model_used=summary_record.model_used
                )
            
            return ResearchResult(
                query_id=cached_query.id,
                query=cached_query.query_text,
                articles=articles,
                summary=summary,
                execution_time=cached_query.execution_time or 0.0,
                status="completed"
            )
            
        except Exception as e:
            logger.error(f"Failed to load cached result: {e}")
            raise
    
    async def _update_query_status(
        self,
        session: AsyncSession,
        query_id: int,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update query status."""
        try:
            query = await session.get(ResearchQuery, query_id)
            if query:
                query.status = status
                if error_message:
                    query.error_message = error_message
                if status == "processing":
                    query.executed_at = datetime.now()
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update query status: {e}")
    
    async def _collect_articles(self, request: ResearchRequest) -> List[NewsArticle]:
        """Collect articles from various sources."""
        try:
            # Collect from multiple sources in parallel
            tasks = []
            
            # Web scraping
            scraping_task = scrape_news_articles(
                query=request.query,
                max_articles=request.max_articles // 2,
                date_from=request.date_from,
                sources=request.sources_filter
            )
            tasks.append(scraping_task)
            
            # API sources
            api_task = fetch_news_from_apis(
                query=request.query,
                max_articles=request.max_articles // 2,
                from_date=request.date_from,
                to_date=request.date_to
            )
            tasks.append(api_task)
            
            # Execute collection tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Article collection task failed: {result}")
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_articles = []
            for article in all_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    unique_articles.append(article)
            
            # Limit to max_articles
            return unique_articles[:request.max_articles]
            
        except Exception as e:
            logger.error(f"Article collection failed: {e}")
            return []
    
    async def _clean_articles(self, articles: List[NewsArticle]) -> List[CleanedArticle]:
        """Clean and validate articles."""
        try:
            # Run cleaning in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            cleaned_articles = await loop.run_in_executor(
                None, clean_articles, articles
            )
            return cleaned_articles
            
        except Exception as e:
            logger.error(f"Article cleaning failed: {e}")
            return []
    
    async def _store_articles(
        self,
        session: AsyncSession,
        articles: List[CleanedArticle],
        query_id: int
    ) -> List[Article]:
        """Store articles in database."""
        try:
            stored_articles = []
            
            for rank, cleaned_article in enumerate(articles):
                # Check if article already exists
                existing_article = await session.execute(
                    select(Article).where(Article.url == cleaned_article.url)
                )
                existing_article = existing_article.scalar_one_or_none()
                
                if existing_article:
                    article = existing_article
                else:
                    # Create new article
                    article = Article(
                        title=cleaned_article.title,
                        content=cleaned_article.content,
                        url=cleaned_article.url,
                        source=cleaned_article.source,
                        author=cleaned_article.author,
                        published_at=cleaned_article.published_at,
                        summary=cleaned_article.summary,
                        word_count=cleaned_article.word_count,
                        language=cleaned_article.language,
                        quality_score=cleaned_article.quality_score,
                        is_processed=True
                    )
                    session.add(article)
                    await session.flush()
                
                # Create query-article association
                query_article = QueryArticle(
                    query_id=query_id,
                    article_id=article.id,
                    relevance_score=cleaned_article.quality_score,
                    rank=rank,
                    retrieval_method="scraping"
                )
                session.add(query_article)
                
                stored_articles.append(article)
            
            await session.commit()
            return stored_articles
            
        except Exception as e:
            logger.error(f"Failed to store articles: {e}")
            await session.rollback()
            return []
    
    async def _add_to_vector_store(
        self,
        session: AsyncSession,
        articles: List[Article]
    ) -> None:
        """Add articles to vector store."""
        try:
            # Convert to CleanedArticle format for vector store
            cleaned_articles = []
            for article in articles:
                cleaned = CleanedArticle(
                    title=article.title,
                    content=article.content,
                    url=article.url,
                    source=article.source,
                    author=article.author,
                    published_at=article.published_at,
                    summary=article.summary,
                    word_count=article.word_count or 0,
                    language=article.language or "en",
                    quality_score=article.quality_score or 0.0
                )
                cleaned_articles.append(cleaned)
            
            # Add to vector store
            added_ids = await self.rag_engine.add_articles_to_context(cleaned_articles)
            
            # Store vector metadata
            for article, vector_id in zip(articles, added_ids):
                if vector_id:
                    vector_metadata = VectorMetadata(
                        article_id=article.id,
                        vector_id=vector_id,
                        vector_store_type=settings.VECTOR_STORE_TYPE,
                        embedding_model=settings.EMBEDDING_MODEL,
                        embedding_dimension=1536  # OpenAI embedding dimension
                    )
                    session.add(vector_metadata)
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Failed to add articles to vector store: {e}")
    
    async def _generate_summary(
        self,
        articles: List[CleanedArticle],
        request: ResearchRequest
    ) -> Optional[AISummary]:
        """Generate AI summary."""
        try:
            if not articles:
                return None
            
            summary = await summarize_articles(
                articles=articles,
                query=request.query,
                summary_type=request.summary_type,
                max_length=500,
                include_sources=True,
                fact_check=True
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return None
    
    async def _store_summary(
        self,
        session: AsyncSession,
        query_id: int,
        summary: AISummary
    ) -> None:
        """Store summary in database."""
        try:
            db_summary = Summary(
                query_id=query_id,
                title=summary.title,
                content=summary.content,
                key_points=summary.key_points,
                summary_type=summary.summary_type,
                word_count=summary.word_count,
                model_used=summary.model_used or "unknown",
                source_articles_count=len(summary.sources),
                primary_sources=summary.sources,
                generation_time=summary.generation_time,
                factual_accuracy_score=summary.confidence_score
            )
            
            session.add(db_summary)
            await session.commit()
            
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")
    
    async def _generate_rag_analysis(self, query: str) -> List[RAGResponse]:
        """Generate RAG analysis responses."""
        try:
            # Generate multiple analysis questions
            analysis_questions = [
                f"What are the key developments regarding {query}?",
                f"What are the main implications of {query}?",
                f"What do experts say about {query}?"
            ]
            
            rag_responses = []
            for question in analysis_questions:
                try:
                    rag_query = RAGQuery(
                        question=question,
                        max_context_articles=5,
                        response_style="analytical"
                    )
                    
                    response = await self.rag_engine.query(rag_query)
                    if response.confidence_score > 0.3:  # Only include confident responses
                        rag_responses.append(response)
                        
                except Exception as e:
                    logger.warning(f"RAG analysis question failed: {e}")
                    continue
            
            return rag_responses
            
        except Exception as e:
            logger.error(f"RAG analysis generation failed: {e}")
            return []
    
    async def _generate_report(
        self,
        request: ResearchRequest,
        articles: List[CleanedArticle],
        summary: Optional[AISummary],
        rag_responses: List[RAGResponse]
    ) -> Optional[str]:
        """Generate final report."""
        try:
            report_path = await asyncio.get_event_loop().run_in_executor(
                None,
                generate_report,
                request.query,
                articles,
                summary,
                rag_responses,
                "comprehensive",
                request.export_format
            )
            
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    async def _update_query_completion(
        self,
        session: AsyncSession,
        query_id: int,
        articles_processed: int,
        execution_time: float
    ) -> None:
        """Update query completion status."""
        try:
            query = await session.get(ResearchQuery, query_id)
            if query:
                query.status = "completed"
                query.articles_processed = articles_processed
                query.execution_time = execution_time
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update query completion: {e}")


# Global orchestrator instance
orchestrator: Optional[NewsResearchOrchestrator] = None


def get_orchestrator() -> NewsResearchOrchestrator:
    """
    Get the global orchestrator instance.
    
    Returns:
        News research orchestrator
    """
    global orchestrator
    if orchestrator is None:
        orchestrator = NewsResearchOrchestrator()
    return orchestrator


async def research_news(
    query: str,
    max_articles: int = 50,
    date_from: Optional[datetime] = None,
    sources_filter: Optional[List[str]] = None,
    summary_type: str = "general",
    include_rag_analysis: bool = True,
    export_format: str = "json",
    cache_results: bool = True
) -> ResearchResult:
    """
    Convenience function to execute news research.
    
    Args:
        query: Search query
        max_articles: Maximum articles to collect
        date_from: Only articles newer than this date
        sources_filter: List of sources to include
        summary_type: Type of summary to generate
        include_rag_analysis: Whether to include RAG analysis
        export_format: Export format for report
        cache_results: Whether to use cached results
        
    Returns:
        Complete research result
    """
    orchestrator = get_orchestrator()
    
    request = ResearchRequest(
        query=query,
        max_articles=max_articles,
        date_from=date_from,
        sources_filter=sources_filter,
        summary_type=summary_type,
        include_rag_analysis=include_rag_analysis,
        export_format=export_format,
        cache_results=cache_results
    )
    
    return await orchestrator.execute_research(request)