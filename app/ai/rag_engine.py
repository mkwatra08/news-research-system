"""
Retrieval-Augmented Generation (RAG) engine for news research.
Combines vector search with AI generation for contextual responses.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from openai import AsyncOpenAI

from app.utils.config import get_openai_config
from app.db.vector_store import get_vector_store, SearchResult, Document
from app.scraper.cleaner import CleanedArticle

logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """
    RAG query configuration.
    """
    question: str
    context_filter: Optional[Dict[str, Any]] = None
    max_context_articles: int = 10
    similarity_threshold: float = 0.7
    include_sources: bool = True
    response_style: str = "informative"  # informative, analytical, conversational


@dataclass
class RAGResponse:
    """
    RAG response with context and metadata.
    """
    answer: str
    context_articles: List[Dict[str, Any]]
    confidence_score: float
    sources: List[str]
    generation_time: float
    tokens_used: int
    model_used: str


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for news research.
    """
    
    def __init__(self):
        """Initialize the RAG engine."""
        self.config = get_openai_config()
        self.client = AsyncOpenAI(api_key=self.config["api_key"])
        self.model = self.config["model"]
        self.max_tokens = self.config["max_tokens"]
        self.vector_store = get_vector_store()
        
        # Response style templates
        self.style_templates = {
            "informative": self._get_informative_template(),
            "analytical": self._get_analytical_template(),
            "conversational": self._get_conversational_template(),
            "brief": self._get_brief_template()
        }
    
    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Process a RAG query and generate a contextual response.
        
        Args:
            rag_query: RAG query configuration
            
        Returns:
            RAG response with generated answer and context
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant context
            context_results = await self._retrieve_context(
                query=rag_query.question,
                max_results=rag_query.max_context_articles,
                filter_dict=rag_query.context_filter,
                similarity_threshold=rag_query.similarity_threshold
            )
            
            if not context_results:
                return self._create_no_context_response(rag_query.question, start_time)
            
            # Step 2: Prepare context for generation
            context_text, context_articles = self._prepare_context(context_results)
            
            # Step 3: Generate response
            answer, tokens_used = await self._generate_response(
                question=rag_query.question,
                context=context_text,
                style=rag_query.response_style
            )
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence(
                question=rag_query.question,
                context_results=context_results,
                answer=answer
            )
            
            # Step 5: Collect sources
            sources = []
            if rag_query.include_sources:
                sources = [result.metadata.get("url", "") for result in context_results if result.metadata.get("url")]
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                answer=answer,
                context_articles=context_articles,
                confidence_score=confidence_score,
                sources=sources,
                generation_time=generation_time,
                tokens_used=tokens_used,
                model_used=self.model
            )
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise
    
    async def _retrieve_context(
        self,
        query: str,
        max_results: int,
        filter_dict: Optional[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[SearchResult]:
        """
        Retrieve relevant context from vector store.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            filter_dict: Metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            # Search vector store
            search_results = await self.vector_store.search(
                query=query,
                k=max_results * 2,  # Get more results for filtering
                filter_dict=filter_dict
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result.score >= similarity_threshold
            ]
            
            # Limit to max_results
            return filtered_results[:max_results]
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def _prepare_context(self, search_results: List[SearchResult]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Prepare context text and metadata from search results.
        
        Args:
            search_results: Vector search results
            
        Returns:
            Tuple of (context_text, context_articles_metadata)
        """
        context_parts = []
        context_articles = []
        
        for i, result in enumerate(search_results, 1):
            # Extract metadata
            metadata = result.metadata
            title = metadata.get("title", "Unknown Title")
            source = metadata.get("source", "Unknown Source")
            content = result.content or metadata.get("content", "")
            
            # Truncate content if too long
            if len(content.split()) > 200:
                content = ' '.join(content.split()[:200]) + "..."
            
            # Format context
            context_part = f"""
            [Article {i}]
            Title: {title}
            Source: {source}
            Content: {content}
            Relevance Score: {result.score:.3f}
            """
            
            context_parts.append(context_part.strip())
            
            # Store metadata for response
            context_articles.append({
                "title": title,
                "source": source,
                "url": metadata.get("url", ""),
                "relevance_score": result.score,
                "published_at": metadata.get("published_at"),
                "author": metadata.get("author")
            })
        
        context_text = "\n\n".join(context_parts)
        return context_text, context_articles
    
    async def _generate_response(
        self,
        question: str,
        context: str,
        style: str
    ) -> Tuple[str, int]:
        """
        Generate AI response using the retrieved context.
        
        Args:
            question: User question
            context: Retrieved context
            style: Response style
            
        Returns:
            Tuple of (generated_answer, tokens_used)
        """
        try:
            # Get the appropriate template
            template = self.style_templates.get(style, self.style_templates["informative"])
            
            # Create the prompt
            prompt = template.format(
                question=question,
                context=context
            )
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert news analyst who provides accurate, well-researched answers based on the provided context. Always cite your sources and acknowledge limitations in your knowledge."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            return answer, tokens_used
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise
    
    def _calculate_confidence(
        self,
        question: str,
        context_results: List[SearchResult],
        answer: str
    ) -> float:
        """
        Calculate confidence score for the generated response.
        
        Args:
            question: Original question
            context_results: Retrieved context
            answer: Generated answer
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            score = 0.0
            
            # Context relevance factor (0-0.4)
            if context_results:
                avg_relevance = sum(result.score for result in context_results) / len(context_results)
                score += avg_relevance * 0.4
            
            # Number of sources factor (0-0.2)
            source_count = len(context_results)
            source_score = min(source_count / 5, 1.0) * 0.2
            score += source_score
            
            # Answer completeness factor (0-0.2)
            answer_length = len(answer.split())
            if 50 <= answer_length <= 500:
                completeness_score = 0.2
            else:
                completeness_score = max(0, 0.2 - abs(answer_length - 200) / 1000)
            score += completeness_score
            
            # Question-answer alignment (0-0.2)
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            alignment = len(question_words & answer_words) / len(question_words) if question_words else 0
            score += alignment * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug(f"Failed to calculate confidence score: {e}")
            return 0.5
    
    def _create_no_context_response(self, question: str, start_time: datetime) -> RAGResponse:
        """
        Create a response when no relevant context is found.
        
        Args:
            question: Original question
            start_time: Query start time
            
        Returns:
            RAG response indicating no context found
        """
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            answer=f"I don't have enough relevant information in my database to answer your question about '{question}'. You might want to try a different query or check if there are recent articles on this topic.",
            context_articles=[],
            confidence_score=0.0,
            sources=[],
            generation_time=generation_time,
            tokens_used=0,
            model_used=self.model
        )
    
    async def add_articles_to_context(self, articles: List[CleanedArticle]) -> List[str]:
        """
        Add cleaned articles to the vector store for future retrieval.
        
        Args:
            articles: List of cleaned articles to add
            
        Returns:
            List of document IDs that were added
        """
        try:
            documents = []
            
            for article in articles:
                # Create document for vector store
                document = Document(
                    id=f"article_{hash(article.url)}",
                    content=article.content,
                    metadata={
                        "title": article.title,
                        "source": article.source,
                        "url": article.url,
                        "author": article.author,
                        "published_at": article.published_at.isoformat() if article.published_at else None,
                        "word_count": article.word_count,
                        "quality_score": article.quality_score,
                        "tags": article.tags
                    }
                )
                documents.append(document)
            
            # Add to vector store
            added_ids = await self.vector_store.add_documents(documents)
            
            logger.info(f"Added {len(added_ids)} articles to vector store")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add articles to vector store: {e}")
            return []
    
    async def search_similar_articles(
        self,
        query: str,
        max_results: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar articles without generating a response.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            filter_dict: Metadata filters
            
        Returns:
            List of similar articles with metadata
        """
        try:
            search_results = await self.vector_store.search(
                query=query,
                k=max_results,
                filter_dict=filter_dict
            )
            
            similar_articles = []
            for result in search_results:
                metadata = result.metadata
                similar_articles.append({
                    "title": metadata.get("title", "Unknown Title"),
                    "source": metadata.get("source", "Unknown Source"),
                    "url": metadata.get("url", ""),
                    "author": metadata.get("author"),
                    "published_at": metadata.get("published_at"),
                    "relevance_score": result.score,
                    "summary": result.content[:200] + "..." if result.content and len(result.content) > 200 else result.content
                })
            
            return similar_articles
            
        except Exception as e:
            logger.error(f"Similar articles search failed: {e}")
            return []
    
    def _get_informative_template(self) -> str:
        """Get template for informative responses."""
        return """
        Based on the following news articles, please provide a comprehensive and informative answer to this question: "{question}"

        Context from relevant news articles:
        {context}

        Instructions:
        - Provide a clear, factual answer based on the information in the articles
        - Include specific details and examples when available
        - Mention different perspectives or viewpoints if present
        - If information is limited, acknowledge this clearly
        - Cite sources by mentioning article titles or sources when relevant

        Answer:
        """
    
    def _get_analytical_template(self) -> str:
        """Get template for analytical responses."""
        return """
        Analyze the following news articles to answer this question: "{question}"

        Context from relevant news articles:
        {context}

        Instructions:
        - Provide an analytical response that examines the implications and significance
        - Identify patterns, trends, or connections across the articles
        - Discuss potential causes, effects, and future implications
        - Present a balanced analysis considering multiple perspectives
        - Support your analysis with specific evidence from the articles

        Analysis:
        """
    
    def _get_conversational_template(self) -> str:
        """Get template for conversational responses."""
        return """
        Help me understand this topic by answering: "{question}"

        Here's what I found in recent news articles:
        {context}

        Instructions:
        - Write in a conversational, accessible tone
        - Explain complex topics in simple terms
        - Use analogies or examples to clarify difficult concepts
        - Anticipate follow-up questions the user might have
        - Make the information engaging and easy to understand

        Response:
        """
    
    def _get_brief_template(self) -> str:
        """Get template for brief responses."""
        return """
        Provide a concise answer to: "{question}"

        Context from news articles:
        {context}

        Instructions:
        - Keep the response brief and to the point (2-3 sentences max)
        - Focus on the most important facts
        - Avoid unnecessary details or elaboration
        - Ensure accuracy despite brevity

        Brief Answer:
        """


# Global RAG engine instance
rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """
    Get the global RAG engine instance.
    
    Returns:
        RAG engine instance
    """
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


async def ask_question(
    question: str,
    context_filter: Optional[Dict[str, Any]] = None,
    max_context_articles: int = 10,
    response_style: str = "informative"
) -> RAGResponse:
    """
    Convenience function to ask a question using RAG.
    
    Args:
        question: Question to ask
        context_filter: Metadata filters for context
        max_context_articles: Maximum context articles
        response_style: Response style
        
    Returns:
        RAG response
    """
    engine = get_rag_engine()
    
    query = RAGQuery(
        question=question,
        context_filter=context_filter,
        max_context_articles=max_context_articles,
        response_style=response_style
    )
    
    return await engine.query(query)