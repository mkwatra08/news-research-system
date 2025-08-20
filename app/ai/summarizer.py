"""
AI-powered summarization module using OpenAI GPT models.
Generates concise, fact-checked summaries from news articles.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from openai import AsyncOpenAI

from app.utils.config import get_openai_config
from app.scraper.cleaner import CleanedArticle

logger = logging.getLogger(__name__)


@dataclass
class SummaryRequest:
    """
    Request configuration for summary generation.
    """
    articles: List[CleanedArticle]
    query: str
    summary_type: str = "general"  # general, executive, detailed, bullet_points
    max_length: int = 500
    focus_areas: Optional[List[str]] = None
    include_sources: bool = True
    fact_check: bool = True


@dataclass
class Summary:
    """
    Generated summary with metadata.
    """
    title: str
    content: str
    key_points: List[str]
    word_count: int
    summary_type: str
    sources: List[str]
    confidence_score: float
    fact_check_notes: Optional[str] = None
    generation_time: Optional[float] = None
    model_used: Optional[str] = None


class NewsSummarizer:
    """
    AI-powered news summarizer using OpenAI GPT models.
    """
    
    def __init__(self):
        """Initialize the summarizer with OpenAI configuration."""
        self.config = get_openai_config()
        self.client = AsyncOpenAI(api_key=self.config["api_key"])
        self.model = self.config["model"]
        self.max_tokens = self.config["max_tokens"]
        
        # Summary templates for different types
        self.templates = {
            "general": self._get_general_template(),
            "executive": self._get_executive_template(),
            "detailed": self._get_detailed_template(),
            "bullet_points": self._get_bullet_points_template()
        }
    
    async def generate_summary(self, request: SummaryRequest) -> Summary:
        """
        Generate a summary from news articles.
        
        Args:
            request: Summary generation request
            
        Returns:
            Generated summary with metadata
        """
        start_time = datetime.now()
        
        try:
            # Prepare articles for processing
            processed_articles = self._prepare_articles(request.articles)
            
            if not processed_articles:
                raise ValueError("No valid articles provided for summarization")
            
            # Generate summary using OpenAI
            summary_content = await self._generate_summary_content(
                articles=processed_articles,
                query=request.query,
                summary_type=request.summary_type,
                max_length=request.max_length,
                focus_areas=request.focus_areas
            )
            
            # Extract key points
            key_points = await self._extract_key_points(summary_content, request.query)
            
            # Generate title
            title = await self._generate_title(summary_content, request.query)
            
            # Fact-check if requested
            fact_check_notes = None
            if request.fact_check:
                fact_check_notes = await self._fact_check_summary(
                    summary_content, processed_articles
                )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                summary_content, processed_articles, request.query
            )
            
            # Collect sources
            sources = []
            if request.include_sources:
                sources = [article.url for article in request.articles]
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return Summary(
                title=title,
                content=summary_content,
                key_points=key_points,
                word_count=len(summary_content.split()),
                summary_type=request.summary_type,
                sources=sources,
                confidence_score=confidence_score,
                fact_check_notes=fact_check_notes,
                generation_time=generation_time,
                model_used=self.model
            )
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise
    
    def _prepare_articles(self, articles: List[CleanedArticle]) -> List[Dict[str, Any]]:
        """
        Prepare articles for summarization by extracting relevant information.
        
        Args:
            articles: List of cleaned articles
            
        Returns:
            List of processed article data
        """
        processed = []
        
        for article in articles:
            # Truncate very long articles to stay within token limits
            content = article.content
            if len(content.split()) > 800:  # Roughly 800 words
                content = ' '.join(content.split()[:800]) + "..."
            
            processed.append({
                "title": article.title,
                "content": content,
                "source": article.source,
                "url": article.url,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "author": article.author,
                "quality_score": article.quality_score
            })
        
        # Sort by quality score and limit to top articles
        processed.sort(key=lambda x: x["quality_score"], reverse=True)
        return processed[:20]  # Limit to top 20 articles
    
    async def _generate_summary_content(
        self,
        articles: List[Dict[str, Any]],
        query: str,
        summary_type: str,
        max_length: int,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Generate summary content using OpenAI.
        
        Args:
            articles: Processed articles
            query: Original search query
            summary_type: Type of summary to generate
            max_length: Maximum length in words
            focus_areas: Specific areas to focus on
            
        Returns:
            Generated summary content
        """
        try:
            # Get the appropriate template
            template = self.templates.get(summary_type, self.templates["general"])
            
            # Prepare context
            articles_text = self._format_articles_for_prompt(articles)
            
            # Build focus areas text
            focus_text = ""
            if focus_areas:
                focus_text = f"\nPay special attention to: {', '.join(focus_areas)}"
            
            # Create the prompt
            prompt = template.format(
                query=query,
                articles=articles_text,
                max_length=max_length,
                focus_areas=focus_text
            )
            
            # Generate summary
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert news analyst and writer. Create accurate, well-structured summaries that capture the most important information while maintaining journalistic integrity."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=min(self.max_tokens, max_length * 2),  # Allow some buffer
                temperature=0.3,  # Lower temperature for more factual content
                top_p=0.9
            )
            
            summary_content = response.choices[0].message.content.strip()
            
            # Ensure summary doesn't exceed max length
            words = summary_content.split()
            if len(words) > max_length:
                summary_content = ' '.join(words[:max_length]) + "..."
            
            return summary_content
            
        except Exception as e:
            logger.error(f"Failed to generate summary content: {e}")
            raise
    
    async def _extract_key_points(self, summary: str, query: str) -> List[str]:
        """
        Extract key points from the generated summary.
        
        Args:
            summary: Generated summary text
            query: Original search query
            
        Returns:
            List of key points
        """
        try:
            prompt = f"""
            Extract 3-5 key points from the following news summary about "{query}".
            Return only the key points as a simple list, one point per line, without numbers or bullets.
            
            Summary:
            {summary}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled editor who extracts the most important points from news content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            key_points_text = response.choices[0].message.content.strip()
            key_points = [point.strip() for point in key_points_text.split('\n') if point.strip()]
            
            return key_points[:5]  # Limit to 5 key points
            
        except Exception as e:
            logger.error(f"Failed to extract key points: {e}")
            return []
    
    async def _generate_title(self, summary: str, query: str) -> str:
        """
        Generate a compelling title for the summary.
        
        Args:
            summary: Generated summary text
            query: Original search query
            
        Returns:
            Generated title
        """
        try:
            prompt = f"""
            Create a compelling, informative title for a news summary about "{query}".
            The title should be 6-12 words long and capture the main theme of the summary.
            
            Summary:
            {summary[:500]}...
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled headline writer who creates engaging, accurate news titles."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=50,
                temperature=0.4
            )
            
            title = response.choices[0].message.content.strip()
            
            # Clean up the title
            title = title.strip('"\'')
            if len(title.split()) > 15:
                title = ' '.join(title.split()[:15])
            
            return title or f"News Summary: {query}"
            
        except Exception as e:
            logger.error(f"Failed to generate title: {e}")
            return f"News Summary: {query}"
    
    async def _fact_check_summary(
        self,
        summary: str,
        articles: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Perform basic fact-checking on the generated summary.
        
        Args:
            summary: Generated summary
            articles: Source articles
            
        Returns:
            Fact-check notes or None
        """
        try:
            # Create a condensed version of source articles for fact-checking
            sources_text = ""
            for article in articles[:5]:  # Use top 5 articles
                sources_text += f"Source: {article['source']}\n{article['content'][:300]}...\n\n"
            
            prompt = f"""
            Review the following summary against the provided source articles and identify any potential factual inconsistencies or unsupported claims.
            
            Summary to check:
            {summary}
            
            Source articles:
            {sources_text}
            
            If you find any issues, list them briefly. If the summary appears factually consistent, respond with "No significant issues found."
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fact-checker who verifies news content against source materials."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            fact_check_result = response.choices[0].message.content.strip()
            
            return fact_check_result if "No significant issues found" not in fact_check_result else None
            
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            return None
    
    def _calculate_confidence_score(
        self,
        summary: str,
        articles: List[Dict[str, Any]],
        query: str
    ) -> float:
        """
        Calculate confidence score for the generated summary.
        
        Args:
            summary: Generated summary
            articles: Source articles
            query: Original query
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            score = 0.0
            
            # Source quality factor (0-0.4)
            if articles:
                avg_quality = sum(article["quality_score"] for article in articles) / len(articles)
                score += avg_quality * 0.4
            
            # Number of sources factor (0-0.2)
            source_count = len(articles)
            source_score = min(source_count / 10, 1.0) * 0.2
            score += source_score
            
            # Summary length appropriateness (0-0.2)
            word_count = len(summary.split())
            if 100 <= word_count <= 800:
                length_score = 0.2
            else:
                length_score = max(0, 0.2 - abs(word_count - 400) / 2000)
            score += length_score
            
            # Query relevance (0-0.2)
            query_words = set(query.lower().split())
            summary_words = set(summary.lower().split())
            relevance = len(query_words & summary_words) / len(query_words) if query_words else 0
            score += relevance * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug(f"Failed to calculate confidence score: {e}")
            return 0.5  # Default moderate confidence
    
    def _format_articles_for_prompt(self, articles: List[Dict[str, Any]]) -> str:
        """
        Format articles for inclusion in the prompt.
        
        Args:
            articles: List of article data
            
        Returns:
            Formatted articles text
        """
        formatted = []
        
        for i, article in enumerate(articles, 1):
            article_text = f"""
            Article {i}:
            Title: {article['title']}
            Source: {article['source']}
            Content: {article['content'][:500]}...
            """
            formatted.append(article_text.strip())
        
        return "\n\n".join(formatted)
    
    def _get_general_template(self) -> str:
        """Get template for general summaries."""
        return """
        Create a comprehensive summary of the news articles about "{query}" below.
        
        Requirements:
        - Write in a clear, objective journalistic style
        - Include the most important facts and developments
        - Maintain chronological flow where relevant
        - Keep the summary to approximately {max_length} words
        - Include different perspectives when available{focus_areas}
        
        Articles:
        {articles}
        
        Summary:
        """
    
    def _get_executive_template(self) -> str:
        """Get template for executive summaries."""
        return """
        Create an executive summary of the news articles about "{query}" below.
        
        Requirements:
        - Focus on key decisions, impacts, and implications
        - Use a professional, concise tone
        - Highlight actionable insights and trends
        - Structure with clear sections if appropriate
        - Keep to approximately {max_length} words{focus_areas}
        
        Articles:
        {articles}
        
        Executive Summary:
        """
    
    def _get_detailed_template(self) -> str:
        """Get template for detailed summaries."""
        return """
        Create a detailed analytical summary of the news articles about "{query}" below.
        
        Requirements:
        - Provide comprehensive coverage of all major points
        - Include background context and analysis
        - Discuss implications and potential outcomes
        - Cite specific examples and data when available
        - Aim for approximately {max_length} words{focus_areas}
        
        Articles:
        {articles}
        
        Detailed Summary:
        """
    
    def _get_bullet_points_template(self) -> str:
        """Get template for bullet point summaries."""
        return """
        Create a bullet-point summary of the news articles about "{query}" below.
        
        Requirements:
        - Use clear, concise bullet points
        - Organize by theme or chronology
        - Each point should be one complete sentence
        - Include 5-10 main points
        - Focus on the most newsworthy information{focus_areas}
        
        Articles:
        {articles}
        
        Summary:
        """


async def summarize_articles(
    articles: List[CleanedArticle],
    query: str,
    summary_type: str = "general",
    max_length: int = 500,
    focus_areas: Optional[List[str]] = None,
    include_sources: bool = True,
    fact_check: bool = True
) -> Summary:
    """
    Convenience function to summarize news articles.
    
    Args:
        articles: List of cleaned articles
        query: Original search query
        summary_type: Type of summary (general, executive, detailed, bullet_points)
        max_length: Maximum length in words
        focus_areas: Specific areas to focus on
        include_sources: Whether to include source URLs
        fact_check: Whether to perform fact-checking
        
    Returns:
        Generated summary
    """
    summarizer = NewsSummarizer()
    
    request = SummaryRequest(
        articles=articles,
        query=query,
        summary_type=summary_type,
        max_length=max_length,
        focus_areas=focus_areas,
        include_sources=include_sources,
        fact_check=fact_check
    )
    
    return await summarizer.generate_summary(request)