"""
API connector for external news APIs.
Supports NewsAPI, Guardian API, and other news service providers.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import aiohttp

from app.utils.config import get_scraping_config
from app.scraper.news_scraper import NewsArticle

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """
    Represents a response from a news API.
    """
    articles: List[NewsArticle]
    total_results: int
    status: str
    error_message: Optional[str] = None


class NewsAPIConnector:
    """
    Connector for NewsAPI.org service.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize NewsAPI connector.
        
        Args:
            api_key: NewsAPI.org API key
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={"X-API-Key": self.api_key}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def search_everything(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int = 20,
        page: int = 1
    ) -> APIResponse:
        """
        Search articles using NewsAPI everything endpoint.
        
        Args:
            query: Search query
            sources: List of news sources/blogs
            domains: List of domains to search
            from_date: Oldest article date
            to_date: Newest article date
            language: Article language
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of articles per page (max 100)
            page: Page number
            
        Returns:
            APIResponse with articles
        """
        try:
            params = {
                "q": query,
                "language": language,
                "sortBy": sort_by,
                "pageSize": min(page_size, 100),
                "page": page
            }
            
            if sources:
                params["sources"] = ",".join(sources)
            
            if domains:
                params["domains"] = ",".join(domains)
            
            if from_date:
                params["from"] = from_date.isoformat()
            
            if to_date:
                params["to"] = to_date.isoformat()
            
            url = f"{self.base_url}/everything"
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if response.status != 200:
                    return APIResponse(
                        articles=[],
                        total_results=0,
                        status="error",
                        error_message=data.get("message", f"HTTP {response.status}")
                    )
                
                articles = []
                for article_data in data.get("articles", []):
                    article = self._parse_newsapi_article(article_data)
                    if article:
                        articles.append(article)
                
                return APIResponse(
                    articles=articles,
                    total_results=data.get("totalResults", 0),
                    status=data.get("status", "ok")
                )
                
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            return APIResponse(
                articles=[],
                total_results=0,
                status="error",
                error_message=str(e)
            )
    
    async def get_top_headlines(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        sources: Optional[List[str]] = None,
        query: Optional[str] = None,
        page_size: int = 20,
        page: int = 1
    ) -> APIResponse:
        """
        Get top headlines using NewsAPI headlines endpoint.
        
        Args:
            country: Country code (e.g., 'us', 'gb')
            category: Category (business, entertainment, general, health, science, sports, technology)
            sources: List of news sources
            query: Search query
            page_size: Number of articles per page (max 100)
            page: Page number
            
        Returns:
            APIResponse with articles
        """
        try:
            params = {
                "pageSize": min(page_size, 100),
                "page": page
            }
            
            if country:
                params["country"] = country
            
            if category:
                params["category"] = category
            
            if sources:
                params["sources"] = ",".join(sources)
            
            if query:
                params["q"] = query
            
            url = f"{self.base_url}/top-headlines"
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if response.status != 200:
                    return APIResponse(
                        articles=[],
                        total_results=0,
                        status="error",
                        error_message=data.get("message", f"HTTP {response.status}")
                    )
                
                articles = []
                for article_data in data.get("articles", []):
                    article = self._parse_newsapi_article(article_data)
                    if article:
                        articles.append(article)
                
                return APIResponse(
                    articles=articles,
                    total_results=len(articles),
                    status=data.get("status", "ok")
                )
                
        except Exception as e:
            logger.error(f"NewsAPI top headlines failed: {e}")
            return APIResponse(
                articles=[],
                total_results=0,
                status="error",
                error_message=str(e)
            )
    
    def _parse_newsapi_article(self, data: Dict[str, Any]) -> Optional[NewsArticle]:
        """
        Parse NewsAPI article data into NewsArticle.
        
        Args:
            data: Article data from NewsAPI
            
        Returns:
            NewsArticle or None if parsing fails
        """
        try:
            title = data.get("title", "").strip()
            url = data.get("url", "").strip()
            content = data.get("content", "").strip()
            description = data.get("description", "").strip()
            
            if not title or not url:
                return None
            
            # Use description as content if content is not available or truncated
            if not content or content.endswith("..."):
                content = description
            
            # Parse published date
            published_at = None
            if data.get("publishedAt"):
                try:
                    published_at = datetime.fromisoformat(
                        data["publishedAt"].replace("Z", "+00:00")
                    )
                except:
                    pass
            
            # Extract source name
            source_data = data.get("source", {})
            source_name = source_data.get("name", "Unknown")
            
            return NewsArticle(
                title=title,
                content=content,
                url=url,
                source=source_name,
                author=data.get("author"),
                published_at=published_at,
                summary=description,
                image_url=data.get("urlToImage")
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse NewsAPI article: {e}")
            return None


class GuardianAPIConnector:
    """
    Connector for The Guardian API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Guardian API connector.
        
        Args:
            api_key: Guardian API key
        """
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def search_content(
        self,
        query: str,
        section: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        order_by: str = "relevance",
        page_size: int = 20,
        page: int = 1,
        show_fields: str = "headline,byline,bodyText,thumbnail,publication"
    ) -> APIResponse:
        """
        Search Guardian content.
        
        Args:
            query: Search query
            section: Guardian section
            from_date: From date
            to_date: To date
            order_by: Order by (newest, oldest, relevance)
            page_size: Results per page
            page: Page number
            show_fields: Fields to include
            
        Returns:
            APIResponse with articles
        """
        try:
            params = {
                "q": query,
                "api-key": self.api_key,
                "order-by": order_by,
                "page-size": min(page_size, 50),
                "page": page,
                "show-fields": show_fields
            }
            
            if section:
                params["section"] = section
            
            if from_date:
                params["from-date"] = from_date.strftime("%Y-%m-%d")
            
            if to_date:
                params["to-date"] = to_date.strftime("%Y-%m-%d")
            
            url = f"{self.base_url}/search"
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if response.status != 200:
                    return APIResponse(
                        articles=[],
                        total_results=0,
                        status="error",
                        error_message=data.get("message", f"HTTP {response.status}")
                    )
                
                response_data = data.get("response", {})
                articles = []
                
                for article_data in response_data.get("results", []):
                    article = self._parse_guardian_article(article_data)
                    if article:
                        articles.append(article)
                
                return APIResponse(
                    articles=articles,
                    total_results=response_data.get("total", 0),
                    status=response_data.get("status", "ok")
                )
                
        except Exception as e:
            logger.error(f"Guardian API search failed: {e}")
            return APIResponse(
                articles=[],
                total_results=0,
                status="error",
                error_message=str(e)
            )
    
    def _parse_guardian_article(self, data: Dict[str, Any]) -> Optional[NewsArticle]:
        """
        Parse Guardian API article data.
        
        Args:
            data: Article data from Guardian API
            
        Returns:
            NewsArticle or None if parsing fails
        """
        try:
            title = data.get("webTitle", "").strip()
            url = data.get("webUrl", "").strip()
            
            if not title or not url:
                return None
            
            fields = data.get("fields", {})
            content = fields.get("bodyText", "").strip()
            
            # Parse published date
            published_at = None
            if data.get("webPublicationDate"):
                try:
                    published_at = datetime.fromisoformat(
                        data["webPublicationDate"].replace("Z", "+00:00")
                    )
                except:
                    pass
            
            return NewsArticle(
                title=title,
                content=content,
                url=url,
                source="The Guardian",
                author=fields.get("byline"),
                published_at=published_at,
                summary=fields.get("headline", title),
                image_url=fields.get("thumbnail")
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Guardian article: {e}")
            return None


class NewsAPIAggregator:
    """
    Aggregates results from multiple news APIs.
    """
    
    def __init__(self):
        """Initialize the API aggregator."""
        self.config = get_scraping_config()
        self.connectors = []
        
        # Initialize available connectors
        if self.config.get("news_api_key"):
            self.newsapi_connector = NewsAPIConnector(self.config["news_api_key"])
        else:
            self.newsapi_connector = None
        
        # Add more API connectors as needed
        # if self.config.get("guardian_api_key"):
        #     self.guardian_connector = GuardianAPIConnector(self.config["guardian_api_key"])
    
    async def search_all_apis(
        self,
        query: str,
        max_articles: int = 50,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[NewsArticle]:
        """
        Search across all available news APIs.
        
        Args:
            query: Search query
            max_articles: Maximum articles to return
            from_date: From date filter
            to_date: To date filter
            
        Returns:
            List of news articles from all APIs
        """
        all_articles = []
        tasks = []
        
        # NewsAPI search
        if self.newsapi_connector:
            task = self._search_newsapi(
                query, max_articles // 2, from_date, to_date
            )
            tasks.append(task)
        
        # Add more API searches here
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"API search failed: {result}")
        
        # Remove duplicates and limit results
        unique_articles = self._deduplicate_by_url(all_articles)
        return unique_articles[:max_articles]
    
    async def _search_newsapi(
        self,
        query: str,
        max_articles: int,
        from_date: Optional[datetime],
        to_date: Optional[datetime]
    ) -> List[NewsArticle]:
        """Search using NewsAPI."""
        try:
            if not self.newsapi_connector:
                logger.warning("NewsAPI connector not available (missing API key)")
                return []
                
            async with self.newsapi_connector as connector:
                response = await connector.search_everything(
                    query=query,
                    from_date=from_date,
                    to_date=to_date,
                    page_size=max_articles,
                    sort_by="relevancy"
                )
                
                if response.status == "ok":
                    return response.articles
                else:
                    logger.error(f"NewsAPI error: {response.error_message}")
                    return []
                    
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            return []
    
    def _deduplicate_by_url(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Remove duplicate articles based on URL.
        
        Args:
            articles: List of articles to deduplicate
            
        Returns:
            List of unique articles
        """
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles


async def fetch_news_from_apis(
    query: str,
    max_articles: int = 50,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[NewsArticle]:
    """
    Convenience function to fetch news from all available APIs.
    
    Args:
        query: Search query
        max_articles: Maximum articles to return
        from_date: From date filter
        to_date: To date filter
        
    Returns:
        List of news articles
    """
    aggregator = NewsAPIAggregator()
    return await aggregator.search_all_apis(
        query=query,
        max_articles=max_articles,
        from_date=from_date,
        to_date=to_date
    )