"""
News scraping module for collecting articles from various sources.
Supports RSS feeds, Google News, and direct website scraping.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from newspaper import Article

from app.utils.config import get_scraping_config

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """
    Represents a scraped news article.
    """
    title: str
    content: str
    url: str
    source: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    summary: Optional[str] = None
    image_url: Optional[str] = None
    tags: List[str] = None
    word_count: int = 0
    language: str = "en"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.content:
            self.word_count = len(self.content.split())


class NewsSource:
    """
    Represents a news source configuration.
    """
    
    def __init__(
        self,
        name: str,
        url: str,
        source_type: str = "rss",  # rss, website, api
        selectors: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.url = url
        self.source_type = source_type
        self.selectors = selectors or {}


class NewsScraper:
    """
    Main news scraper that handles multiple sources and methods.
    """
    
    def __init__(self):
        """Initialize the news scraper with configuration."""
        self.config = get_scraping_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.scraped_urls: Set[str] = set()
        
        # Default news sources
        self.default_sources = [
            NewsSource("BBC News", "http://feeds.bbci.co.uk/news/rss.xml", "rss"),
            NewsSource("Reuters", "http://feeds.reuters.com/reuters/topNews", "rss"),
            NewsSource("CNN", "http://rss.cnn.com/rss/edition.rss", "rss"),
            NewsSource("Associated Press", "https://feeds.apnews.com/rss/apf-topnews", "rss"),
            NewsSource("NPR", "https://feeds.npr.org/1001/rss.xml", "rss"),
            NewsSource("The Guardian", "https://www.theguardian.com/world/rss", "rss"),
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config["timeout"]),
            headers={"User-Agent": self.config["user_agent"]}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def scrape_by_query(
        self,
        query: str,
        max_articles: Optional[int] = None,
        date_from: Optional[datetime] = None,
        sources: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """
        Scrape news articles based on a search query.
        
        Args:
            query: Search query
            max_articles: Maximum number of articles to return
            date_from: Only return articles newer than this date
            sources: List of source names to search in
            
        Returns:
            List of scraped news articles
        """
        max_articles = max_articles or self.config["max_articles"]
        
        try:
            # Combine results from different scraping methods
            all_articles = []
            
            # 1. Search Google News
            google_articles = await self._scrape_google_news(query, max_articles // 2)
            all_articles.extend(google_articles)
            
            # 2. Search RSS feeds
            rss_articles = await self._scrape_rss_feeds(query, max_articles // 2, sources)
            all_articles.extend(rss_articles)
            
            # 3. Filter by date if specified
            if date_from:
                all_articles = [
                    article for article in all_articles
                    if article.published_at and article.published_at >= date_from
                ]
            
            # 4. Remove duplicates and limit results
            unique_articles = self._deduplicate_articles(all_articles)
            
            return unique_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Failed to scrape articles for query '{query}': {e}")
            return []
    
    async def _scrape_google_news(self, query: str, max_articles: int) -> List[NewsArticle]:
        """
        Scrape articles from Google News RSS feed.
        
        Args:
            query: Search query
            max_articles: Maximum articles to return
            
        Returns:
            List of news articles
        """
        try:
            # Construct Google News RSS URL with query
            google_rss_url = f"{self.config['google_news_rss']}?q={query}&hl=en&gl=US&ceid=US:en"
            
            async with self.session.get(google_rss_url) as response:
                if response.status != 200:
                    logger.warning(f"Google News RSS returned status {response.status}")
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                articles = []
                for entry in feed.entries[:max_articles]:
                    try:
                        article = await self._parse_rss_entry(entry, "Google News")
                        if article:
                            articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to parse Google News entry: {e}")
                        continue
                
                logger.info(f"Scraped {len(articles)} articles from Google News")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape Google News: {e}")
            return []
    
    async def _scrape_rss_feeds(
        self,
        query: str,
        max_articles: int,
        sources: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """
        Scrape articles from RSS feeds.
        
        Args:
            query: Search query (used for filtering)
            max_articles: Maximum articles to return
            sources: List of source names to include
            
        Returns:
            List of news articles
        """
        try:
            # Filter sources if specified
            target_sources = self.default_sources
            if sources:
                target_sources = [
                    source for source in self.default_sources
                    if source.name.lower() in [s.lower() for s in sources]
                ]
            
            # Scrape from each RSS source
            all_articles = []
            tasks = []
            
            for source in target_sources:
                if source.source_type == "rss":
                    task = self._scrape_single_rss_feed(source, query, max_articles // len(target_sources))
                    tasks.append(task)
            
            # Execute all RSS scraping tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"RSS feed scraping failed: {result}")
            
            return all_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Failed to scrape RSS feeds: {e}")
            return []
    
    async def _scrape_single_rss_feed(
        self,
        source: NewsSource,
        query: str,
        max_articles: int
    ) -> List[NewsArticle]:
        """
        Scrape articles from a single RSS feed.
        
        Args:
            source: News source configuration
            query: Search query for filtering
            max_articles: Maximum articles to return
            
        Returns:
            List of news articles
        """
        try:
            async with self.session.get(source.url) as response:
                if response.status != 200:
                    logger.warning(f"RSS feed {source.name} returned status {response.status}")
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                articles = []
                query_lower = query.lower()
                
                for entry in feed.entries:
                    try:
                        # Filter by query relevance
                        title = entry.get('title', '').lower()
                        summary = entry.get('summary', '').lower()
                        
                        if query_lower not in title and query_lower not in summary:
                            continue
                        
                        article = await self._parse_rss_entry(entry, source.name)
                        if article:
                            articles.append(article)
                            
                        if len(articles) >= max_articles:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse RSS entry from {source.name}: {e}")
                        continue
                
                logger.info(f"Scraped {len(articles)} articles from {source.name}")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape RSS feed {source.name}: {e}")
            return []
    
    async def _parse_rss_entry(self, entry: Dict[str, Any], source_name: str) -> Optional[NewsArticle]:
        """
        Parse a single RSS entry into a NewsArticle.
        
        Args:
            entry: RSS entry dictionary
            source_name: Name of the news source
            
        Returns:
            NewsArticle or None if parsing fails
        """
        try:
            title = entry.get('title', '').strip()
            url = entry.get('link', '').strip()
            
            if not title or not url or url in self.scraped_urls:
                return None
            
            # Parse published date
            published_at = None
            if 'published_parsed' in entry and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6])
            elif 'published' in entry:
                try:
                    published_at = datetime.fromisoformat(entry.published.replace('Z', '+00:00'))
                except:
                    pass
            
            # Get summary/description
            summary = entry.get('summary', '').strip()
            
            # Extract full content using newspaper3k
            content = await self._extract_article_content(url)
            if not content:
                content = summary  # Fallback to summary
            
            # Extract author
            author = entry.get('author', '').strip() or None
            
            self.scraped_urls.add(url)
            
            return NewsArticle(
                title=title,
                content=content,
                url=url,
                source=source_name,
                author=author,
                published_at=published_at,
                summary=summary
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse RSS entry: {e}")
            return None
    
    async def _extract_article_content(self, url: str) -> Optional[str]:
        """
        Extract full article content from URL using newspaper3k.
        
        Args:
            url: Article URL
            
        Returns:
            Article content or None if extraction fails
        """
        try:
            # Use newspaper3k to extract content
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text.strip()) > 100:
                return article.text.strip()
            
            # Fallback: try manual extraction
            return await self._manual_content_extraction(url)
            
        except Exception as e:
            logger.debug(f"Failed to extract content from {url}: {e}")
            return None
    
    async def _manual_content_extraction(self, url: str) -> Optional[str]:
        """
        Manual content extraction using BeautifulSoup as fallback.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted content or None
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                
                # Try common content selectors
                content_selectors = [
                    'article',
                    '.article-content',
                    '.post-content',
                    '.entry-content',
                    '.content',
                    'main',
                    '.main-content'
                ]
                
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        text = content_elem.get_text(strip=True, separator=' ')
                        if len(text) > 200:
                            return text
                
                # Fallback: get all paragraph text
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                
                return text if len(text) > 100 else None
                
        except Exception as e:
            logger.debug(f"Manual content extraction failed for {url}: {e}")
            return None
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Remove duplicate articles based on URL and title similarity.
        
        Args:
            articles: List of articles to deduplicate
            
        Returns:
            List of unique articles
        """
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # Skip if URL already seen
            if article.url in seen_urls:
                continue
            
            # Check for similar titles
            title_words = set(article.title.lower().split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.lower().split())
                # If more than 70% of words overlap, consider it duplicate
                overlap = len(title_words & seen_words)
                if overlap / max(len(title_words), len(seen_words)) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_urls.add(article.url)
                seen_titles.add(article.title)
                unique_articles.append(article)
        
        return unique_articles
    
    async def scrape_trending_topics(self) -> List[str]:
        """
        Scrape trending news topics.
        
        Returns:
            List of trending topics
        """
        try:
            # This is a simplified implementation
            # In practice, you might want to use Google Trends API or similar
            trending_url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
            
            async with self.session.get(trending_url) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                topics = []
                for entry in feed.entries[:10]:
                    title = entry.get('title', '').strip()
                    if title:
                        topics.append(title)
                
                return topics
                
        except Exception as e:
            logger.error(f"Failed to scrape trending topics: {e}")
            return []


async def scrape_news_articles(
    query: str,
    max_articles: int = 50,
    date_from: Optional[datetime] = None,
    sources: Optional[List[str]] = None
) -> List[NewsArticle]:
    """
    Convenience function to scrape news articles.
    
    Args:
        query: Search query
        max_articles: Maximum number of articles
        date_from: Only articles newer than this date
        sources: List of source names to include
        
    Returns:
        List of scraped news articles
    """
    async with NewsScraper() as scraper:
        return await scraper.scrape_by_query(
            query=query,
            max_articles=max_articles,
            date_from=date_from,
            sources=sources
        )