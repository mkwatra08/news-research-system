"""
Text cleaning and normalization module.
Cleans and preprocesses scraped news articles for storage and analysis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass

from bs4 import BeautifulSoup

from app.scraper.news_scraper import NewsArticle

logger = logging.getLogger(__name__)


@dataclass
class CleanedArticle:
    """
    Represents a cleaned and processed news article.
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
    quality_score: float = 0.0
    readability_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class TextCleaner:
    """
    Main text cleaning and normalization class.
    """
    
    def __init__(self):
        """Initialize the text cleaner with configuration."""
        # Common words to remove (stop words for content analysis)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'long', 'down', 'day', 'did',
            'get', 'come', 'made', 'may', 'part'
        }
        
        # Patterns for cleaning
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\-.,!?;:()\'""]')
        
        # Patterns for content validation
        self.min_word_count = 50
        self.max_word_count = 10000
        self.min_sentence_count = 3
    
    def clean_article(self, article: NewsArticle) -> CleanedArticle:
        """
        Clean and process a single news article.
        
        Args:
            article: Raw news article to clean
            
        Returns:
            CleanedArticle with cleaned and processed content
        """
        try:
            # Clean title
            clean_title = self._clean_title(article.title)
            
            # Clean content
            clean_content = self._clean_content(article.content)
            
            # Clean summary if available
            clean_summary = None
            if article.summary:
                clean_summary = self._clean_text(article.summary)
            
            # Extract tags/keywords
            tags = self._extract_tags(clean_content, clean_title)
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(clean_content, clean_title)
            readability_score = self._calculate_readability_score(clean_content)
            
            # Count words
            word_count = len(clean_content.split()) if clean_content else 0
            
            return CleanedArticle(
                title=clean_title,
                content=clean_content,
                url=article.url,
                source=article.source,
                author=self._clean_author_name(article.author),
                published_at=article.published_at,
                summary=clean_summary,
                image_url=article.image_url,
                tags=tags,
                word_count=word_count,
                language=article.language,
                quality_score=quality_score,
                readability_score=readability_score
            )
            
        except Exception as e:
            logger.error(f"Failed to clean article {article.url}: {e}")
            # Return minimal cleaned version
            return CleanedArticle(
                title=article.title,
                content=article.content or "",
                url=article.url,
                source=article.source,
                quality_score=0.0
            )
    
    def _clean_title(self, title: str) -> str:
        """
        Clean article title.
        
        Args:
            title: Raw title text
            
        Returns:
            Cleaned title
        """
        if not title:
            return ""
        
        # Remove HTML tags
        title = self.html_pattern.sub('', title)
        
        # Remove extra whitespace
        title = self.whitespace_pattern.sub(' ', title).strip()
        
        # Remove common title prefixes/suffixes from news sites
        prefixes_to_remove = [
            r'^BREAKING:?\s*',
            r'^URGENT:?\s*',
            r'^UPDATE:?\s*',
            r'^EXCLUSIVE:?\s*',
        ]
        
        for prefix in prefixes_to_remove:
            title = re.sub(prefix, '', title, flags=re.IGNORECASE)
        
        # Remove trailing source attributions like " - CNN" or " | BBC"
        title = re.sub(r'\s*[-|]\s*[A-Z]{2,10}$', '', title)
        
        return title.strip()
    
    def _clean_content(self, content: str) -> str:
        """
        Clean article content.
        
        Args:
            content: Raw content text
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        # Remove HTML tags and decode entities
        content = self._remove_html(content)
        
        # Remove URLs
        content = self.url_pattern.sub('', content)
        
        # Remove email addresses
        content = self.email_pattern.sub('', content)
        
        # Remove phone numbers
        content = self.phone_pattern.sub('', content)
        
        # Remove advertisement text and boilerplate
        content = self._remove_boilerplate(content)
        
        # Clean whitespace
        content = self.whitespace_pattern.sub(' ', content).strip()
        
        # Remove lines that are too short or look like metadata
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not self._is_metadata_line(line):
                clean_lines.append(line)
        
        content = '\n'.join(clean_lines)
        
        # Final cleanup
        content = self._clean_text(content)
        
        return content
    
    def _clean_text(self, text: str) -> str:
        """
        General text cleaning.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Clean whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def _remove_html(self, text: str) -> str:
        """
        Remove HTML tags and decode entities.
        
        Args:
            text: Text with HTML
            
        Returns:
            Text without HTML
        """
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text(separator=' ')
        except:
            # Fallback to regex if BeautifulSoup fails
            return self.html_pattern.sub('', text)
    
    def _remove_boilerplate(self, content: str) -> str:
        """
        Remove common boilerplate text from news articles.
        
        Args:
            content: Article content
            
        Returns:
            Content without boilerplate
        """
        # Common boilerplate patterns
        boilerplate_patterns = [
            r'Subscribe to our newsletter.*?$',
            r'Follow us on.*?$',
            r'Copyright \d{4}.*?$',
            r'All rights reserved.*?$',
            r'Read more:.*?$',
            r'Related:.*?$',
            r'Advertisement\s*',
            r'ADVERTISEMENT\s*',
            r'This story is developing.*?$',
            r'Breaking news.*?will be updated',
            r'Sign up for.*?newsletter',
            r'Download our app.*?$',
        ]
        
        for pattern in boilerplate_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        return content
    
    def _is_metadata_line(self, line: str) -> bool:
        """
        Check if a line contains metadata rather than content.
        
        Args:
            line: Line of text
            
        Returns:
            True if line appears to be metadata
        """
        # Common metadata patterns
        metadata_patterns = [
            r'^Published:?\s',
            r'^Updated:?\s',
            r'^By:?\s',
            r'^Source:?\s',
            r'^Tags?:?\s',
            r'^Categories?:?\s',
            r'^\d{1,2}:\d{2}\s*(AM|PM)',  # Timestamps
            r'^Share this:',
            r'^Filed under:',
            r'^Photo credit:',
        ]
        
        for pattern in metadata_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _clean_author_name(self, author: Optional[str]) -> Optional[str]:
        """
        Clean author name.
        
        Args:
            author: Raw author name
            
        Returns:
            Cleaned author name or None
        """
        if not author:
            return None
        
        # Remove common prefixes
        author = re.sub(r'^By:?\s*', '', author, flags=re.IGNORECASE)
        
        # Remove email addresses
        author = self.email_pattern.sub('', author)
        
        # Clean whitespace
        author = self.whitespace_pattern.sub(' ', author).strip()
        
        # Return None if author name is too short or contains numbers
        if len(author) < 2 or re.search(r'\d', author):
            return None
        
        return author
    
    def _extract_tags(self, content: str, title: str) -> List[str]:
        """
        Extract relevant tags/keywords from content.
        
        Args:
            content: Article content
            title: Article title
            
        Returns:
            List of extracted tags
        """
        try:
            # Combine title and content for analysis
            text = f"{title} {content}".lower()
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            
            # Filter out stop words
            meaningful_words = [word for word in words if word not in self.stop_words]
            
            # Count word frequency
            word_freq = {}
            for word in meaningful_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most frequent words as tags
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            tags = [word for word, freq in sorted_words[:10] if freq > 1]
            
            return tags
            
        except Exception as e:
            logger.debug(f"Failed to extract tags: {e}")
            return []
    
    def _calculate_quality_score(self, content: str, title: str) -> float:
        """
        Calculate content quality score based on various factors.
        
        Args:
            content: Article content
            title: Article title
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            if not content or not title:
                return 0.0
            
            score = 0.0
            
            # Word count factor (0-0.3)
            word_count = len(content.split())
            if word_count >= self.min_word_count:
                word_score = min(word_count / 500, 1.0) * 0.3
                score += word_score
            
            # Sentence structure factor (0-0.2)
            sentences = re.split(r'[.!?]+', content)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            if len(valid_sentences) >= self.min_sentence_count:
                sentence_score = min(len(valid_sentences) / 20, 1.0) * 0.2
                score += sentence_score
            
            # Title quality factor (0-0.2)
            if 10 <= len(title) <= 100:
                title_score = 0.2
                score += title_score
            
            # Content structure factor (0-0.15)
            paragraphs = content.split('\n')
            valid_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
            if len(valid_paragraphs) >= 2:
                structure_score = min(len(valid_paragraphs) / 10, 1.0) * 0.15
                score += structure_score
            
            # Language quality factor (0-0.15)
            # Simple check for proper capitalization and punctuation
            sentences_with_caps = sum(1 for s in valid_sentences if s[0].isupper())
            if valid_sentences:
                caps_ratio = sentences_with_caps / len(valid_sentences)
                language_score = caps_ratio * 0.15
                score += language_score
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug(f"Failed to calculate quality score: {e}")
            return 0.0
    
    def _calculate_readability_score(self, content: str) -> float:
        """
        Calculate readability score (simplified Flesch Reading Ease).
        
        Args:
            content: Article content
            
        Returns:
            Readability score between 0 and 1
        """
        try:
            if not content:
                return 0.0
            
            # Count sentences and syllables (simplified)
            sentences = re.split(r'[.!?]+', content)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            words = content.split()
            
            if not valid_sentences or not words:
                return 0.0
            
            # Simplified syllable count (count vowels)
            syllables = 0
            for word in words:
                syllables += max(1, len(re.findall(r'[aeiouAEIOU]', word)))
            
            # Simplified Flesch Reading Ease formula
            avg_sentence_length = len(words) / len(valid_sentences)
            avg_syllables_per_word = syllables / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-1 scale
            normalized_score = max(0, min(100, flesch_score)) / 100
            
            return normalized_score
            
        except Exception as e:
            logger.debug(f"Failed to calculate readability score: {e}")
            return 0.0
    
    def is_valid_article(self, article: CleanedArticle) -> bool:
        """
        Check if a cleaned article meets quality standards.
        
        Args:
            article: Cleaned article to validate
            
        Returns:
            True if article is valid, False otherwise
        """
        try:
            # Basic content requirements
            if not article.title or len(article.title.strip()) < 10:
                return False
            
            if not article.content or len(article.content.strip()) < self.min_word_count:
                return False
            
            if article.word_count > self.max_word_count:
                return False
            
            # Quality threshold
            if article.quality_score < 0.3:
                return False
            
            # URL validation
            if not article.url or not article.url.startswith('http'):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Article validation failed: {e}")
            return False


def clean_articles(articles: List[NewsArticle]) -> List[CleanedArticle]:
    """
    Clean a list of news articles.
    
    Args:
        articles: List of raw news articles
        
    Returns:
        List of cleaned articles that pass validation
    """
    cleaner = TextCleaner()
    cleaned_articles = []
    
    for article in articles:
        try:
            cleaned = cleaner.clean_article(article)
            if cleaner.is_valid_article(cleaned):
                cleaned_articles.append(cleaned)
            else:
                logger.debug(f"Article failed validation: {article.url}")
        except Exception as e:
            logger.error(f"Failed to clean article {article.url}: {e}")
    
    logger.info(f"Cleaned {len(cleaned_articles)} out of {len(articles)} articles")
    return cleaned_articles