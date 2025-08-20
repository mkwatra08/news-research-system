"""
Report generation module for creating structured outputs.
Supports multiple formats: JSON, Markdown, HTML, and PDF reports.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import tempfile

from app.ai.summarizer import Summary
from app.ai.rag_engine import RAGResponse
from app.scraper.cleaner import CleanedArticle

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """
    Metadata for generated reports.
    """
    title: str
    query: str
    generated_at: datetime
    total_articles: int
    sources: List[str]
    generation_time: float
    report_type: str
    format: str


@dataclass
class NewsReport:
    """
    Complete news research report.
    """
    metadata: ReportMetadata
    summary: Optional[Summary] = None
    key_findings: List[str] = None
    detailed_analysis: Optional[str] = None
    articles: List[Dict[str, Any]] = None
    rag_responses: List[RAGResponse] = None
    
    def __post_init__(self):
        if self.key_findings is None:
            self.key_findings = []
        if self.articles is None:
            self.articles = []
        if self.rag_responses is None:
            self.rag_responses = []


class ReportGenerator:
    """
    Generates structured reports from news research data.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.supported_formats = ["json", "markdown", "html", "pdf"]
    
    def create_report(
        self,
        query: str,
        articles: List[CleanedArticle],
        summary: Optional[Summary] = None,
        rag_responses: Optional[List[RAGResponse]] = None,
        report_type: str = "comprehensive",
        include_articles: bool = True
    ) -> NewsReport:
        """
        Create a structured news report.
        
        Args:
            query: Original search query
            articles: List of cleaned articles
            summary: Generated summary
            rag_responses: List of RAG responses
            report_type: Type of report (comprehensive, executive, brief)
            include_articles: Whether to include full article data
            
        Returns:
            Structured news report
        """
        try:
            # Calculate generation time
            generation_time = 0.0
            if summary:
                generation_time += summary.generation_time or 0.0
            if rag_responses:
                generation_time += sum(r.generation_time for r in rag_responses)
            
            # Create metadata
            metadata = ReportMetadata(
                title=self._generate_report_title(query, report_type),
                query=query,
                generated_at=datetime.now(),
                total_articles=len(articles),
                sources=list(set(article.source for article in articles)),
                generation_time=generation_time,
                report_type=report_type,
                format="structured"
            )
            
            # Extract key findings
            key_findings = self._extract_key_findings(summary, rag_responses)
            
            # Generate detailed analysis if needed
            detailed_analysis = None
            if report_type == "comprehensive":
                detailed_analysis = self._generate_detailed_analysis(articles, summary)
            
            # Prepare article data
            article_data = []
            if include_articles:
                article_data = self._prepare_article_data(articles)
            
            return NewsReport(
                metadata=metadata,
                summary=summary,
                key_findings=key_findings,
                detailed_analysis=detailed_analysis,
                articles=article_data,
                rag_responses=rag_responses or []
            )
            
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            raise
    
    def export_report(self, report: NewsReport, format: str, output_path: Optional[str] = None) -> str:
        """
        Export report to specified format.
        
        Args:
            report: News report to export
            format: Export format (json, markdown, html, pdf)
            output_path: Optional output file path
            
        Returns:
            Path to exported file or content string
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        try:
            if format == "json":
                return self._export_json(report, output_path)
            elif format == "markdown":
                return self._export_markdown(report, output_path)
            elif format == "html":
                return self._export_html(report, output_path)
            elif format == "pdf":
                return self._export_pdf(report, output_path)
            else:
                raise ValueError(f"Format {format} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to export report to {format}: {e}")
            raise
    
    def _generate_report_title(self, query: str, report_type: str) -> str:
        """Generate a title for the report."""
        type_prefix = {
            "comprehensive": "Comprehensive Analysis",
            "executive": "Executive Summary",
            "brief": "Brief Report"
        }.get(report_type, "News Report")
        
        return f"{type_prefix}: {query}"
    
    def _extract_key_findings(
        self,
        summary: Optional[Summary],
        rag_responses: Optional[List[RAGResponse]]
    ) -> List[str]:
        """Extract key findings from summary and RAG responses."""
        findings = []
        
        # From summary
        if summary and summary.key_points:
            findings.extend(summary.key_points)
        
        # From RAG responses (extract main points)
        if rag_responses:
            for response in rag_responses:
                if response.answer and len(response.answer.split()) > 10:
                    # Extract first sentence as a key finding
                    first_sentence = response.answer.split('.')[0] + '.'
                    if len(first_sentence.split()) >= 5:
                        findings.append(first_sentence)
        
        return findings[:10]  # Limit to top 10 findings
    
    def _generate_detailed_analysis(
        self,
        articles: List[CleanedArticle],
        summary: Optional[Summary]
    ) -> str:
        """Generate detailed analysis section."""
        try:
            analysis_parts = []
            
            # Source analysis
            source_counts = {}
            for article in articles:
                source_counts[article.source] = source_counts.get(article.source, 0) + 1
            
            top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            analysis_parts.append("## Source Analysis")
            analysis_parts.append(f"Analysis based on {len(articles)} articles from {len(source_counts)} sources.")
            analysis_parts.append("Top sources:")
            for source, count in top_sources:
                analysis_parts.append(f"- {source}: {count} articles")
            
            # Timeline analysis
            dated_articles = [a for a in articles if a.published_at]
            if dated_articles:
                dated_articles.sort(key=lambda x: x.published_at)
                earliest = dated_articles[0].published_at
                latest = dated_articles[-1].published_at
                
                analysis_parts.append("\n## Timeline")
                analysis_parts.append(f"Coverage spans from {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
            
            # Quality analysis
            if articles:
                avg_quality = sum(a.quality_score for a in articles) / len(articles)
                high_quality_count = sum(1 for a in articles if a.quality_score > 0.7)
                
                analysis_parts.append("\n## Content Quality")
                analysis_parts.append(f"Average quality score: {avg_quality:.2f}")
                analysis_parts.append(f"High-quality articles: {high_quality_count}/{len(articles)}")
            
            # Summary integration
            if summary:
                analysis_parts.append("\n## Summary Analysis")
                analysis_parts.append(f"Generated summary contains {summary.word_count} words")
                analysis_parts.append(f"Summary confidence: {summary.confidence_score:.2f}")
                if summary.fact_check_notes:
                    analysis_parts.append(f"Fact-check notes: {summary.fact_check_notes}")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate detailed analysis: {e}")
            return "Detailed analysis could not be generated."
    
    def _prepare_article_data(self, articles: List[CleanedArticle]) -> List[Dict[str, Any]]:
        """Prepare article data for inclusion in report."""
        article_data = []
        
        for article in articles:
            data = {
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "author": article.author,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "word_count": article.word_count,
                "quality_score": article.quality_score,
                "readability_score": article.readability_score,
                "tags": article.tags,
                "summary": article.summary
            }
            article_data.append(data)
        
        return article_data
    
    def _export_json(self, report: NewsReport, output_path: Optional[str]) -> str:
        """Export report as JSON."""
        try:
            # Convert report to dictionary
            report_dict = self._report_to_dict(report)
            
            # Serialize to JSON
            json_content = json.dumps(report_dict, indent=2, default=str, ensure_ascii=False)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                return output_path
            else:
                return json_content
                
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise
    
    def _export_markdown(self, report: NewsReport, output_path: Optional[str]) -> str:
        """Export report as Markdown."""
        try:
            md_content = self._generate_markdown_content(report)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                return output_path
            else:
                return md_content
                
        except Exception as e:
            logger.error(f"Markdown export failed: {e}")
            raise
    
    def _export_html(self, report: NewsReport, output_path: Optional[str]) -> str:
        """Export report as HTML."""
        try:
            html_content = self._generate_html_content(report)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return output_path
            else:
                return html_content
                
        except Exception as e:
            logger.error(f"HTML export failed: {e}")
            raise
    
    def _export_pdf(self, report: NewsReport, output_path: Optional[str]) -> str:
        """Export report as PDF."""
        try:
            # First generate HTML content
            html_content = self._generate_html_content(report)
            
            # Use weasyprint to convert HTML to PDF
            try:
                from weasyprint import HTML, CSS
                from weasyprint.text.fonts import FontConfiguration
                
                font_config = FontConfiguration()
                
                # Create CSS for better PDF formatting
                css = CSS(string='''
                    @page {
                        size: A4;
                        margin: 2cm;
                    }
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                    }
                    h1, h2, h3 {
                        color: #2c3e50;
                        page-break-after: avoid;
                    }
                    .metadata {
                        background-color: #f8f9fa;
                        padding: 1em;
                        border-left: 4px solid #007bff;
                        margin-bottom: 2em;
                    }
                    .key-findings {
                        background-color: #fff3cd;
                        padding: 1em;
                        border-left: 4px solid #ffc107;
                        margin: 1em 0;
                    }
                    .article {
                        margin-bottom: 2em;
                        padding-bottom: 1em;
                        border-bottom: 1px solid #dee2e6;
                    }
                    pre {
                        background-color: #f8f9fa;
                        padding: 1em;
                        border-radius: 4px;
                        overflow-wrap: break-word;
                        white-space: pre-wrap;
                    }
                ''', font_config=font_config)
                
                if output_path:
                    HTML(string=html_content).write_pdf(output_path, stylesheets=[css], font_config=font_config)
                    return output_path
                else:
                    # Return path to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        HTML(string=html_content).write_pdf(tmp.name, stylesheets=[css], font_config=font_config)
                        return tmp.name
                        
            except ImportError:
                logger.error("weasyprint not available for PDF generation")
                raise ImportError("weasyprint is required for PDF export. Install with: pip install weasyprint")
                
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise
    
    def _report_to_dict(self, report: NewsReport) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        report_dict = {
            "metadata": asdict(report.metadata),
            "summary": asdict(report.summary) if report.summary else None,
            "key_findings": report.key_findings,
            "detailed_analysis": report.detailed_analysis,
            "articles": report.articles,
            "rag_responses": []
        }
        
        # Convert RAG responses
        for response in report.rag_responses:
            report_dict["rag_responses"].append(asdict(response))
        
        return report_dict
    
    def _generate_markdown_content(self, report: NewsReport) -> str:
        """Generate Markdown content for the report."""
        md_parts = []
        
        # Title and metadata
        md_parts.append(f"# {report.metadata.title}")
        md_parts.append(f"**Query:** {report.metadata.query}")
        md_parts.append(f"**Generated:** {report.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        md_parts.append(f"**Articles Analyzed:** {report.metadata.total_articles}")
        md_parts.append(f"**Sources:** {', '.join(report.metadata.sources[:5])}")
        if len(report.metadata.sources) > 5:
            md_parts.append(f" and {len(report.metadata.sources) - 5} more")
        md_parts.append("")
        
        # Summary
        if report.summary:
            md_parts.append("## Executive Summary")
            md_parts.append(report.summary.content)
            md_parts.append("")
        
        # Key findings
        if report.key_findings:
            md_parts.append("## Key Findings")
            for finding in report.key_findings:
                md_parts.append(f"- {finding}")
            md_parts.append("")
        
        # Detailed analysis
        if report.detailed_analysis:
            md_parts.append(report.detailed_analysis)
            md_parts.append("")
        
        # RAG responses
        if report.rag_responses:
            md_parts.append("## Analysis & Insights")
            for i, response in enumerate(report.rag_responses, 1):
                md_parts.append(f"### Insight {i}")
                md_parts.append(response.answer)
                if response.sources:
                    md_parts.append("**Sources:**")
                    for source in response.sources[:3]:
                        md_parts.append(f"- {source}")
                md_parts.append("")
        
        # Articles (if included and not too many)
        if report.articles and len(report.articles) <= 10:
            md_parts.append("## Source Articles")
            for article in report.articles:
                md_parts.append(f"### {article['title']}")
                md_parts.append(f"**Source:** {article['source']}")
                if article['author']:
                    md_parts.append(f"**Author:** {article['author']}")
                if article['published_at']:
                    md_parts.append(f"**Published:** {article['published_at']}")
                md_parts.append(f"**URL:** {article['url']}")
                md_parts.append("")
        
        return "\n".join(md_parts)
    
    def _generate_html_content(self, report: NewsReport) -> str:
        """Generate HTML content for the report."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{report.metadata.title}</title>",
            "<style>",
            self._get_html_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>"
        ]
        
        # Header
        html_parts.extend([
            f"<h1>{report.metadata.title}</h1>",
            "<div class='metadata'>",
            f"<p><strong>Query:</strong> {report.metadata.query}</p>",
            f"<p><strong>Generated:</strong> {report.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p><strong>Articles Analyzed:</strong> {report.metadata.total_articles}</p>",
            f"<p><strong>Sources:</strong> {', '.join(report.metadata.sources[:5])}",
        ])
        
        if len(report.metadata.sources) > 5:
            html_parts.append(f" and {len(report.metadata.sources) - 5} more")
        html_parts.extend(["</p>", "</div>"])
        
        # Summary
        if report.summary:
            html_parts.extend([
                "<h2>Executive Summary</h2>",
                f"<div class='summary'>{report.summary.content}</div>"
            ])
        
        # Key findings
        if report.key_findings:
            html_parts.append("<h2>Key Findings</h2>")
            html_parts.append("<div class='key-findings'><ul>")
            for finding in report.key_findings:
                html_parts.append(f"<li>{finding}</li>")
            html_parts.extend(["</ul></div>"])
        
        # Detailed analysis
        if report.detailed_analysis:
            # Convert markdown-style headers to HTML
            analysis_html = report.detailed_analysis.replace("## ", "<h3>").replace("\n", "</h3>\n<p>") + "</p>"
            analysis_html = analysis_html.replace("<p></p>", "").replace("<p><h3>", "<h3>")
            html_parts.extend([
                "<h2>Detailed Analysis</h2>",
                f"<div class='analysis'>{analysis_html}</div>"
            ])
        
        # RAG responses
        if report.rag_responses:
            html_parts.append("<h2>Analysis & Insights</h2>")
            for i, response in enumerate(report.rag_responses, 1):
                html_parts.extend([
                    f"<h3>Insight {i}</h3>",
                    f"<div class='insight'>{response.answer}</div>"
                ])
                if response.sources:
                    html_parts.append("<h4>Sources:</h4><ul>")
                    for source in response.sources[:3]:
                        html_parts.append(f"<li><a href='{source}'>{source}</a></li>")
                    html_parts.append("</ul>")
        
        html_parts.extend([
            "</div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML reports."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .container {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: #7f8c8d;
            margin-top: 1.5rem;
        }
        
        .metadata {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-left: 4px solid #007bff;
            margin-bottom: 2rem;
            border-radius: 4px;
        }
        
        .summary {
            background-color: #e8f5e8;
            padding: 1.5rem;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
            border-radius: 4px;
        }
        
        .key-findings {
            background-color: #fff3cd;
            padding: 1.5rem;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
            border-radius: 4px;
        }
        
        .insight {
            background-color: #f0f8ff;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            border-left: 4px solid #6c757d;
        }
        
        .analysis {
            margin: 1rem 0;
        }
        
        ul {
            padding-left: 1.5rem;
        }
        
        li {
            margin-bottom: 0.5rem;
        }
        
        a {
            color: #007bff;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        @media print {
            .container {
                box-shadow: none;
                padding: 1rem;
            }
        }
        """


def generate_report(
    query: str,
    articles: List[CleanedArticle],
    summary: Optional[Summary] = None,
    rag_responses: Optional[List[RAGResponse]] = None,
    report_type: str = "comprehensive",
    export_format: str = "json",
    output_path: Optional[str] = None
) -> str:
    """
    Convenience function to generate and export a news report.
    
    Args:
        query: Original search query
        articles: List of cleaned articles
        summary: Generated summary
        rag_responses: List of RAG responses
        report_type: Type of report
        export_format: Export format
        output_path: Output file path
        
    Returns:
        Path to exported file or content string
    """
    generator = ReportGenerator()
    
    # Create report
    report = generator.create_report(
        query=query,
        articles=articles,
        summary=summary,
        rag_responses=rag_responses,
        report_type=report_type
    )
    
    # Export report
    return generator.export_report(
        report=report,
        format=export_format,
        output_path=output_path
    )