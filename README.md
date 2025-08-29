# News Research & Summarization System

An AI-powered news research and summarization system that collects, processes, stores, and analyzes news articles to generate comprehensive summaries and insights.

## 🏗️ Architecture Overview

The system follows a modular microservices architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │    │   FastAPI API    │    │  Orchestrator   │
│                 │───▶│                  │───▶│                 │
│ /research       │    │ /ask /summary    │    │ Workflow Mgmt   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Sources   │    │   AI Processing  │    │    Storage      │
│                 │    │                  │    │                 │
│ • RSS Feeds     │    │ • Summarizer     │    │ • PostgreSQL    │
│ • News APIs     │    │ • RAG Engine     │    │ • FAISS/Pinecone│
│ • Web Scraping  │    │ • Report Gen     │    │ • Redis Cache   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

- **Multi-Source Data Collection**: RSS feeds, News APIs, web scraping
- **AI-Powered Summarization**: GPT-4 based summaries with fact-checking
- **Semantic Search**: FAISS/Pinecone vector database for similarity search
- **RAG (Retrieval-Augmented Generation)**: Context-aware Q&A system
- **Multiple Export Formats**: JSON, Markdown, HTML, PDF reports
- **Caching & Performance**: Redis caching with query optimization
- **Production Ready**: Docker containerization with Nginx reverse proxy

## 📦 Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL + FAISS/Pinecone
- **AI Models**: OpenAI GPT-4, text-embedding-3-small
- **Scraping**: Requests, BeautifulSoup, Newspaper3k
- **Containerization**: Docker + Docker Compose
- **Reverse Proxy**: Nginx
- **Caching**: Redis

## 🛠️ Installation & Setup

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key
- (Optional) News API Key, Pinecone API Key

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd upivot-assignment
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Start the system**
   ```bash
   docker-compose up -d
   ```

4. **Access the API**
   - API Documentation: http://localhost/docs
   - Health Check: http://localhost/health

### Manual Installation

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL**
   ```bash
   # Create database
   createdb news_research
   ```

3. **Configure environment**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   export DATABASE_URL="postgresql://user:pass@localhost:5432/news_research"
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# AI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/news_research

# Vector Store (choose one)
VECTOR_STORE_TYPE=faiss  # or pinecone
FAISS_INDEX_PATH=./data/faiss_index
# PINECONE_API_KEY=your_pinecone_key
# PINECONE_ENVIRONMENT=your_environment

# News APIs (optional)
NEWS_API_KEY=your_news_api_key

# Scraping
MAX_ARTICLES_PER_QUERY=50
SCRAPING_TIMEOUT=30
```

## 📚 API Endpoints

### Research Endpoints

**POST /research**
```json
{
  "query": "climate change",
  "max_articles": 50,
  "summary_type": "general",
  "include_rag_analysis": true,
  "export_format": "json"
}
```

**GET /summary?topic=climate change**
- Quick topic summary using existing data

### Q&A Endpoints

**POST /ask**
```json
{
  "question": "What are the latest developments in renewable energy?",
  "response_style": "informative",
  "max_context_articles": 10
}
```

### Search & Discovery

**GET /search?query=renewable energy&limit=10**
- Search articles in vector database

**GET /stats**
- System statistics and health metrics

## 🔄 Workflow

1. **Query Processing**: User submits research query
2. **Data Collection**: System scrapes news from multiple sources
3. **Content Cleaning**: Articles are cleaned and validated
4. **Storage**: Articles stored in PostgreSQL + vector embeddings
5. **AI Processing**: Generate summaries and insights using GPT-4
6. **RAG Analysis**: Answer specific questions using retrieved context
7. **Report Generation**: Create structured reports in multiple formats

## 📊 Data Flow

```mermaid
graph TD
    A[User Query] --> B[Orchestrator]
    B --> C[News Scraper]
    B --> D[API Connector]
    C --> E[Text Cleaner]
    D --> E
    E --> F[PostgreSQL]
    E --> G[Vector Store]
    F --> H[AI Summarizer]
    G --> I[RAG Engine]
    H --> J[Report Generator]
    I --> J
    J --> K[API Response]
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Test specific components:
```bash
# Test scraping
python -m pytest tests/test_scraper.py

# Test AI modules
python -m pytest tests/test_ai.py

# Test API endpoints
python -m pytest tests/test_api.py
```

## 📈 Performance & Scaling

- **Caching**: Redis for query results and embeddings
- **Background Tasks**: Async processing with FastAPI
- **Database Optimization**: Indexed queries and connection pooling
- **Vector Search**: Optimized FAISS indexing
- **Rate Limiting**: Nginx-based request throttling

For production scaling:
- Use Celery + Redis for background task queue
- Implement database read replicas
- Add horizontal scaling with load balancer
- Consider Pinecone for managed vector search

## 🔒 Security Considerations

- API key management through environment variables
- Input validation and sanitization
- Rate limiting and request throttling
- SQL injection prevention with SQLAlchemy ORM
- CORS configuration for web clients

## 📝 Example Usage

### Research Climate Change News
```python
import requests

response = requests.post('http://localhost/research', json={
    "query": "climate change renewable energy",
    "max_articles": 30,
    "summary_type": "executive",
    "include_rag_analysis": True
})

result = response.json()
print(f"Found {result['articles_count']} articles")
print(f"Summary: {result['summary']['content']}")
```

### Ask Questions
```python
response = requests.post('http://localhost/ask', json={
    "question": "What are the main challenges facing renewable energy adoption?",
    "response_style": "analytical"
})

answer = response.json()
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['confidence_score']}")
```

## 🆘 Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection
docker-compose exec postgres pg_isready -U postgres
```

**Vector Store Issues**
```bash
# Check FAISS index directory
ls -la data/faiss_index/

# Reset vector store
rm -rf data/faiss_index/*
```

**API Key Issues**
```bash
# Verify OpenAI API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Logs

View application logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
```

---
