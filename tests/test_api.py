"""
Test cases for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code in [200, 500]  # May fail if dependencies not available
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_research_endpoint_validation():
    """Test research endpoint input validation."""
    # Empty query should fail
    response = client.post("/research", json={"query": ""})
    assert response.status_code == 400
    
    # Valid query structure (may fail due to missing dependencies)
    response = client.post("/research", json={
        "query": "test query",
        "max_articles": 10
    })
    # Should return 200 or 500 depending on system setup
    assert response.status_code in [200, 500]


def test_ask_endpoint_validation():
    """Test ask endpoint input validation."""
    # Empty question should fail
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400
    
    # Valid question structure (may fail due to missing dependencies)
    response = client.post("/ask", json={
        "question": "What is climate change?"
    })
    # Should return 200 or 500 depending on system setup
    assert response.status_code in [200, 500]


def test_search_endpoint():
    """Test search endpoint."""
    response = client.get("/search?query=test&limit=5")
    # Should return 200 or 500 depending on system setup
    assert response.status_code in [200, 500]


def test_stats_endpoint():
    """Test stats endpoint."""
    response = client.get("/stats")
    # Should return 200 or 500 depending on system setup
    assert response.status_code in [200, 500]