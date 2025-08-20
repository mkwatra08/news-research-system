#!/bin/bash

# Local Development Setup Script
echo "🚀 Setting up News Research System locally..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/faiss_index logs

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your API keys!"
    echo "   Required: OPENAI_API_KEY"
    read -p "Press Enter after editing .env file..."
fi

# Start PostgreSQL (if using Homebrew)
echo "🗄️  Starting PostgreSQL..."
if command -v brew &> /dev/null && brew services list | grep -q postgresql; then
    brew services start postgresql
    
    # Create database if it doesn't exist
    echo "📊 Setting up database..."
    createdb news_research 2>/dev/null || echo "Database already exists"
else
    echo "⚠️  PostgreSQL not found via Homebrew."
    echo "   Please install: brew install postgresql"
    echo "   Or update DATABASE_URL in .env to point to your PostgreSQL instance"
fi

# Start Redis (if using Homebrew)
echo "🔄 Starting Redis..."
if command -v brew &> /dev/null && brew services list | grep -q redis; then
    brew services start redis
else
    echo "⚠️  Redis not found. Installing via Homebrew..."
    brew install redis
    brew services start redis
fi

echo ""
echo "✅ Local setup complete!"
echo ""
echo "🚀 Starting the application..."

# Set environment variables for local development
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/news_research"
export REDIS_URL="redis://localhost:6379/0"

# Start the FastAPI application
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📋 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload