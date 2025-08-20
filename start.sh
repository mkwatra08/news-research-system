#!/bin/bash

# News Research System Startup Script

echo "🚀 Starting News Research & Summarization System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/faiss_index logs nginx/ssl

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your API keys before continuing!"
    echo "   Required: OPENAI_API_KEY"
    echo "   Optional: NEWS_API_KEY, PINECONE_API_KEY"
    echo ""
    read -p "Press Enter after editing .env file..."
fi

# Pull Docker images
echo "📦 Pulling Docker images..."
docker-compose pull

# Build the application
echo "🔨 Building application..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Display URLs
echo ""
echo "✅ News Research System is starting up!"
echo ""
echo "📋 Available endpoints:"
echo "   • API Documentation: http://localhost/docs"
echo "   • Health Check: http://localhost/health"
echo "   • Research: POST http://localhost/research"
echo "   • Ask Questions: POST http://localhost/ask"
echo "   • Search: GET http://localhost/search"
echo ""
echo "📊 Monitor with:"
echo "   • View logs: docker-compose logs -f"
echo "   • Check status: docker-compose ps"
echo "   • Stop system: docker-compose down"
echo ""

# Test health endpoint
echo "🏥 Testing health endpoint..."
sleep 5
curl -f http://localhost/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ System is healthy and ready!"
else
    echo "⚠️  System may still be starting up. Check logs with: docker-compose logs -f"
fi

echo ""
echo "🎉 Setup complete! Visit http://localhost/docs to get started."