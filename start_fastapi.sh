#!/bin/bash

# LUMIR-AI FastAPI Server Startup Script
# This script starts the FastAPI server with proper error handling

echo "🚀 Starting LUMIR-AI FastAPI Server..."
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "   Please install Python 3.8+ and try again"
    exit 1
fi

# Check if required files exist
if [ ! -f "fastapi_app.py" ]; then
    echo "❌ fastapi_app.py not found in current directory"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Check if requirements are installed
echo "🔍 Checking dependencies..."
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "⚠️  FastAPI dependencies not found"
    echo "   Installing requirements..."
    if [ -f "requirements_fastapi.txt" ]; then
        pip3 install -r requirements_fastapi.txt
    else
        echo "❌ requirements_fastapi.txt not found"
        echo "   Please install FastAPI manually: pip install fastapi uvicorn"
        exit 1
    fi
fi

# Check if main LUMIR-AI dependencies are available
echo "🔍 Checking LUMIR-AI dependencies..."
if ! python3 -c "import langchain" 2>/dev/null; then
    echo "⚠️  LUMIR-AI dependencies not found"
    echo "   Please install main requirements: pip install -r requirements.txt"
    echo "   Continuing anyway (server may fail to start)..."
fi

# Start the server
echo "✅ Starting FastAPI server..."
echo "📖 API Documentation will be available at: http://localhost:8000/docs"
echo "🔍 API Info will be available at: http://localhost:8000/api/info"
echo "💡 Health check will be available at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"

# Start the server
python3 fastapi_app.py

# Handle exit
echo ""
echo "🛑 FastAPI server stopped"
echo "   To restart, run this script again"
