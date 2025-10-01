# LUMIR AI RAG System - Quick Start Guide

## 🚀 Overview

Advanced RAG (Retrieval-Augmented Generation) system for trading knowledge with:
- **Qwen3-Embedding-0.6B** (1024 dimensions) with Apple Silicon GPU support
- **Semantic chunking** for intelligent document processing
- **Hybrid search** with cross-encoder reranking
- **576 knowledge chunks** from 12 trading documents

## 📋 Prerequisites

### Hardware Requirements
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for models and documents
- **GPU**: Optional (Apple Silicon MPS, CUDA, or CPU fallback)

### Software Requirements
```bash
# Python 3.10+ with conda/miniconda
conda create -n rag python=3.10
conda activate rag

# Install dependencies
pip install -r requirements_advanced.txt
```

### Start Qdrant Vector Database
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## 🗂️ Project Structure

```
lumir-agentic/
├── module/
│   ├── advanced_rag_orchestrator.py    # Main RAG system
│   ├── document/
│   │   ├── embedding_manager.py        # Qwen3 embeddings + device auto-detect
│   │   ├── semantic_chunker.py         # Advanced semantic chunking
│   │   └── document_processor.py       # Document processing
│   ├── retrieval/
│   │   ├── hybrid_search.py           # Dense + sparse retrieval
│   │   └── reranker.py                # Cross-encoder reranking
│   └── database/
│       └── qdrant_manager.py          # Vector database operations
├── trading_data/                       # Document storage (12 files)
├── rag_query.py                       # Interactive query interface
├── ingest_documents.py                # Document processing
└── test_rag_*.py                     # Test suites
```

## 🛠️ Quick Setup

### 1. Index Documents
```bash
# Process all trading documents into vector database
python ingest_documents.py
```

Expected output:
```
🚀 Starting Document Ingestion into RAG System
📊 Found 12 documents
✅ Successfully processed: 12
🧩 Total chunks stored: 576
🎉 DOCUMENT INGESTION COMPLETED!
```

### 2. Test the System
```bash
# Run comprehensive tests
python test_rag_no_fallbacks.py

# Test query capabilities  
python test_rag_queries.py
```

### 3. Start Querying

#### Command Line Mode
```bash
python rag_query.py "tại sao tôi hay thoát lệnh quá sớm?"
```

#### Interactive Mode
```bash
python rag_query.py
```

Example queries:
- "stop loss là gì và cách sử dụng?"
- "100 lệnh kỷ luật đầu tiên quan trọng như thế nào?"
- "Pioneer NFT có cơ chế bảo đảm giá trị không?"
- "cách phục hồi tâm lý sau chuỗi lệnh sai"

## 📊 Performance

| Component | Performance | Notes |
|-----------|-------------|-------|
| **Embedding Generation** | ~0.5s | Qwen3-0.6B with Apple Silicon GPU |
| **Document Retrieval** | ~0.02s | Semantic search in 576 chunks |
| **Cross-encoder Reranking** | ~0.8s | High-quality relevance ranking |
| **End-to-End Query** | ~1.5s | Full pipeline with reranking |

## 🎯 Query Results

Example query: *"tại sao tôi hay thoát lệnh quá sớm?"*

```
🔍 Processing question: "tại sao tôi hay thoát lệnh quá sớm?"
📊 Searching top 5 results (threshold: 0.3)
🔄 Reranked 10 results
⏱️ Processing time: 1.730s

🎯 SEARCH RESULTS
📊 Result #1 - Score: 0.5567
📄 Source: 📌 FAQ HANH VI TRADER  - LUMIR AI .docx
🏷️ Type: unknown | Chunk: 87 | Length: 165 chars
Q: Q37. Tôi có thói quen di chuyển stop-loss để né thua, TBI sẽ nói gì?
A: A: Đây là hành vi Stop-Loss Avoidance...
```

## 🔧 Configuration

### Custom Query Parameters
```python
from rag_query import RAGQuerySystem

query_system = RAGQuerySystem()
results = query_system.ask_question(
    question="your question",
    top_n=10,                    # Number of results
    score_threshold=0.3          # Minimum similarity score
)
```

### Advanced RAG Features
```python
from module.advanced_rag_orchestrator import AdvancedRAGOrchestrator

orchestrator = AdvancedRAGOrchestrator(
    qdrant_host="localhost",
    qdrant_port=6333,
    embedding_model="Qwen3-Embedding-0.6B",
    embedding_provider="hf"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   ```bash
   # Check if Qdrant is running
   docker ps | grep qdrant
   
   # Restart Qdrant
   docker restart qdrant
   ```

2. **Model Loading Issues**
   ```bash
   # Check Python environment
   conda activate rag
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
   ```

3. **No Documents Found**
   ```bash
   # Check if documents are indexed
   python -c "
   from module.database.qdrant_manager import QdrantManager
   qdrant = QdrantManager('localhost', 6333, 'lumir_advanced')
   collections = qdrant.client.get_collections()
   print(f'Collections: {len(collections.collections)}')
   "
   ```

### Error Handling

The system has **no fallback mechanisms** - if something goes wrong, it will raise clear errors to help debug:

- `RuntimeError: Failed to generate embedding with HuggingFace`
- `ConnectionError: Qdrant connection refused`  
- `ValueError: No documents found in database`

## 📈 System Architecture

```
User Query → Embedding Generation → Vector Search → Reranking → Top Results
     ↓              ↓                   ↓           ↓           ↓
Qwen3-0.6B    Semantic Search      Qdrant DB   Cross-encoder  Final Output
```

## 🤝 Contributing

1. **Add new documents**: Place in `trading_data/` and run `ingest_documents.py`
2. **Test changes**: Run `test_rag_no_fallbacks.py` to verify system integrity
3. **Performance**: Monitor query times and reranking quality

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test outputs for detailed error messages  
3. Verify all prerequisites are installed correctly

---

**System Status**: ✅ Production Ready with 576 knowledge chunks and advanced reranking capabilities