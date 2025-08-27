"""
LUMIR RAG System Module
Hệ thống RAG tối ưu cho trading documents
"""

# Core RAG components
from .rag_orchestrator import RAGOrchestrator, RAGOrchestratorFactory, RAGQuery, RAGResponse

# Document processing
from .document.document_processor import DocumentProcessor, DocumentProcessorFactory, DocumentChunk, DocumentInfo

# Embedding management
from .document.embedding_manager import EmbeddingManager, EmbeddingManagerFactory, EmbeddingCache

# Database management
from .database.qdrant_manager import (
    QdrantManager, 
    QdrantManagerFactory, 
    RAGCollectionManager,
    CollectionConfig,
    SearchResult
)

__version__ = "1.0.0"
__author__ = "LUMIR Team"

__all__ = [
    # Core RAG
    "RAGOrchestrator",
    "RAGOrchestratorFactory", 
    "RAGQuery",
    "RAGResponse",
    
    # Document processing
    "DocumentProcessor",
    "DocumentProcessorFactory",
    "DocumentChunk", 
    "DocumentInfo",
    
    # Embedding
    "EmbeddingManager",
    "EmbeddingManagerFactory",
    "EmbeddingCache",
    
    # Database
    "QdrantManager",
    "QdrantManagerFactory",
    "RAGCollectionManager",
    "CollectionConfig",
    "SearchResult"
]
