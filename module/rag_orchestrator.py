import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Import necessary modules
from .document.document_processor import DocumentProcessor, DocumentChunk, DocumentInfo
from .document.embedding_manager import EmbeddingManager, EmbeddingManagerFactory
from .database.qdrant_manager import QdrantManager, SearchResult, RAGCollectionManager


@dataclass
class RAGQuery:
    """Query for RAG system"""
    text: str
    language: str = "vi"
    collection_filter: Optional[str] = None  # 'faq' or 'knowledge_base'
    chunk_type_filter: Optional[str] = None
    source_file_filter: Optional[str] = None
    document_type_filter: Optional[str] = None  # faq_behavior, faq_training, handbook, presentation
    limit: int = 10
    score_threshold: float = 0.3


@dataclass
class RAGResponse:
    """Response from RAG system"""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    metadata: Dict[str, Any]
    collection_used: str
    search_strategy: str


class RAGOrchestrator:
    """
    RAG Orchestrator - Orchestrate the entire RAG system
    Optimized for trading documents with 2 main collections
    """
    
    def __init__(self, 
                 documents_dir: str = "trading_data/general_infor",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 1237,
                 embedding_model: str = "Qwen3-Embedding-0.6B",
                 embedding_provider: str = "hf"):
        
        self.documents_dir = Path(documents_dir)
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Initialize components
        self._initialize_components(embedding_model, embedding_provider)
        
        print(f"RAG Orchestrator initialized")
        print(f"Documents directory: {self.documents_dir}")
        print(f"Qdrant: {qdrant_host}:{qdrant_port}")
        print(f"Embedding: {embedding_provider} - {embedding_model}")
    
    def _initialize_components(self, embedding_model: str, embedding_provider: str):
        """Initialize components of the system"""
        try:
            # Initialize Qdrant manager
            self.qdrant_manager = QdrantManager(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            
            # Initialize collection manager
            self.collection_manager = RAGCollectionManager(self.qdrant_manager)
            
            # Initialize embedding manager
            # STRICT: No fallback if HF configuration fails
            self.embedding_manager = EmbeddingManager(
                model_name=embedding_model,
                provider=embedding_provider
            )
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            print("All components initialized successfully")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise
    
    def setup_rag_system(self) -> bool:
        """
        Setup RAG system
        Create 2 main collections: faq and knowledge_base
        """
        try:
            print("Setting up RAG system...")
            
            # Setup collections
            if not self.collection_manager.setup_collections():
                print("Failed to setup collections")
                return False
            
            print("RAG system setup completed")
            return True
            
        except Exception as e:
            print(f"Error setting up RAG system: {e}")
            return False
    
    def process_documents(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process all documents in the directory
        Chunking, embedding and upsert into Qdrant
        """
        try:
            print(f"Processing documents from {self.documents_dir}")
            
            if not self.documents_dir.exists():
                print(f"Documents directory not found: {self.documents_dir}")
                return {"success": False, "error": "Documents directory not found"}
            
            # Find all documents:
            document_files = self._find_documents()
            if not document_files:
                print("No documents found")
                return {"success": False, "error": "No documents found"}
            
            print(f"Found {len(document_files)} documents")
            
            total_chunks = 0
            total_embeddings = 0
            processing_stats = {}
            
            for doc_file in document_files:
                try:
                    print(f"\nProcessing: {doc_file.name}")
                    
                    # Process document
                    chunks, doc_info = self.document_processor.process_document(str(doc_file))
                    
                    if not chunks:
                        print(f"No chunks generated for {doc_file.name}")
                        continue
                    
                    # Determine appropriate collection
                    collection_name = self.collection_manager.get_collection_for_document(doc_file.name)
                    print(f"Using collection: {collection_name}")
                    
                    # Create embeddings for chunks
                    chunk_texts = [chunk.content for chunk in chunks]
                    embeddings = self.embedding_manager.get_embeddings_batch(chunk_texts)
                    
                    if len(embeddings) != len(chunks):
                        print(f"Embedding count mismatch for {doc_file.name}")
                        continue
                    
                    # Dimension guard: ensure embedding dimension matches collection (1024)
                    expected_dim = 1024
                    filtered = []
                    for ch, emb in zip(chunks, embeddings):
                        if isinstance(emb, list) and len(emb) == expected_dim:
                            filtered.append((ch, emb))
                    if len(filtered) != len(chunks):
                        print(f"Some embeddings dropped due to wrong dim. kept={len(filtered)}/{len(chunks)}")
                    kept_chunks = [c for c,_ in filtered]
                    kept_embeddings = [e for _,e in filtered]
                    # Prepare points for Qdrant
                    points = self._prepare_qdrant_points(kept_chunks, kept_embeddings, doc_file, collection_name)
                    
                    # Upsert into Qdrant
                    if self.qdrant_manager.upsert_points(collection_name, points):
                        total_chunks += len(kept_chunks)
                        total_embeddings += len(kept_embeddings)
                        
                        processing_stats[doc_file.name] = {
                            "chunks": len(chunks),
                            "embeddings": len(embeddings),
                            "collection": collection_name,
                            "processing_time": doc_info.processing_time,
                            "chunking_strategy": doc_info.chunking_strategy,
                            "document_type": doc_info.document_type
                        }
                        
                        print(f"{doc_file.name}: {len(chunks)} chunks processed")
                    else:
                        print(f"Failed to upsert {doc_file.name}")
                
                except Exception as e:
                    print(f"Error processing {doc_file.name}: {e}")
                    continue
            
            # Create summary result
            result = {
                "success": True,
                "total_documents": len(document_files),
                "total_chunks": total_chunks,
                "total_embeddings": total_embeddings,
                "processing_stats": processing_stats,
                "collections_used": list(self.collection_manager.collections_config.keys())
            }
            
            print(f"\nDocument processing completed:")
            print(f"Total chunks: {total_chunks}")
            print(f"Total embeddings: {total_embeddings}")
            print(f"Collections: {result['collections_used']}")
            
            return result
            
        except Exception as e:
            print(f"Error in document processing: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_documents(self) -> List[Path]:
        """Find all documents in the directory"""
        supported_extensions = {'.docx', '.pptx', '.pdf', '.txt'}
        documents = []
        
        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.append(file_path)
        
        return sorted(documents)
    
    def _determine_collection(self, doc_file: Path) -> str:
        """
        Determine appropriate collection for document
        Use collection manager to decide
        """
        return self.collection_manager.get_collection_for_document(doc_file.name)
    
    def _prepare_qdrant_points(self, chunks: List[DocumentChunk], 
                               embeddings: List[List[float]], 
                               doc_file: Path,
                               collection_name: str) -> List[Dict[str, Any]]:
        """
        Prepare points for Qdrant with full metadata
        """
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create payload metadata
            payload = {
                "content": chunk.content,
                "source_file": chunk.source_file,
                "document_type": self._get_document_type_from_filename(doc_file.name),
                "chunk_type": chunk.chunk_type,
                "chunk_strategy": chunk.chunk_strategy,
                "header_level": chunk.header_level,
                "qa_count": chunk.qa_count,
                "token_count": chunk.token_count,
                "language": chunk.language,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "collection": collection_name,
                "processing_timestamp": time.time()
            }
            
            # Add metadata from chunk
            if chunk.metadata:
                payload.update(chunk.metadata)
            
            point = {
                "id": chunk.chunk_id,
                "vector": embedding,
                "payload": payload
            }
            
            points.append(point)
        
        return points
    
    def _get_document_type_from_filename(self, filename: str) -> str:
        """Determine document type from filename"""
        filename_lower = filename.lower()
        
        if "faq_behavior" in filename_lower:
            return "faq_behavior"
        elif "faq_training" in filename_lower:
            return "faq_training"
        elif "handbook" in filename_lower:
            return "handbook"
        elif "present" in filename_lower:
            return "presentation"
        else:
            return "unknown"
    
    def query_rag(self, query: RAGQuery) -> RAGResponse:
        """
        Query RAG system
        Automatically determine appropriate collection and perform search
        """
        start_time = time.time()
        
        try:
            print(f"Querying RAG system: {query.text[:100]}...")
            
            # Create embedding for query
            query_embedding = self.embedding_manager.get_embedding(query.text)
            
            # Determine collections for search
            search_collections = self._determine_search_collections(query)
            
            # Perform search
            all_results = []
            collection_used = "mixed"
            search_strategy = "multi_collection"
            
            if len(search_collections) == 1:
                # Search in 1 collection
                collection_used = search_collections[0]
                search_strategy = "single_collection"
                
                results = self._search_in_collection(
                    collection_used, query_embedding, query
                )
                all_results.extend(results)
            else:
                # Search in multiple collections
                for collection in search_collections:
                    results = self._search_in_collection(
                        collection, query_embedding, query
                    )
                    all_results.extend(results)
                
                # Sort results by score
                all_results.sort(key=lambda x: x.score, reverse=True)
                all_results = all_results[:query.limit]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create response
            response = RAGResponse(
                query=query.text,
                results=all_results,
                total_results=len(all_results),
                processing_time=processing_time,
                metadata={
                    "collections_searched": search_collections,
                    "query_language": query.language,
                    "score_threshold": query.score_threshold
                },
                collection_used=collection_used,
                search_strategy=search_strategy
            )
            
            print(f"RAG query completed: {len(all_results)} results in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            # Return empty response on error
            return RAGResponse(
                query=query.text,
                results=[],
                total_results=0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
                collection_used="error",
                search_strategy="error"
            )
    
    def _determine_search_collections(self, query: RAGQuery) -> List[str]:
        """
        Determine collections for search based on query
        """
        # If there is a specific filter
        if query.collection_filter:
            return [query.collection_filter]
        # Default: search both collections to maximize recall
        return ["faq", "knowledge_base"]
    
    def _search_in_collection(self, collection_name: str, query_embedding: List[float], 
                             query: RAGQuery) -> List[SearchResult]:
        """
        Search in specific collection
        """
        try:
            # Simplify: remove filter and threshold to confirm data/recall
            results = self.qdrant_manager.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=max(30, query.limit),
                score_threshold=None
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching in collection {collection_name}: {e}")
            return []
    
    def _create_search_filter(self, query: RAGQuery) -> Dict[str, Any]:
        """
        Create filter conditions for search
        """
        filter_conditions = {}
        
        if query.chunk_type_filter:
            filter_conditions["chunk_type"] = query.chunk_type_filter
        
        if query.source_file_filter:
            filter_conditions["source_file"] = query.source_file_filter
        
        if query.document_type_filter:
            filter_conditions["document_type"] = query.document_type_filter
        
        return filter_conditions
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system status"""
        try:
            # Collection status
            collection_status = self.collection_manager.get_collection_status()
            
            # Statistics documents
            document_files = self._find_documents()
            document_stats = {
                "total_files": len(document_files),
                "file_types": {},
                "document_types": {}
            }
            
            for doc_file in document_files:
                # File types:
                ext = doc_file.suffix.lower()
                document_stats["file_types"][ext] = document_stats["file_types"].get(ext, 0) + 1
                
                # Document types:
                doc_type = self._get_document_type_from_filename(doc_file.name)
                document_stats["document_types"][doc_type] = document_stats["document_types"].get(doc_type, 0) + 1
            
            # Embedding model info
            embedding_info = self.embedding_manager.get_model_info()
            
            return {
                "status": "operational",
                "collections": collection_status,
                "documents": document_stats,
                "embedding_model": embedding_info,
                "qdrant_connection": {
                    "host": self.qdrant_host,
                    "port": self.qdrant_port,
                    "connected": True
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def optimize_system(self) -> Dict[str, Any]:
        """
        Optimize RAG system
        """
        try:
            print("Optimizing RAG system...")
            
            optimization_results = {}
            
            # Optimize collections
            for collection_name in self.collection_manager.collections_config.keys():
                try:
                    # Create payload indexes for important fields
                    index_fields = [
                        ("document_type", "keyword"),
                        ("chunk_type", "keyword"),
                        ("chunk_strategy", "keyword"),
                        ("source_file", "keyword")
                    ]
                    
                    for field_name, field_type in index_fields:
                        self.qdrant_manager.create_payload_index(collection_name, field_name, field_type)
                    
                    optimization_results[collection_name] = "indexes_created"
                    
                except Exception as e:
                    optimization_results[collection_name] = f"error: {e}"
            
            # Optimize embedding batch size
            if hasattr(self.embedding_manager, 'optimize_batch_size'):
                sample_texts = ["Sample text for optimization"] * 10
                optimal_batch = self.embedding_manager.optimize_batch_size(sample_texts)
                self.embedding_manager.batch_size = optimal_batch
                optimization_results["embedding_batch_size"] = optimal_batch
            
            print("System optimization completed")
            return {
                "success": True,
                "optimization_results": optimization_results
            }
            
        except Exception as e:
            print(f"Error optimizing system: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def clear_system(self) -> bool:
        """
        Clear all data in the system
        """
        try:
            print("Clearing RAG system...")
            
            # Delete all collections
            for collection_name in self.collection_manager.collections_config.keys():
                self.qdrant_manager.delete_collection(collection_name)
            
            print("System cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing system: {e}")
            return False


class RAGOrchestratorFactory:
    """Factory to create RAG orchestrator"""
    
    @staticmethod
    def create_orchestrator(documents_dir: str = "trading_data/general_infor",
                           qdrant_host: str = "localhost",
                           qdrant_port: int = 1237,
                           embedding_model: str = "embedding-002",
                           embedding_provider: str = "gemini") -> RAGOrchestrator:
        """Create RAG orchestrator with specific configuration"""
        return RAGOrchestrator(
            documents_dir=documents_dir,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider
        )
    
    @staticmethod
    def create_optimal_orchestrator(documents_dir: str = "trading_data/general_infor") -> RAGOrchestrator:
        """Create RAG orchestrator optimized for trading documents"""
        # Prefer HF if HF_TOKEN
        if os.getenv("HF_TOKEN"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                embedding_model="Qwen3-Embedding-0.6B",
                embedding_provider="hf"
            )
        # Fallback: Gemini if GOOGLE_API_KEY:
        if os.getenv("GOOGLE_API_KEY"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                embedding_model="embedding-002",
                embedding_provider="gemini"
            )
        # Finally: auto-manager to handle (sentence-transformers):
        return RAGOrchestrator(
            documents_dir=documents_dir,
            embedding_model="all-MiniLM-L6-v2",
            embedding_provider="sentence_transformers"
        )
    
    @staticmethod
    def create_local_orchestrator(documents_dir: str = "trading_data/general_infor") -> RAGOrchestrator:
        """Create RAG orchestrator for local development"""
        if os.getenv("HF_TOKEN"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                qdrant_host="localhost",
                qdrant_port=1237,
                embedding_model="Qwen3-Embedding-0.6B",
                embedding_provider="hf"
            )
        if os.getenv("GOOGLE_API_KEY"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                qdrant_host="localhost",
                qdrant_port=1237,
                embedding_model="embedding-002",
                embedding_provider="gemini"
            )
        return RAGOrchestrator(
            documents_dir=documents_dir,
            qdrant_host="localhost",
            qdrant_port=1237,
            embedding_model="all-MiniLM-L6-v2",
            embedding_provider="sentence_transformers"
        )
