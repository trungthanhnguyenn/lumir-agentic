"""
RAG Orchestrator Module
Điều phối toàn bộ hệ thống RAG - Tối ưu cho trading documents
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Import các module cần thiết
from .document.document_processor import DocumentProcessor, DocumentChunk, DocumentInfo
from .document.embedding_manager import EmbeddingManager, EmbeddingManagerFactory
from .database.qdrant_manager import QdrantManager, SearchResult, RAGCollectionManager


@dataclass
class RAGQuery:
    """Query cho hệ thống RAG"""
    text: str
    language: str = "vi"
    collection_filter: Optional[str] = None  # 'faq' hoặc 'knowledge_base'
    chunk_type_filter: Optional[str] = None
    source_file_filter: Optional[str] = None
    document_type_filter: Optional[str] = None  # faq_behavior, faq_training, handbook, presentation
    limit: int = 10
    score_threshold: float = 0.3


@dataclass
class RAGResponse:
    """Response từ hệ thống RAG"""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    metadata: Dict[str, Any]
    collection_used: str
    search_strategy: str


class RAGOrchestrator:
    """
    RAG Orchestrator - Điều phối toàn bộ hệ thống RAG
    Tối ưu cho trading documents với 2 collections chính
    """
    
    def __init__(self, 
                 documents_dir: str = "trading_data/general_infor",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 embedding_model: str = "Qwen3-Embedding-0.6B",
                 embedding_provider: str = "hf"):
        
        self.documents_dir = Path(documents_dir)
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Khởi tạo các components
        self._initialize_components(embedding_model, embedding_provider)
        
        print(f"🚀 RAG Orchestrator initialized")
        print(f"📁 Documents directory: {self.documents_dir}")
        print(f"🗄️ Qdrant: {qdrant_host}:{qdrant_port}")
        print(f"🔤 Embedding: {embedding_provider} - {embedding_model}")
    
    def _initialize_components(self, embedding_model: str, embedding_provider: str):
        """Khởi tạo các components của hệ thống"""
        try:
            # Khởi tạo Qdrant manager
            self.qdrant_manager = QdrantManager(
                host=self.qdrant_host,
                port=self.qdrant_port,
                collection_prefix="lumir_rag"
            )
            
            # Khởi tạo collection manager với 2 collections tối ưu
            self.collection_manager = RAGCollectionManager(self.qdrant_manager)
            
            # Khởi tạo embedding manager với auto-fallback theo môi trường
            # STRICT: Không fallback nếu cấu hình HF bị lỗi
            self.embedding_manager = EmbeddingManager(
                model_name=embedding_model,
                provider=embedding_provider
            )
            
            # Khởi tạo document processor
            self.document_processor = DocumentProcessor(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            print("✅ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
    
    def setup_rag_system(self) -> bool:
        """
        Thiết lập hệ thống RAG
        Tạo 2 collections chính: faq và knowledge_base
        """
        try:
            print("🔧 Setting up RAG system...")
            
            # Thiết lập collections
            if not self.collection_manager.setup_collections():
                print("❌ Failed to setup collections")
                return False
            
            print("✅ RAG system setup completed")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up RAG system: {e}")
            return False
    
    def process_documents(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Xử lý tất cả documents trong thư mục
        Chunking, embedding và upsert vào Qdrant
        """
        try:
            print(f"📄 Processing documents from {self.documents_dir}")
            
            if not self.documents_dir.exists():
                print(f"❌ Documents directory not found: {self.documents_dir}")
                return {"success": False, "error": "Documents directory not found"}
            
            # Tìm tất cả documents
            document_files = self._find_documents()
            if not document_files:
                print("⚠️ No documents found")
                return {"success": False, "error": "No documents found"}
            
            print(f"📚 Found {len(document_files)} documents")
            
            total_chunks = 0
            total_embeddings = 0
            processing_stats = {}
            
            for doc_file in document_files:
                try:
                    print(f"\n📄 Processing: {doc_file.name}")
                    
                    # Xử lý document
                    chunks, doc_info = self.document_processor.process_document(str(doc_file))
                    
                    if not chunks:
                        print(f"⚠️ No chunks generated for {doc_file.name}")
                        continue
                    
                    # Xác định collection phù hợp
                    collection_name = self.collection_manager.get_collection_for_document(doc_file.name)
                    print(f"🗄️ Using collection: {collection_name}")
                    
                    # Tạo embeddings cho chunks
                    chunk_texts = [chunk.content for chunk in chunks]
                    embeddings = self.embedding_manager.get_embeddings_batch(chunk_texts)
                    
                    if len(embeddings) != len(chunks):
                        print(f"⚠️ Embedding count mismatch for {doc_file.name}")
                        continue
                    
                    # Dimension guard: đảm bảo embedding dimension khớp collection (1024)
                    expected_dim = 1024
                    filtered = []
                    for ch, emb in zip(chunks, embeddings):
                        if isinstance(emb, list) and len(emb) == expected_dim:
                            filtered.append((ch, emb))
                    if len(filtered) != len(chunks):
                        print(f"⚠️ Some embeddings dropped due to wrong dim. kept={len(filtered)}/{len(chunks)}")
                    kept_chunks = [c for c,_ in filtered]
                    kept_embeddings = [e for _,e in filtered]
                    # Chuẩn bị points cho Qdrant
                    points = self._prepare_qdrant_points(kept_chunks, kept_embeddings, doc_file, collection_name)
                    
                    # Upsert vào Qdrant
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
                        
                        print(f"✅ {doc_file.name}: {len(chunks)} chunks processed")
                    else:
                        print(f"❌ Failed to upsert {doc_file.name}")
                
                except Exception as e:
                    print(f"❌ Error processing {doc_file.name}: {e}")
                    continue
            
            # Tạo kết quả tổng hợp
            result = {
                "success": True,
                "total_documents": len(document_files),
                "total_chunks": total_chunks,
                "total_embeddings": total_embeddings,
                "processing_stats": processing_stats,
                "collections_used": list(self.collection_manager.collections_config.keys())
            }
            
            print(f"\n🎉 Document processing completed:")
            print(f"📊 Total chunks: {total_chunks}")
            print(f"🔤 Total embeddings: {total_embeddings}")
            print(f"🗄️ Collections: {result['collections_used']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in document processing: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_documents(self) -> List[Path]:
        """Tìm tất cả documents trong thư mục"""
        supported_extensions = {'.docx', '.pptx', '.pdf', '.txt'}
        documents = []
        
        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.append(file_path)
        
        return sorted(documents)
    
    def _determine_collection(self, doc_file: Path) -> str:
        """
        Xác định collection phù hợp cho document
        Sử dụng collection manager để quyết định
        """
        return self.collection_manager.get_collection_for_document(doc_file.name)
    
    def _prepare_qdrant_points(self, chunks: List[DocumentChunk], 
                               embeddings: List[List[float]], 
                               doc_file: Path,
                               collection_name: str) -> List[Dict[str, Any]]:
        """
        Chuẩn bị points cho Qdrant với metadata đầy đủ
        """
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Tạo payload metadata
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
            
            # Thêm metadata từ chunk
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
        """Xác định loại document từ tên file"""
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
        Query hệ thống RAG
        Tự động xác định collection phù hợp và thực hiện tìm kiếm
        """
        start_time = time.time()
        
        try:
            print(f"🔍 Querying RAG system: {query.text[:100]}...")
            
            # Tạo embedding cho query
            query_embedding = self.embedding_manager.get_embedding(query.text)
            
            # Xác định collections để tìm kiếm
            search_collections = self._determine_search_collections(query)
            
            # Thực hiện tìm kiếm
            all_results = []
            collection_used = "mixed"
            search_strategy = "multi_collection"
            
            if len(search_collections) == 1:
                # Tìm kiếm trong 1 collection
                collection_used = search_collections[0]
                search_strategy = "single_collection"
                
                results = self._search_in_collection(
                    collection_used, query_embedding, query
                )
                all_results.extend(results)
            else:
                # Tìm kiếm trong nhiều collections
                for collection in search_collections:
                    results = self._search_in_collection(
                        collection, query_embedding, query
                    )
                    all_results.extend(results)
                
                # Sắp xếp kết quả theo score
                all_results.sort(key=lambda x: x.score, reverse=True)
                all_results = all_results[:query.limit]
            
            # Tính toán thời gian xử lý
            processing_time = time.time() - start_time
            
            # Tạo response
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
            
            print(f"✅ RAG query completed: {len(all_results)} results in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            print(f"❌ Error in RAG query: {e}")
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
        Xác định collections để tìm kiếm dựa trên query
        """
        # Nếu có filter cụ thể
        if query.collection_filter:
            return [query.collection_filter]
        # Mặc định: tìm cả 2 collections để tối đa recall
        return ["faq", "knowledge_base"]
    
    def _search_in_collection(self, collection_name: str, query_embedding: List[float], 
                             query: RAGQuery) -> List[SearchResult]:
        """
        Tìm kiếm trong collection cụ thể
        """
        try:
            # Đơn giản hoá: bỏ filter và ngưỡng để xác nhận dữ liệu/recall
            results = self.qdrant_manager.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=max(30, query.limit),
                score_threshold=None
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching in collection {collection_name}: {e}")
            return []
    
    def _create_search_filter(self, query: RAGQuery) -> Dict[str, Any]:
        """
        Tạo filter conditions cho tìm kiếm
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
        """Lấy trạng thái hệ thống RAG"""
        try:
            # Trạng thái collections
            collection_status = self.collection_manager.get_collection_status()
            
            # Thống kê documents
            document_files = self._find_documents()
            document_stats = {
                "total_files": len(document_files),
                "file_types": {},
                "document_types": {}
            }
            
            for doc_file in document_files:
                # File types
                ext = doc_file.suffix.lower()
                document_stats["file_types"][ext] = document_stats["file_types"].get(ext, 0) + 1
                
                # Document types
                doc_type = self._get_document_type_from_filename(doc_file.name)
                document_stats["document_types"][doc_type] = document_stats["document_types"].get(doc_type, 0) + 1
            
            # Thông tin embedding model
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
        Tối ưu hệ thống RAG
        """
        try:
            print("🔧 Optimizing RAG system...")
            
            optimization_results = {}
            
            # Tối ưu collections
            for collection_name in self.collection_manager.collections_config.keys():
                try:
                    # Tạo payload indexes cho các field quan trọng
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
            
            # Tối ưu embedding batch size
            if hasattr(self.embedding_manager, 'optimize_batch_size'):
                sample_texts = ["Sample text for optimization"] * 10
                optimal_batch = self.embedding_manager.optimize_batch_size(sample_texts)
                self.embedding_manager.batch_size = optimal_batch
                optimization_results["embedding_batch_size"] = optimal_batch
            
            print("✅ System optimization completed")
            return {
                "success": True,
                "optimization_results": optimization_results
            }
            
        except Exception as e:
            print(f"❌ Error optimizing system: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def clear_system(self) -> bool:
        """
        Xóa toàn bộ dữ liệu trong hệ thống
        """
        try:
            print("🗑️ Clearing RAG system...")
            
            # Xóa tất cả collections
            for collection_name in self.collection_manager.collections_config.keys():
                self.qdrant_manager.delete_collection(collection_name)
            
            print("✅ System cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing system: {e}")
            return False


class RAGOrchestratorFactory:
    """Factory để tạo RAG orchestrator"""
    
    @staticmethod
    def create_orchestrator(documents_dir: str = "trading_data/general_infor",
                           qdrant_host: str = "localhost",
                           qdrant_port: int = 6333,
                           embedding_model: str = "embedding-002",
                           embedding_provider: str = "gemini") -> RAGOrchestrator:
        """Tạo RAG orchestrator với cấu hình cụ thể"""
        return RAGOrchestrator(
            documents_dir=documents_dir,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider
        )
    
    @staticmethod
    def create_optimal_orchestrator(documents_dir: str = "trading_data/general_infor") -> RAGOrchestrator:
        """Tạo RAG orchestrator tối ưu cho trading documents"""
        # Ưu tiên HF nếu có HF_TOKEN
        if os.getenv("HF_TOKEN"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                embedding_model="Qwen3-Embedding-0.6B",
                embedding_provider="hf"
            )
        # Fallback: Gemini nếu có GOOGLE_API_KEY
        if os.getenv("GOOGLE_API_KEY"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                embedding_model="embedding-002",
                embedding_provider="gemini"
            )
        # Cuối cùng: để auto-manager xử lý (sentence-transformers)
        return RAGOrchestrator(
            documents_dir=documents_dir,
            embedding_model="all-MiniLM-L6-v2",
            embedding_provider="sentence_transformers"
        )
    
    @staticmethod
    def create_local_orchestrator(documents_dir: str = "trading_data/general_infor") -> RAGOrchestrator:
        """Tạo RAG orchestrator cho local development"""
        if os.getenv("HF_TOKEN"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                qdrant_host="localhost",
                qdrant_port=6333,
                embedding_model="Qwen3-Embedding-0.6B",
                embedding_provider="hf"
            )
        if os.getenv("GOOGLE_API_KEY"):
            return RAGOrchestrator(
                documents_dir=documents_dir,
                qdrant_host="localhost",
                qdrant_port=6333,
                embedding_model="embedding-002",
                embedding_provider="gemini"
            )
        return RAGOrchestrator(
            documents_dir=documents_dir,
            qdrant_host="localhost",
            qdrant_port=6333,
            embedding_model="all-MiniLM-L6-v2",
            embedding_provider="sentence_transformers"
        )
