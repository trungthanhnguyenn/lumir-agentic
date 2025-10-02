"""
Advanced RAG Orchestrator - Integration of all enhanced components
State-of-the-art RAG system with semantic chunking, hybrid search, multi-hop reasoning, and reranking
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

# Import existing components
from .document.semantic_chunker import MultiLevelChunker, ChunkingConfig
from .retrieval.hybrid_search import HybridSearchEngine, SearchQuery, SearchStrategy, HybridSearchFactory, SearchConfig, HybridSearchResult
from .retrieval.multi_hop_reasoning import MultiHopReasoner, ReasoningConfig, MultiHopReasonerFactory
from .retrieval.reranker import RerankingConfig, RerankerFactory
from .evaluation.rag_evaluator import EvaluationDataset, RAGEvaluatorFactory

# Import database and embedding managers
from .database.qdrant_manager import QdrantManager
from .document.embedding_manager import EmbeddingManager


@dataclass
class AdvancedRAGConfig:
    """Configuration for advanced RAG system"""
    # Chunking configuration
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    
    # Search configuration
    search_config: SearchConfig = field(default_factory=SearchConfig)
    
    # Reasoning configuration
    reasoning_config: ReasoningConfig = field(default_factory=ReasoningConfig)
    
    # Reranking configuration
    reranking_config: RerankingConfig = field(default_factory=RerankingConfig)
    
    # System settings
    enable_multi_hop: bool = True
    enable_reranking: bool = True
    enable_evaluation: bool = False
    cache_enabled: bool = True
    parallel_processing: bool = True
    max_concurrent_queries: int = 10
    
    # Embedding configuration
    embedding_dimension: int = 1024


@dataclass
class AdvancedRAGQuery:
    """Enhanced query for advanced RAG system"""
    text: str
    language: str = "vi"
    collection_filter: Optional[str] = None
    query_type: SearchStrategy = SearchStrategy.HYBRID
    require_multi_hop: bool = False
    require_reranking: bool = True
    max_context_length: int = 4000
    include_sources: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AdvancedRAGResponse:
    """Enhanced response from advanced RAG system"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    reasoning_path: Optional[Any] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor and optimize system performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics"""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    "avg": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "count": len(times)
                }
        return stats


class AdvancedRAGOrchestrator:
    """
    Advanced RAG Orchestrator integrating all enhanced components
    """
    
    def __init__(self, 
                 documents_dir: str = "trading_data/general_infor",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 1237,
                 embedding_model: str = "Qwen3-Embedding-0.6B",
                 embedding_provider: str = "hf",
                 config: Optional[AdvancedRAGConfig] = None):
        
        self.config = config or AdvancedRAGConfig()
        self.documents_dir = Path(documents_dir)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Query cache
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize components
        self._initialize_components(qdrant_host, int(qdrant_port), embedding_model, embedding_provider)
        
        print("🚀 Advanced RAG Orchestrator initialized")
        print(f"📁 Documents: {self.documents_dir}")
        print(f"🔍 Multi-hop reasoning: {self.config.enable_multi_hop}")
        print(f"🎯 Reranking: {self.config.enable_reranking}")
        print(f"📊 Evaluation: {self.config.enable_evaluation}")
    
    def _initialize_components(self, qdrant_host: str, qdrant_port: int, 
                              embedding_model: str, embedding_provider: str):
        """Initialize all system components"""
        try:
            # 1. Initialize base components
            self.qdrant_manager = QdrantManager(qdrant_host, qdrant_port)
            self.embedding_manager = EmbeddingManager(embedding_model, embedding_provider)
            
            # 2. Initialize advanced chunker
            from .document.semantic_chunker import AdvancedChunkerFactory
            self.semantic_chunker = AdvancedChunkerFactory.create_trading_optimized_chunker()
            self.multi_level_chunker = MultiLevelChunker(self.semantic_chunker)
            
            # 3. Initialize hybrid search engine
            self.hybrid_search = HybridSearchFactory.create_hybrid_engine(
                self.qdrant_manager, 
                self.embedding_manager,
                self.config.search_config
            )
            
            # 4. Initialize multi-hop reasoner
            if self.config.enable_multi_hop:
                self.multi_hop_reasoner = MultiHopReasonerFactory.create_trading_reasoner(
                    self.hybrid_search
                )
            else:
                self.multi_hop_reasoner = None
            
            # 5. Initialize reranker
            if self.config.enable_reranking:
                self.reranker = RerankerFactory.create_trading_reranker()
            else:
                self.reranker = None
            
            # 6. Initialize evaluator
            if self.config.enable_evaluation:
                self.evaluator = RAGEvaluatorFactory.create_trading_evaluator(
                    self.hybrid_search,
                    self.multi_hop_reasoner
                )
            else:
                self.evaluator = None
            
            print("✅ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
    
    async def setup_advanced_system(self) -> bool:
        """Setup the advanced RAG system with enhanced features"""
        try:
            print("🔧 Setting up Advanced RAG System...")
            
            # Setup base collections
            from .rag_orchestrator import RAGCollectionManager
            collection_manager = RAGCollectionManager(self.qdrant_manager)
            
            if not collection_manager.setup_collections():
                print("❌ Failed to setup collections")
                return False
            
            # Process documents with advanced chunking
            if self.documents_dir.exists():
                processing_result = await self._process_documents_advanced()
                if not processing_result["success"]:
                    print(f"❌ Document processing failed: {processing_result.get('error', 'Unknown error')}")
                    return False
                
                print(f"✅ Processed {processing_result['total_chunks']} advanced chunks")
            
            # Optimize system performance
            optimization_result = await self._optimize_system()
            print(f"✅ System optimization completed: {optimization_result}")
            
            print("🎉 Advanced RAG System setup completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error in advanced system setup: {e}")
            return False
    
    async def query_advanced(self, query: AdvancedRAGQuery) -> AdvancedRAGResponse:
        """
        Perform advanced query with multi-hop reasoning and reranking
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_query_cache_key(query)
        if self.config.cache_enabled and cache_key in self.query_cache:
            cached_response = self.query_cache[cache_key]
            if time.time() - cached_response["timestamp"] < self.cache_ttl:
                return cached_response["response"]
        
        self.performance_monitor.start_timer("total_query_time")
        
        try:
            print(f"🔍 Processing advanced query: {query.text[:100]}...")
            
            # Step 1: Initial retrieval
            self.performance_monitor.start_timer("retrieval")
            search_query = SearchQuery(
                text=query.text,
                query_type=query.query_type,
                limit=20,
                rerank=False,  # We'll rerank separately
                expand_query=True
            )
            
            initial_results = await self.hybrid_search.search(search_query)
            retrieval_time = self.performance_monitor.end_timer("retrieval")
            
            # Step 2: Multi-hop reasoning if required
            reasoning_path = None
            reasoning_time = 0.0
            
            if query.require_multi_hop and self.multi_hop_reasoner:
                self.performance_monitor.start_timer("reasoning")
                reasoning_path = await self.multi_hop_reasoner.reason(query.text)
                reasoning_time = self.performance_monitor.end_timer("reasoning")
                
                # Combine reasoning results with initial results
                if reasoning_path.supporting_evidence:
                    reasoning_ids = {r.id for r in reasoning_path.supporting_evidence}
                    initial_results = [r for r in initial_results if r.id not in reasoning_ids]
                    initial_results.extend(reasoning_path.supporting_evidence)
            
            # Step 3: Reranking if required
            reranked_results = initial_results
            reranking_time = 0.0
            
            if query.require_reranking and self.reranker and initial_results:
                self.performance_monitor.start_timer("reranking")
                reranking_result = await self.reranker.rerank(query.text, initial_results)
                reranked_results = reranking_result.reranked_results
                reranking_time = self.performance_monitor.end_timer("reranking")
            
            # Step 4: Generate response
            self.performance_monitor.start_timer("response_generation")
            answer, sources, confidence = self._generate_response(query, reranked_results, reasoning_path)
            response_time = self.performance_monitor.end_timer("response_generation")
            
            # Calculate total processing time
            total_time = time.time() - start_time
            self.performance_monitor.end_timer("total_query_time")
            
            # Create response
            response = AdvancedRAGResponse(
                query=query.text,
                answer=answer,
                sources=sources,
                reasoning_path=reasoning_path,
                confidence_score=confidence,
                processing_time=total_time,
                metadata={
                    "retrieval_time": retrieval_time,
                    "reasoning_time": reasoning_time,
                    "reranking_time": reranking_time,
                    "response_time": response_time,
                    "num_initial_results": len(initial_results),
                    "num_final_results": len(reranked_results),
                    "query_type": query.query_type.value,
                    "multi_hop_used": reasoning_path is not None,
                    "reranking_used": reranking_time > 0
                },
                performance_metrics=self.performance_monitor.get_metrics()
            )
            
            # Cache response
            if self.config.cache_enabled:
                self.query_cache[cache_key] = {
                    "response": response,
                    "timestamp": time.time()
                }
            
            print(f"✅ Query completed in {total_time:.2f}s (confidence: {confidence:.2f})")
            return response
            
        except Exception as e:
            print(f"❌ Error in advanced query: {e}")
            
            # Return error response
            return AdvancedRAGResponse(
                query=query.text,
                answer=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _process_documents_advanced(self) -> Dict[str, Any]:
        """Process documents using advanced semantic chunking"""
        try:
            print("📄 Processing documents with advanced semantic chunking...")
            
            if not self.documents_dir.exists():
                return {"success": False, "error": "Documents directory not found"}
            
            # Find documents
            document_files = self._find_documents()
            if not document_files:
                return {"success": False, "error": "No documents found"}
            
            total_chunks = 0
            processing_stats = {}
            
            for doc_file in document_files:
                try:
                    print(f"📖 Processing: {doc_file.name}")
                    
                    # Read document content
                    from .document.document_processor import DocumentProcessor
                    doc_processor = DocumentProcessor()
                    document_content = doc_processor._read_document(doc_file)
                    
                    # Process document with multi-level chunking
                    chunks, chunking_metadata = self.multi_level_chunker.create_multi_level_chunks(
                        document_content, doc_file
                    )
                    
                    if not chunks:
                        print(f"⚠️ No chunks generated for {doc_file.name}")
                        continue
                    
                    # Determine collection
                    collection_name = self._determine_collection(doc_file.name)
                    
                    # Create embeddings
                    chunk_texts = [chunk.content for chunk in chunks]
                    embeddings = self.embedding_manager.get_embeddings_batch(chunk_texts)
                    
                    # Filter valid embeddings
                    valid_chunks = []
                    valid_embeddings = []
                    
                    for chunk, embedding in zip(chunks, embeddings):
                        if isinstance(embedding, list) and len(embedding) == 1024:  # Expected dimension
                            valid_chunks.append(chunk)
                            valid_embeddings.append(embedding)
                    
                    # Prepare points for Qdrant
                    points = self._prepare_advanced_qdrant_points(
                        valid_chunks, valid_embeddings, doc_file, collection_name
                    )
                    
                    # Upsert to Qdrant
                    if self.qdrant_manager.upsert_points(collection_name, points):
                        total_chunks += len(valid_chunks)
                        
                        processing_stats[doc_file.name] = {
                            "chunks": len(chunks),
                            "valid_chunks": len(valid_chunks),
                            "embeddings": len(valid_embeddings),
                            "collection": collection_name,
                            "chunking_metadata": chunking_metadata
                        }
                        
                        print(f"✅ {doc_file.name}: {len(valid_chunks)} advanced chunks processed")
                    else:
                        print(f"❌ Failed to upsert {doc_file.name}")
                
                except Exception as e:
                    print(f"❌ Error processing {doc_file.name}: {e}")
                    continue
            
            return {
                "success": True,
                "total_documents": len(document_files),
                "total_chunks": total_chunks,
                "processing_stats": processing_stats
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _prepare_advanced_qdrant_points(self, chunks: List, embeddings: List[List[float]], 
                                       doc_file: Path, collection_name: str) -> List[Dict[str, Any]]:
        """Prepare Qdrant points with enhanced metadata"""
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Enhanced payload with semantic information
            payload = {
                "content": chunk.content,
                "source_file": chunk.source_file,
                "chunk_type": chunk.chunk_type,
                "chunk_strategy": chunk.chunk_strategy,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "language": chunk.language,
                "collection": collection_name,
                "processing_timestamp": time.time(),
                # Enhanced semantic metadata
                "semantic_score": chunk.metadata.get("semantic_score", 0.0),
                "cluster_coherence": chunk.metadata.get("cluster_coherence", 0.0),
                "unit_types": chunk.metadata.get("unit_types", []),
                "cluster_id": chunk.metadata.get("cluster_id", 0)
            }
            
            # Add all metadata from chunk
            if chunk.metadata:
                payload.update(chunk.metadata)
            
            point = {
                "id": chunk.chunk_id,
                "vector": embedding,
                "payload": payload
            }
            
            points.append(point)
        
        return points
    
    def _generate_response(self, query: AdvancedRAGQuery, 
                          results: List[HybridSearchResult],
                          reasoning_path: Optional[Any] = None) -> Tuple[str, List[Dict[str, Any]], float]:
        """Generate response from retrieved results"""
        
        if not results:
            return "I couldn't find relevant information to answer your question.", [], 0.0
        
        # Prepare context
        context_parts = []
        sources = []
        
        for result in results[:10]:  # Use top 10 results
            content = result.payload.get("content", "")
            if content:
                context_parts.append(content)
                
                sources.append({
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "score": result.combined_score,
                    "source": result.payload.get("source_file", "Unknown"),
                    "chunk_type": result.payload.get("chunk_type", "unknown")
                })
        
        # Combine with reasoning path if available
        if reasoning_path and reasoning_path.final_answer:
            context_parts.append(f"Reasoning analysis: {reasoning_path.final_answer}")
        
        # Generate answer based on reasoning type
        if reasoning_path:
            answer = reasoning_path.final_answer
            confidence = reasoning_path.confidence_score
        else:
            # Simple synthesis from context
            context = "\n\n".join(context_parts)
            answer = self._synthesize_simple_answer(query.text, context, sources)
            confidence = min(0.8, np.mean([s["score"] for s in sources[:3]]) if sources else 0.0)
        
        # Truncate if too long
        if len(answer) > query.max_context_length:
            answer = answer[:query.max_context_length] + "..."
        
        return answer, sources, confidence
    
    def _synthesize_simple_answer(self, query: str, context: str, sources: List[Dict[str, Any]]) -> str:
        """Simple answer synthesis from context"""
        # This is a simplified implementation
        # In production, you'd use an LLM for better synthesis
        
        answer_parts = [f"Based on the available information about {query}:"]
        
        # Add key points from top sources
        for i, source in enumerate(sources[:3], 1):
            if source["content"]:
                answer_parts.append(f"{i}. {source['content']}")
        
        # Add conclusion
        avg_confidence = np.mean([s["score"] for s in sources[:3]]) if sources else 0.0
        
        if avg_confidence > 0.7:
            answer_parts.append("This information provides a comprehensive answer to your question.")
        elif avg_confidence > 0.4:
            answer_parts.append("This information addresses your question, though additional details may be helpful.")
        else:
            answer_parts.append("While I found some relevant information, you may want to provide more specific details for a more complete answer.")
        
        return " ".join(answer_parts)
    
    async def _optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance"""
        optimization_results = {}
        
        try:
            # Optimize Qdrant collections
            collections = ["faq", "knowledge_base"]
            
            for collection_name in collections:
                # Create payload indexes
                index_fields = [
                    ("semantic_score", "float"),
                    ("cluster_coherence", "float"),
                    ("chunk_strategy", "keyword"),
                    ("cluster_id", "integer")
                ]
                
                for field_name, field_type in index_fields:
                    try:
                        self.qdrant_manager.create_payload_index(collection_name, field_name, field_type)
                    except Exception as e:
                        print(f"Index creation failed for {field_name}: {e}")
                
                optimization_results[collection_name] = "indexes_created"
            
            # Warm up embedding cache
            sample_texts = ["sample query for optimization"] * 5
            self.embedding_manager.get_embeddings_batch(sample_texts)
            
            # Pre-build sparse search index
            if self.hybrid_search:
                await self.hybrid_search._build_sparse_index()
                optimization_results["sparse_index"] = "built"
            
            optimization_results["status"] = "completed"
            
        except Exception as e:
            optimization_results["status"] = f"error: {e}"
        
        return optimization_results
    
    def _find_documents(self) -> List[Path]:
        """Find documents in the directory"""
        supported_extensions = {'.docx', '.pptx', '.pdf', '.txt'}
        documents = []
        
        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.append(file_path)
        
        return sorted(documents)
    
    def _determine_collection(self, filename: str) -> str:
        """Determine collection for document"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['faq', 'behavior', 'training']):
            return "faq"
        elif any(keyword in filename_lower for keyword in ['handbook', 'present', 'guide']):
            return "knowledge_base"
        else:
            return "knowledge_base"
    
    def _generate_query_cache_key(self, query: AdvancedRAGQuery) -> str:
        """Generate cache key for query"""
        import hashlib
        
        query_str = f"{query.text}_{query.query_type.value}_{query.require_multi_hop}_{query.require_reranking}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Base system status
            base_status = {
                "advanced_features": {
                    "semantic_chunking": True,
                    "hybrid_search": True,
                    "multi_hop_reasoning": self.config.enable_multi_hop,
                    "reranking": self.config.enable_reranking,
                    "evaluation": self.config.enable_evaluation
                },
                "performance_metrics": self.performance_monitor.get_metrics(),
                "cache_stats": {
                    "query_cache_size": len(self.query_cache),
                    "cache_enabled": self.config.cache_enabled
                }
            }
            
            # Component status
            if hasattr(self.qdrant_manager, 'get_collection_stats'):
                collection_stats = {}
                for collection in ["faq", "knowledge_base"]:
                    stats = self.qdrant_manager.get_collection_stats(collection)
                    if stats:
                        collection_stats[collection] = stats
                base_status["collections"] = collection_stats
            
            # Embedding model info
            if hasattr(self.embedding_manager, 'get_model_info'):
                base_status["embedding_model"] = self.embedding_manager.get_model_info()
            
            return base_status
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def evaluate_system(self, dataset: Optional[EvaluationDataset] = None) -> Dict[str, Any]:
        """Evaluate system performance"""
        if not self.evaluator:
            return {"error": "Evaluator not enabled"}
        
        try:
            if dataset is None:
                # Create synthetic dataset for evaluation
                dataset = RAGEvaluatorFactory.create_synthetic_dataset(20)
            
            # Run comprehensive evaluation
            report = await self.evaluator.comprehensive_evaluation(dataset, evaluate_reasoning=True)
            
            return {
                "overall_score": report.overall_score,
                "processing_time": report.processing_time,
                "detailed_results": [
                    {
                        "metric": result.metric.value,
                        "score": result.score,
                        "details": result.details
                    }
                    for result in report.results
                ],
                "recommendations": report.recommendations
            }
            
        except Exception as e:
            return {"error": str(e)}


# Factory class
class AdvancedRAGOrchestratorFactory:
    """Factory for creating advanced RAG orchestrators"""
    
    @staticmethod
    def create_trading_orchestrator(
        documents_dir: str = "trading_data/general_info",
        qdrant_host: str = "localhost",
        qdrant_port: int = 1237,
        embedding_model: str = "Qwen3-Embedding-0.6B",
        embedding_provider: str = "hf"
    ) -> AdvancedRAGOrchestrator:
        """Create orchestrator optimized for trading domain with Qwen3 embedding"""
        
        config = AdvancedRAGConfig(
            enable_multi_hop=True,
            enable_reranking=True,
            enable_evaluation=False,  # Set to True for development
            cache_enabled=True,
            parallel_processing=True,
            # Configure for Qwen3-0.6B (1024 dimensions)
            embedding_dimension=1024
        )
        
        return AdvancedRAGOrchestrator(
            documents_dir=documents_dir,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            config=config
        )
    
    @staticmethod
    def create_production_orchestrator(
        documents_dir: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 1237
    ) -> AdvancedRAGOrchestrator:
        """Create orchestrator optimized for production"""
        
        config = AdvancedRAGConfig(
            enable_multi_hop=True,
            enable_reranking=True,
            enable_evaluation=True,
            cache_enabled=True,
            parallel_processing=True,
            max_concurrent_queries=20
        )
        
        return AdvancedRAGOrchestrator(
            documents_dir=documents_dir,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            config=config
        )