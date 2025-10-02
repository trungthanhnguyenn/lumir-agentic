#!/usr/bin/env python3
"""
Interactive RAG Query System
Ask questions and get top N results with retrieve and rerank functionality
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

# Add module to path
sys.path.append(str(Path(__file__).parent))

class RAGQuerySystem:
    """Interactive RAG query system with retrieve and rerank"""
    
    def __init__(self, qdrant_host="localhost", qdrant_port=1237, default_collection: str = "lumir_advanced_trading_docs"):
        """
        Initialize RAG system
        
        Args:
            qdrant_host: Qdrant host (default: localhost)
            qdrant_port: Qdrant port (default: 1237)
            default_collection: Default collection name (default: lumir_advanced_trading_docs)
        """
        try:
            from module.advanced_rag_orchestrator import AdvancedRAGOrchestrator
            
            print("🔧 Initializing RAG Query System...")
            self.orchestrator = AdvancedRAGOrchestrator(
                qdrant_host=qdrant_host, 
                qdrant_port=qdrant_port
            )
            
            self.default_collection = default_collection
            self.total_chunks = 0
            
            # Try to get initial count from default collection
            try:
                collection_info = self.orchestrator.qdrant_manager.client.get_collection(default_collection)
                self.total_chunks = collection_info.points_count
                print(f"📊 Default collection '{default_collection}' ready: {self.total_chunks} chunks indexed")
            except Exception as e:
                print(f"⚠️ Default collection '{default_collection}' not found or empty: {e}")
                print("⚠️ Collections will be queried dynamically per request")
                
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            self.orchestrator = None
            self.total_chunks = 0
    
    def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a specific collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict with collection info (points_count, etc.)
        """
        try:
            collection_info = self.orchestrator.qdrant_manager.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": collection_info.points_count,
                "exists": True
            }
        except Exception as e:
            return {
                "name": collection_name,
                "points_count": 0,
                "exists": False,
                "error": str(e)
            }
    
    def ask_question(self, question: str, top_n: int = 10, score_threshold: float = 0.3, collection_name: Optional[str] = None):
        """
        Ask question and get top N results with retrieve and rerank
        
        Args:
            question: Your question
            top_n: Number of top results to return (default: 10)
            score_threshold: Minimum similarity score (default: 0.3)
            collection_name: Collection to query (default: use default_collection from init)
            
        Returns:
            List of results with content, score, and source
        """
        if not self.orchestrator:
            print("❌ RAG system not initialized")
            return []
        
        # Use default collection if not specified
        if collection_name is None:
            collection_name = self.default_collection
        
        # Check if collection exists
        collection_info = self.get_collection_info(collection_name)
        if not collection_info.get("exists"):
            print(f"❌ Collection '{collection_name}' not found in database")
            return []
        
        if collection_info.get("points_count", 0) == 0:
            print(f"❌ Collection '{collection_name}' is empty")
            return []
        
        print(f"\n🔍 Processing question: \"{question}\"")
        print(f"📊 Searching in '{collection_name}' - top {top_n} results (threshold: {score_threshold})")
        
        try:
            start_time = time.time()
            
            # Step 1: Generate embedding for question
            query_embedding = self.orchestrator.embedding_manager.get_embedding(question)
            embedding_time = time.time() - start_time
            
            # Step 2: Initial retrieval from Qdrant
            retrieval_start = time.time()
            initial_results = self.orchestrator.qdrant_manager.search(
                collection_name,
                query_embedding,
                limit=top_n * 5,  # Get more for reranking
                score_threshold=score_threshold
            )
            retrieval_time = time.time() - retrieval_start
            
            # Step 3: Rerank results (if available)
            rerank_start = time.time()
            if hasattr(self.orchestrator, 'reranker') and self.orchestrator.reranker and len(initial_results) > 1:
                try:
                    # Convert SearchResult to HybridSearchResult for reranker
                    from module.retrieval.hybrid_search import HybridSearchResult
                    
                    hybrid_results = []
                    for result in initial_results:
                        payload = result.payload.copy()
                        payload['sources'] = [result.payload.get('source', '')]
                        
                        hybrid_result = HybridSearchResult(
                            id=result.id,
                            score=result.score,
                            payload=payload
                        )
                        hybrid_results.append(hybrid_result)
                    
                    # Rerank using async method - handle event loop properly
                    import asyncio
                    try:
                        # Try to get current event loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're in an async context, we can't use asyncio.run
                            # Fall back to synchronous approach
                            print("⚠️ Running in async context, skipping reranking")
                            raise RuntimeError("Cannot rerank in running event loop")
                        else:
                            reranking_result = asyncio.run(self.orchestrator.reranker.rerank(question, hybrid_results))
                    except RuntimeError:
                        # If event loop is already running or not available, skip reranking
                        print("⚠️ Reranking skipped (async context), using original scores")
                        final_results = initial_results[:top_n]
                    else:
                        # Use reranked results directly - they are already sorted by score (descending)
                        # Just need to map back to SearchResult format
                        final_results = []
                        for reranked_result in reranking_result.reranked_results[:top_n]:
                            # Find corresponding original SearchResult to preserve structure
                            for orig_result in initial_results:
                                if orig_result.id == reranked_result.id:
                                    # Update the score with reranked score
                                    orig_result.score = reranked_result.score
                                    final_results.append(orig_result)
                                    break
                        
                        print(f"🔄 Reranked {len(hybrid_results)} results → Selected top {len(final_results)}")
                    
                except Exception as e:
                    print(f"⚠️ Reranking failed, using original scores: {e}")
                    final_results = initial_results[:top_n]
            else:
                final_results = initial_results[:top_n]
            
            rerank_time = time.time() - rerank_start
            total_time = time.time() - start_time
            
            # Format results
            formatted_results = []
            for i, result in enumerate(final_results):
                formatted_result = {
                    'rank': i + 1,
                    'score': round(result.score, 4),
                    'content': result.payload.get('content', ''),
                    'filename': result.payload.get('filename', 'Unknown'),
                    'document_type': result.payload.get('document_type', 'unknown'),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'char_count': result.payload.get('char_count', 0)
                }
                formatted_results.append(formatted_result)
            
            # Performance summary
            print(f"⏱️  Processing time: {total_time:.3f}s")
            print(f"   - Embedding: {embedding_time:.3f}s")
            print(f"   - Retrieval: {retrieval_time:.3f}s") 
            print(f"   - Reranking: {rerank_time:.3f}s")
            print(f"📋 Found {len(formatted_results)} relevant results")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    async def ask_question_async(self, question: str, top_n: int = 10, score_threshold: float = 0.3, collection_name: Optional[str] = None):
        """
        Async version: Ask question and get top N results with retrieve and rerank
        
        Args:
            question: Your question
            top_n: Number of top results to return (default: 10)
            score_threshold: Minimum similarity score (default: 0.3)
            collection_name: Collection to query (default: use default_collection from init)
            
        Returns:
            List of results with content, score, and source
        """
        if not self.orchestrator:
            print("❌ RAG system not initialized")
            return []
        
        # Use default collection if not specified
        if collection_name is None:
            collection_name = self.default_collection
        
        # Check if collection exists
        collection_info = self.get_collection_info(collection_name)
        if not collection_info.get("exists"):
            print(f"❌ Collection '{collection_name}' not found in database")
            return []
        
        if collection_info.get("points_count", 0) == 0:
            print(f"❌ Collection '{collection_name}' is empty")
            return []
        
        print(f"\n🔍 Processing question: \"{question}\"")
        print(f"📊 Searching in '{collection_name}' - top {top_n} results (threshold: {score_threshold})")
        
        try:
            start_time = time.time()
            
            # Step 1: Generate embedding for question
            query_embedding = self.orchestrator.embedding_manager.get_embedding(question)
            embedding_time = time.time() - start_time
            
            # Step 2: Initial retrieval from Qdrant
            retrieval_start = time.time()
            initial_results = self.orchestrator.qdrant_manager.search(
                collection_name,
                query_embedding,
                limit=top_n * 5,  # Get more for reranking
                score_threshold=score_threshold
            )
            retrieval_time = time.time() - retrieval_start
            
            # Step 3: Rerank results (if available) - ASYNC version
            rerank_start = time.time()
            if hasattr(self.orchestrator, 'reranker') and self.orchestrator.reranker and len(initial_results) > 1:
                try:
                    # Convert SearchResult to HybridSearchResult for reranker
                    from module.retrieval.hybrid_search import HybridSearchResult
                    
                    hybrid_results = []
                    for result in initial_results:
                        payload = result.payload.copy()
                        payload['sources'] = [result.payload.get('source', '')]
                        
                        hybrid_result = HybridSearchResult(
                            id=result.id,
                            score=result.score,
                            payload=payload
                        )
                        hybrid_results.append(hybrid_result)
                    
                    # Rerank using async method directly (we're already in async context)
                    reranking_result = await self.orchestrator.reranker.rerank(question, hybrid_results)
                    
                    # Use reranked results directly - they are already sorted by score (descending)
                    # Just need to map back to SearchResult format
                    final_results = []
                    for reranked_result in reranking_result.reranked_results[:top_n]:
                        # Find corresponding original SearchResult to preserve structure
                        for orig_result in initial_results:
                            if orig_result.id == reranked_result.id:
                                # Update the score with reranked score
                                orig_result.score = reranked_result.score
                                final_results.append(orig_result)
                                break
                    
                    print(f"🔄 Reranked {len(hybrid_results)} results → Selected top {len(final_results)}")
                    
                except Exception as e:
                    print(f"⚠️ Reranking failed, using original scores: {e}")
                    final_results = initial_results[:top_n]
            else:
                final_results = initial_results[:top_n]
            
            rerank_time = time.time() - rerank_start
            total_time = time.time() - start_time
            
            # Format results
            formatted_results = []
            for i, result in enumerate(final_results):
                formatted_result = {
                    'rank': i + 1,
                    'score': round(result.score, 4),
                    'content': result.payload.get('content', ''),
                    'filename': result.payload.get('filename', 'Unknown'),
                    'document_type': result.payload.get('document_type', 'unknown'),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'char_count': result.payload.get('char_count', 0)
                }
                formatted_results.append(formatted_result)
            
            # Performance summary
            print(f"⏱️  Processing time: {total_time:.3f}s")
            print(f"   - Embedding: {embedding_time:.3f}s")
            print(f"   - Retrieval: {retrieval_time:.3f}s") 
            print(f"   - Reranking: {rerank_time:.3f}s")
            print(f"📋 Found {len(formatted_results)} relevant results")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    def format_results(self, results: list):
        """Format and display results nicely"""
        if not results:
            print("❌ No results found")
            return
        
        print("\n" + "=" * 80)
        print("🎯 SEARCH RESULTS")
        print("=" * 80)
        
        for result in results:
            print(f"\n📊 Result #{result['rank']} - Score: {result['score']}")
            print(f"📄 Source: {result['filename']}")
            print(f"🏷️  Type: {result['document_type']} | Chunk: {result['chunk_index']} | Length: {result['char_count']} chars")
            print("-" * 60)
            print(result['content'])
            print("-" * 60)

def interactive_mode():
    """Interactive mode for continuous questioning"""
    query_system = RAGQuerySystem()
    
    if not query_system.orchestrator:
        print("❌ Cannot start interactive mode")
        return
    
    print("\n🎯 RAG Interactive Query System")
    print("=" * 50)
    print("Commands:")
    print("  • Type your question and press Enter")
    print("  • 'quit' or 'exit' to exit")
    print("  • 'help' for usage information")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n❓ Ask your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\n📖 Usage:")
                print("  • Direct question: What is stop loss?")
                print("  • Psychology: Why do I exit trades early?")
                print("  • Training: How important are first 100 trades?")
                print("  • NFT: Does Pioneer NFT have value guarantee?")
                continue
            
            if not question:
                print("⚠️ Please enter a question")
                continue
            
            # Get top 5 results by default
            results = query_system.ask_question(question, top_n=10)
            query_system.format_results(results)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function for command line usage"""
    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        query_system = RAGQuerySystem()
        if query_system.orchestrator:
            results = query_system.ask_question(question, top_n=10)
            query_system.format_results(results)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()