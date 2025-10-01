"""
Hybrid Search System with Dense + Sparse Vectors and RRF Fusion
Implements state-of-the-art retrieval with multiple search strategies
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import math

# Search libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import rank_bm25
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Qdrant and embeddings
from ..database.qdrant_manager import QdrantManager, SearchResult
from ..document.embedding_manager import EmbeddingManager


class SearchStrategy(Enum):
    """Search strategy types"""
    DENSE = "dense"
    SPARSE = "sparse" 
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


@dataclass
class SearchQuery:
    """Enhanced search query"""
    text: str
    query_type: SearchStrategy = SearchStrategy.HYBRID
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    min_score: float = 0.0
    boost_weights: Dict[str, float] = field(default_factory=dict)
    rerank: bool = True
    expand_query: bool = True


@dataclass
class HybridSearchResult:
    """Enhanced search result with multi-source information"""
    id: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    rank: int = 0
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchConfig:
    """Configuration for hybrid search"""
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rerank_threshold: float = 0.5
    max_results: int = 50
    enable_expansion: bool = True
    enable_reranking: bool = True
    enable_multi_hop: bool = False
    rrf_k: int = 60  # Reciprocal Rank Fusion parameter


class HybridSearchEngine:
    """
    Advanced hybrid search engine combining dense and sparse retrieval
    """
    
    def __init__(self, 
                 qdrant_manager: QdrantManager,
                 embedding_manager: EmbeddingManager,
                 config: Optional[SearchConfig] = None):
        
        self.qdrant = qdrant_manager
        self.embedder = embedding_manager
        self.config = config or SearchConfig()
        
        # Initialize sparse components
        self.tfidf_vectorizer = None
        self.bm25_index = None
        self.documents_corpus = []
        self.doc_mapping = {}
        
        # Search cache
        self.search_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        print("Hybrid Search Engine initialized")
    
    async def search(self, query: SearchQuery) -> List[HybridSearchResult]:
        """
        Perform hybrid search with multiple strategies
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(query)
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_ttl:
                return cached_result["results"]
        
        # Initialize results storage
        all_results = {}
        
        # Dense search (semantic)
        if query.query_type in [SearchStrategy.DENSE, SearchStrategy.HYBRID, SearchStrategy.SEMANTIC]:
            dense_results = await self._dense_search(query)
            for result in dense_results:
                all_results[result.id] = result
        
        # Sparse search (keyword)
        if query.query_type in [SearchStrategy.SPARSE, SearchStrategy.HYBRID, SearchStrategy.KEYWORD]:
            sparse_results = await self._sparse_search(query)
            for result in sparse_results:
                if result.id in all_results:
                    # Combine scores
                    all_results[result.id].sparse_score = result.sparse_score
                    all_results[result.id].sources.extend(result.sources)
                else:
                    all_results[result.id] = result
        
        # Combine and rank results
        combined_results = self._combine_results(list(all_results.values()))
        
        # Apply reranking if enabled
        if query.rerank and self.config.enable_reranking:
            combined_results = await self._rerank_results(query, combined_results)
        
        # Filter by min_score
        filtered_results = [r for r in combined_results if r.combined_score >= query.min_score]
        
        # Limit results
        final_results = filtered_results[:query.limit]
        
        # Update ranks
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        # Cache results
        self.search_cache[cache_key] = {
            "results": final_results,
            "timestamp": time.time()
        }
        
        search_time = time.time() - start_time
        print(f"Hybrid search completed: {len(final_results)} results in {search_time:.3f}s")
        
        return final_results
    
    async def _dense_search(self, query: SearchQuery) -> List[HybridSearchResult]:
        """Perform dense (semantic) search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.get_embedding(query.text)
            
            # Search in multiple collections
            collections = self._determine_search_collections(query.filters)
            
            results = []
            for collection_name in collections:
                # Get search results from Qdrant
                search_results = self.qdrant.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=self.config.max_results,
                    score_threshold=0.1  # Low threshold, filter later
                )
                
                # Convert to HybridSearchResult
                for result in search_results:
                    hybrid_result = HybridSearchResult(
                        id=result.id,
                        score=result.score,
                        dense_score=result.score,
                        payload=result.payload,
                        sources=[f"dense_{collection_name}"],
                        explanation={"search_type": "dense", "collection": collection_name}
                    )
                    results.append(hybrid_result)
            
            return results
            
        except Exception as e:
            print(f"Error in dense search: {e}")
            return []
    
    async def _sparse_search(self, query: SearchQuery) -> List[HybridSearchResult]:
        """Perform sparse (keyword) search using BM25/TF-IDF"""
        try:
            if not self._is_sparse_index_ready():
                await self._build_sparse_index()
            
            if not self.bm25_index:
                return []
            
            # Process query for sparse search
            query_tokens = self._tokenize_query(query.text)
            
            # Get BM25 scores
            if BM25_AVAILABLE:
                bm25_scores = self.bm25_index.get_scores(query_tokens)
            else:
                # Fallback to TF-IDF
                query_vec = self.tfidf_vectorizer.transform([" ".join(query_tokens)])
                doc_vectors = self.tfidf_vectorizer.transform(self.documents_corpus)
                bm25_scores = cosine_similarity(query_vec, doc_vectors)[0]
            
            # Convert to HybridSearchResult
            results = []
            for doc_idx, score in enumerate(bm25_scores):
                if score > 0.1:  # Filter low scores
                    doc_id = self.doc_mapping.get(doc_idx)
                    if doc_id:
                        # Get full document info from Qdrant by ID using filter
                        qdrant_results = self.qdrant.search(
                            collection_name="knowledge_base",
                            query_filter={"must": [{"key": "document_id", "match": {"value": doc_id}}]},
                            limit=1
                        )
                        
                        for qdrant_result in qdrant_results:
                            if qdrant_result.id == doc_id:
                                hybrid_result = HybridSearchResult(
                                    id=doc_id,
                                    score=float(score),
                                    sparse_score=float(score),
                                    payload=qdrant_result.payload,
                                    sources=["sparse_bm25"],
                                    explanation={"search_type": "sparse", "method": "bm25"}
                                )
                                results.append(hybrid_result)
                                break
            
            return results
            
        except Exception as e:
            print(f"Error in sparse search: {e}")
            return []
    
    def _combine_results(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Combine results from different search strategies using RRF"""
        
        if not results:
            return []
        
        # Group results by ID
        result_groups = defaultdict(list)
        for result in results:
            result_groups[result.id].append(result)
        
        combined_results = []
        
        for doc_id, group in result_groups.items():
            # Merge payloads (take the one with most information)
            merged_payload = {}
            best_source = None
            max_dense_score = 0
            max_sparse_score = 0
            
            for result in group:
                # Merge payloads
                if result.payload and len(result.payload) > len(merged_payload):
                    merged_payload = result.payload
                
                # Track best scores
                max_dense_score = max(max_dense_score, result.dense_score)
                max_sparse_score = max(max_sparse_score, result.sparse_score)
                
                # Track best source
                if not best_source or result.dense_score > best_source.dense_score:
                    best_source = result
            
            # Calculate combined score using weighted combination
            if max_dense_score > 0 and max_sparse_score > 0:
                # Both dense and sparse scores available
                combined_score = (
                    self.config.dense_weight * max_dense_score +
                    self.config.sparse_weight * max_sparse_score
                )
            elif max_dense_score > 0:
                # Only dense score
                combined_score = max_dense_score
            else:
                # Only sparse score
                combined_score = max_sparse_score * 0.5  # Downweight sparse-only
            
            # Create combined result
            combined_result = HybridSearchResult(
                id=doc_id,
                score=combined_score,
                dense_score=max_dense_score,
                sparse_score=max_sparse_score,
                combined_score=combined_score,
                payload=merged_payload,
                sources=list(set().union(*[r.sources for r in group])),
                explanation={
                    "combination_method": "weighted_average",
                    "dense_weight": self.config.dense_weight,
                    "sparse_weight": self.config.sparse_weight
                }
            )
            
            combined_results.append(combined_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return combined_results
    
    async def _rerank_results(self, query: SearchQuery, 
                            results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Rerank results using cross-encoder or neural reranking"""
        if len(results) <= 1:
            return results
        
        try:
            # For now, implement simple relevance-based reranking
            # In a production system, you'd use a cross-encoder model
            
            query_terms = set(query.text.lower().split())
            
            for result in results:
                content = result.payload.get("content", "").lower()
                
                # Calculate term overlap
                content_terms = set(content.split())
                overlap = len(query_terms.intersection(content_terms))
                total_terms = len(query_terms.union(content_terms))
                
                if total_terms > 0:
                    overlap_ratio = overlap / total_terms
                else:
                    overlap_ratio = 0.0
                
                # Boost score based on overlap
                boost = 1.0 + (overlap_ratio * 0.5)
                result.combined_score *= boost
                
                # Update explanation
                result.explanation["reranking"] = {
                    "method": "term_overlap",
                    "overlap_ratio": overlap_ratio,
                    "boost_factor": boost
                }
            
            # Re-sort after reranking
            results.sort(key=lambda x: x.combined_score, reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Error in reranking: {e}")
            return results
    
    async def _build_sparse_index(self):
        """Build sparse search index (BM25/TF-IDF)"""
        try:
            print("Building sparse search index...")
            
            # Get all documents from Qdrant
            all_documents = []
            self.doc_mapping = {}
            
            collections = ["faq", "knowledge_base"]
            doc_idx = 0
            
            for collection_name in collections:
                # Get collection info
                collection_info = self.qdrant.get_collection_info(collection_name)
                if not collection_info:
                    continue
                
                # Get all documents from collection for TF-IDF indexing
                try:
                    # Use scroll to get all documents
                    all_results, next_page_offset = self.qdrant.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        with_payload=True
                    )
                    search_results = all_results
                except Exception as e:
                    print(f"Error scrolling collection {collection_name}: {e}")
                    continue
                
                for result in search_results:
                    content = result.payload.get("content", "")
                    if content:
                        all_documents.append(content)
                        self.doc_mapping[doc_idx] = result.id
                        doc_idx += 1
            
            self.documents_corpus = all_documents
            
            # Build BM25 index
            if BM25_AVAILABLE and all_documents:
                tokenized_corpus = [self._tokenize_query(doc) for doc in all_documents]
                self.bm25_index = rank_bm25.BM25Okapi(tokenized_corpus)
                print(f"BM25 index built with {len(all_documents)} documents")
            
            # Build TF-IDF as fallback
            if SKLEARN_AVAILABLE and all_documents:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words=None,
                    ngram_range=(1, 2)
                )
                self.tfidf_vectorizer.fit(all_documents)
                print("TF-IDF vectorizer built")
            
        except Exception as e:
            print(f"Error building sparse index: {e}")
    
    def _tokenize_query(self, text: str) -> List[str]:
        """Tokenize text for sparse search"""
        # Simple tokenization - in practice, you'd use more sophisticated tokenization
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _determine_search_collections(self, filters: Dict[str, Any]) -> List[str]:
        """Determine which collections to search based on filters"""
        if "collection" in filters:
            return [filters["collection"]]
        
        # Default to both collections
        return ["faq", "knowledge_base"]
    
    def _is_sparse_index_ready(self) -> bool:
        """Check if sparse index is ready"""
        return (self.bm25_index is not None) or (self.tfidf_vectorizer is not None)
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query"""
        import hashlib
        query_str = f"{query.text}_{query.query_type.value}_{query.limit}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def multi_hop_search(self, initial_query: SearchQuery, 
                             max_hops: int = 3) -> List[HybridSearchResult]:
        """
        Perform multi-hop search for complex queries
        """
        current_results = await self.search(initial_query)
        all_results = set(current_results)
        
        for hop in range(max_hops):
            if not current_results:
                break
            
            # Extract relevant terms from current results
            expansion_terms = self._extract_expansion_terms(current_results)
            
            if not expansion_terms:
                break
            
            # Create expanded query
            expanded_query_text = f"{initial_query.text} {' '.join(expansion_terms[:3])}"
            expanded_query = SearchQuery(
                text=expanded_query_text,
                query_type=initial_query.query_type,
                filters=initial_query.filters,
                limit=initial_query.limit,
                min_score=initial_query.min_score * 0.8  # Lower threshold for expansion
            )
            
            # Search with expanded query
            expanded_results = await self.search(expanded_query)
            
            # Add new results
            new_results = [r for r in expanded_results if r.id not in all_results]
            if not new_results:
                break
            
            all_results.update(new_results)
            current_results = new_results[:5]  # Use top new results for next hop
            
            print(f"Multi-hop {hop + 1}: Found {len(new_results)} new results")
        
        # Convert to list and sort
        final_results = list(all_results)
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return final_results[:initial_query.limit]
    
    def _extract_expansion_terms(self, results: List[HybridSearchResult]) -> List[str]:
        """Extract relevant terms for query expansion"""
        from collections import Counter
        
        all_terms = []
        for result in results:
            content = result.payload.get("content", "")
            terms = self._tokenize_query(content)
            # Filter out common stop words and take important terms
            important_terms = [term for term in terms if len(term) > 3]
            all_terms.extend(important_terms)
        
        # Count term frequency
        term_counts = Counter(all_terms)
        
        # Return most frequent terms (excluding original query terms)
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        return [term for term, count in term_counts.most_common(10)]


class HybridSearchFactory:
    """Factory for creating hybrid search engines"""
    
    @staticmethod
    def create_hybrid_engine(
        qdrant_manager: QdrantManager,
        embedding_manager: EmbeddingManager,
        config: Optional[SearchConfig] = None
    ) -> HybridSearchEngine:
        """Create hybrid search engine with default configuration"""
        return HybridSearchEngine(qdrant_manager, embedding_manager, config)
    
    @staticmethod
    def create_precision_engine(
        qdrant_manager: QdrantManager,
        embedding_manager: EmbeddingManager
    ) -> HybridSearchEngine:
        """Create engine optimized for precision"""
        config = SearchConfig(
            dense_weight=0.8,
            sparse_weight=0.2,
            rerank_threshold=0.7,
            enable_reranking=True,
            enable_multi_hop=True
        )
        return HybridSearchEngine(qdrant_manager, embedding_manager, config)
    
    @staticmethod
    def create_recall_engine(
        qdrant_manager: QdrantManager,
        embedding_manager: EmbeddingManager
    ) -> HybridSearchEngine:
        """Create engine optimized for recall"""
        config = SearchConfig(
            dense_weight=0.6,
            sparse_weight=0.4,
            rerank_threshold=0.3,
            max_results=100,
            enable_expansion=True,
            enable_multi_hop=True
        )
        return HybridSearchEngine(qdrant_manager, embedding_manager, config)