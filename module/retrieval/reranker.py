"""
Advanced Reranking System with Cross-Encoders
Implements state-of-the-art reranking for improved retrieval quality
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

try:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.util import cos_sim
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from .hybrid_search import HybridSearchResult


class RerankingStrategy(Enum):
    """Reranking strategies"""
    CROSS_ENCODER = "cross_encoder"
    LEARNING_TO_RANK = "learning_to_rank"
    GRAPH_BASED = "graph_based"
    DIVERSITY_AWARE = "diversity_aware"
    MULTI_CRITERIA = "multi_criteria"


@dataclass
class RerankingConfig:
    """Configuration for reranking"""
    strategy: RerankingStrategy = RerankingStrategy.MULTI_CRITERIA
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    diversity_weight: float = 0.2
    relevance_weight: float = 0.6
    freshness_weight: float = 0.1
    authority_weight: float = 0.1
    top_k: int = 20
    min_score_threshold: float = 0.1
    enable_parallel: bool = True


@dataclass
class RerankingResult:
    """Result of reranking process"""
    reranked_results: List[HybridSearchResult]
    original_scores: List[float]
    reranked_scores: List[float]
    strategy_used: RerankingStrategy
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossEncoderReranker:
    """Cross-encoder based reranking"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("Sentence transformers not available. Install: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize cross-encoder model"""
        try:
            self.model = CrossEncoder(self.model_name)
            print(f"Cross-encoder model loaded: {self.model_name}")
        except Exception as e:
            print(f"Error loading cross-encoder {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                self.model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
                print(f"Fallback to TinyBERT cross-encoder")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise
    
    def rerank(self, query: str, documents: List[HybridSearchResult]) -> Tuple[List[HybridSearchResult], List[float]]:
        """
        Rerank documents using cross-encoder
        """
        if not documents:
            return documents, []
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = doc.payload.get("content", "")
            # Truncate content if too long
            if len(content) > 512:
                content = content[:512] + "..."
            pairs.append([query, content])
        
        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs)
            
            # Normalize scores to 0-1 range
            scores = np.array(scores)
            if len(scores) > 1:
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    scores = (scores - min_score) / (max_score - min_score)
            
            # Sort by new scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs = [doc for doc, _ in scored_docs]
            reranked_scores = [float(score) for _, score in scored_docs]
            
            return reranked_docs, reranked_scores
            
        except Exception as e:
            print(f"Error in cross-encoder reranking: {e}")
            return documents, [doc.combined_score for doc in documents]


class DiversityAwareReranker:
    """Diversity-aware reranking to maximize result diversity"""
    
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight
    
    def rerank(self, query: str, documents: List[HybridSearchResult]) -> Tuple[List[HybridSearchResult], List[float]]:
        """
        Rerank documents balancing relevance and diversity
        """
        if not documents:
            return documents, []
        
        selected_docs = []
        remaining_docs = documents.copy()
        final_scores = []
        
        # Greedy selection: pick most relevant, then diverse
        while remaining_docs and len(selected_docs) < len(documents):
            if not selected_docs:
                # First document: pick most relevant
                best_doc = max(remaining_docs, key=lambda x: x.combined_score)
                selected_docs.append(best_doc)
                remaining_docs.remove(best_doc)
                final_scores.append(best_doc.combined_score)
            else:
                # Subsequent documents: balance relevance and diversity
                best_doc = None
                best_score = -1
                
                for doc in remaining_docs:
                    # Calculate diversity score
                    diversity_score = self._calculate_diversity_score(doc, selected_docs)
                    
                    # Combined score
                    combined_score = (
                        (1 - self.diversity_weight) * doc.combined_score +
                        self.diversity_weight * diversity_score
                    )
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_doc = doc
                
                if best_doc:
                    selected_docs.append(best_doc)
                    remaining_docs.remove(best_doc)
                    final_scores.append(best_score)
        
        return selected_docs, final_scores
    
    def _calculate_diversity_score(self, doc: HybridSearchResult, selected_docs: List[HybridSearchResult]) -> float:
        """Calculate diversity score against already selected documents"""
        if not selected_docs:
            return 1.0
        
        doc_content = doc.payload.get("content", "").lower()
        min_similarity = 1.0
        
        for selected_doc in selected_docs:
            selected_content = selected_doc.payload.get("content", "").lower()
            
            # Simple content similarity
            similarity = self._content_similarity(doc_content, selected_content)
            min_similarity = min(min_similarity, similarity)
        
        # Diversity = 1 - minimum similarity
        return 1.0 - min_similarity
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using Jaccard similarity"""
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class GraphBasedReranker:
    """Graph-based reranking considering document relationships"""
    
    def __init__(self, authority_weight: float = 0.2):
        self.authority_weight = authority_weight
    
    def rerank(self, query: str, documents: List[HybridSearchResult]) -> Tuple[List[HybridSearchResult], List[float]]:
        """
        Rerank using graph-based authority scores
        """
        if not documents:
            return documents, []
        
        # Build document similarity graph
        similarity_matrix = self._build_similarity_matrix(documents)
        
        # Calculate authority scores (simplified PageRank)
        authority_scores = self._calculate_authority_scores(similarity_matrix)
        
        # Combine with original relevance scores
        final_scores = []
        for i, doc in enumerate(documents):
            combined_score = (
                (1 - self.authority_weight) * doc.combined_score +
                self.authority_weight * authority_scores[i]
            )
            final_scores.append(combined_score)
        
        # Sort by combined scores
        scored_docs = list(zip(documents, final_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, _ in scored_docs]
        reranked_scores = [score for _, score in scored_docs]
        
        return reranked_docs, reranked_scores
    
    def _build_similarity_matrix(self, documents: List[HybridSearchResult]) -> np.ndarray:
        """Build similarity matrix between documents"""
        n = len(documents)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                content1 = documents[i].payload.get("content", "").lower()
                content2 = documents[j].payload.get("content", "").lower()
                
                similarity = self._content_similarity(content1, content2)
                matrix[i][j] = matrix[j][i] = similarity
        
        return matrix
    
    def _calculate_authority_scores(self, similarity_matrix: np.ndarray, iterations: int = 10) -> np.ndarray:
        """Calculate authority scores using simplified PageRank"""
        n = similarity_matrix.shape[0]
        if n == 0:
            return np.array([])
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Power iteration
        for _ in range(iterations):
            new_scores = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and similarity_matrix[i][j] > 0.1:  # Threshold
                        new_scores[i] += scores[j] * similarity_matrix[i][j]
            
            # Normalize
            if new_scores.sum() > 0:
                new_scores = new_scores / new_scores.sum()
            
            scores = 0.85 * new_scores + 0.15 * (1.0 / n)  # Damping factor
        
        return scores
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity"""
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class MultiCriteriaReranker:
    """Multi-criteria reranking combining multiple signals"""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self.cross_encoder = CrossEncoderReranker(config.cross_encoder_model) if CROSS_ENCODER_AVAILABLE else None
        self.diversity_reranker = DiversityAwareReranker(config.diversity_weight)
        self.graph_reranker = GraphBasedReranker(config.authority_weight)
    
    async def rerank(self, query: str, documents: List[HybridSearchResult]) -> RerankingResult:
        """
        Perform multi-criteria reranking
        """
        start_time = time.time()
        
        if not documents:
            return RerankingResult(
                reranked_results=[],
                original_scores=[],
                reranked_scores=[],
                strategy_used=self.config.strategy,
                processing_time=0.0
            )
        
        original_scores = [doc.combined_score for doc in documents]
        
        # Apply different reranking strategies
        reranked_results = documents.copy()
        all_scores = [original_scores]
        
        # 1. Cross-encoder reranking
        if self.cross_encoder:
            try:
                ce_docs, ce_scores = self.cross_encoder.rerank(query, reranked_results)
                reranked_results = ce_docs
                all_scores.append(ce_scores)
            except Exception as e:
                print(f"Cross-encoder reranking failed: {e}")
        
        # 2. Diversity-aware reranking
        try:
            div_docs, div_scores = self.diversity_reranker.rerank(query, reranked_results)
            all_scores.append(div_scores)
        except Exception as e:
            print(f"Diversity reranking failed: {e}")
        
        # 3. Graph-based reranking
        try:
            graph_docs, graph_scores = self.graph_reranker.rerank(query, reranked_results)
            all_scores.append(graph_scores)
        except Exception as e:
            print(f"Graph reranking failed: {e}")
        
        # Combine scores from different strategies
        final_scores = self._combine_scores(all_scores)
        
        # Sort by final scores
        scored_docs = list(zip(reranked_results, final_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        final_results = [doc for doc, _ in scored_docs]
        final_scores_sorted = [score for _, score in scored_docs]
        
        # Filter by threshold
        filtered_results = []
        filtered_scores = []
        for doc, score in zip(final_results, final_scores_sorted):
            if score >= self.config.min_score_threshold:
                filtered_results.append(doc)
                filtered_scores.append(score)
        
        processing_time = time.time() - start_time
        
        return RerankingResult(
            reranked_results=filtered_results[:self.config.top_k],
            original_scores=original_scores,
            reranked_scores=filtered_scores[:self.config.top_k],
            strategy_used=self.config.strategy,
            processing_time=processing_time,
            metadata={
                "num_strategies_used": len(all_scores) - 1,
                "cross_encoder_available": self.cross_encoder is not None,
                "score_improvement": self._calculate_improvement(original_scores, filtered_scores)
            }
        )
    
    def _combine_scores(self, all_scores: List[List[float]]) -> List[float]:
        """Combine scores from multiple reranking strategies"""
        if not all_scores:
            return []
        
        if len(all_scores) == 1:
            return all_scores[0]
        
        # Weighted combination
        weights = [self.config.relevance_weight]  # Original scores
        if len(all_scores) > 1:
            weights.append(0.3)  # Cross-encoder
        if len(all_scores) > 2:
            weights.append(self.config.diversity_weight)  # Diversity
        if len(all_scores) > 3:
            weights.append(self.config.authority_weight)  # Graph authority
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Combine scores
        num_docs = len(all_scores[0])
        combined_scores = []
        
        for i in range(num_docs):
            score_sum = 0.0
            for j, scores in enumerate(all_scores):
                if i < len(scores):
                    score_sum += scores[i] * weights[j]
            combined_scores.append(score_sum)
        
        return combined_scores
    
    def _calculate_improvement(self, original_scores: List[float], 
                              reranked_scores: List[float]) -> float:
        """Calculate improvement in score distribution"""
        if not original_scores or not reranked_scores:
            return 0.0
        
        # Compare top-k average scores
        k = min(5, len(original_scores), len(reranked_scores))
        
        original_avg = np.mean(original_scores[:k])
        reranked_avg = np.mean(reranked_scores[:k])
        
        if original_avg > 0:
            improvement = (reranked_avg - original_avg) / original_avg
        else:
            improvement = reranked_avg
        
        return improvement


class AdvancedReranker:
    """Main reranker that orchestrates different strategies"""
    
    def __init__(self, config: Optional[RerankingConfig] = None):
        self.config = config or RerankingConfig()
        self.multi_criteria_reranker = MultiCriteriaReranker(self.config)
        
        # Reranking cache
        self.rerank_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        print("Advanced Reranker initialized")
    
    async def rerank(self, query: str, documents: List[HybridSearchResult], 
                    strategy: Optional[RerankingStrategy] = None) -> RerankingResult:
        """
        Rerank documents using specified or default strategy
        """
        # Check cache first
        cache_key = self._generate_cache_key(query, documents, strategy or self.config.strategy)
        if cache_key in self.rerank_cache:
            cached_result = self.rerank_cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_ttl:
                return cached_result["result"]
        
        # Select strategy
        selected_strategy = strategy or self.config.strategy
        
        # Perform reranking
        if selected_strategy == RerankingStrategy.MULTI_CRITERIA:
            result = await self.multi_criteria_reranker.rerank(query, documents)
        else:
            # For single strategies, use the multi-criteria with adjusted weights
            temp_config = RerankingConfig(
                strategy=selected_strategy,
                cross_encoder_model=self.config.cross_encoder_model,
                top_k=self.config.top_k
            )
            
            temp_reranker = MultiCriteriaReranker(temp_config)
            result = await temp_reranker.rerank(query, documents)
            result.strategy_used = selected_strategy
        
        # Cache result
        self.rerank_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        return result
    
    def _generate_cache_key(self, query: str, documents: List[HybridSearchResult], 
                          strategy: RerankingStrategy) -> str:
        """Generate cache key for reranking"""
        import hashlib
        
        # Create a simple representation of documents
        doc_ids = [doc.id for doc in documents]
        doc_str = "_".join(sorted(doc_ids))
        
        cache_str = f"{query}_{doc_str}_{strategy.value}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear reranking cache"""
        self.rerank_cache.clear()
        print("Reranking cache cleared")


# Factory classes
class RerankerFactory:
    """Factory for creating rerankers"""
    
    @staticmethod
    def create_default_reranker() -> AdvancedReranker:
        """Create reranker with default configuration"""
        return AdvancedReranker()
    
    @staticmethod
    def create_trading_reranker() -> AdvancedReranker:
        """Create reranker optimized for trading domain"""
        config = RerankingConfig(
            strategy=RerankingStrategy.MULTI_CRITERIA,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            diversity_weight=0.15,  # Lower diversity for trading (consistency important)
            relevance_weight=0.7,   # Higher relevance weight
            freshness_weight=0.1,   # Fresh trading information important
            authority_weight=0.05,  # Lower authority weight
            top_k=15
        )
        return AdvancedReranker(config)
    
    @staticmethod
    def create_precision_reranker() -> AdvancedReranker:
        """Create reranker optimized for precision"""
        config = RerankingConfig(
            strategy=RerankingStrategy.MULTI_CRITERIA,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            relevance_weight=0.8,
            diversity_weight=0.1,
            authority_weight=0.1,
            top_k=10
        )
        return AdvancedReranker(config)
    
    @staticmethod
    def create_diversity_reranker() -> AdvancedReranker:
        """Create reranker optimized for diversity"""
        config = RerankingConfig(
            strategy=RerankingStrategy.DIVERSITY_AWARE,
            diversity_weight=0.4,
            relevance_weight=0.4,
            top_k=20
        )
        return AdvancedReranker(config)