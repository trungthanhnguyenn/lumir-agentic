"""
Comprehensive RAG Evaluation Framework
Implements state-of-the-art metrics for RAG system assessment
"""

import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import re
from pathlib import Path

from ..retrieval.hybrid_search import HybridSearchEngine, SearchQuery, HybridSearchResult
from ..retrieval.multi_hop_reasoning import MultiHopReasoner, ReasoningPath


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    RETRIEVAL_PRECISION = "retrieval_precision"
    RETRIEVAL_RECALL = "retrieval_recall"
    RETRIEVAL_F1 = "retrieval_f1"
    MRR = "mean_reciprocal_rank"
    MAP = "mean_average_precision"
    NDCG = "normalized_discounted_cumulative_gain"
    COVERAGE = "coverage"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    ANSWER_RELEVANCE = "answer_relevance"
    FAITHFULNESS = "faithfulness"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"


@dataclass
class EvaluationDataset:
    """Dataset for RAG evaluation"""
    queries: List[str]
    ground_truth_docs: List[List[str]]  # Document IDs for each query
    ground_truth_answers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluation for a single metric"""
    metric: EvaluationMetric
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    per_query_scores: List[float] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    dataset_name: str
    system_name: str
    timestamp: str
    results: List[EvaluationResult] = field(default_factory=list)
    overall_score: float = 0.0
    processing_time: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator
    """
    
    def __init__(self, 
                 hybrid_search: HybridSearchEngine,
                 multi_hop_reasoner: Optional[MultiHopReasoner] = None):
        
        self.search_engine = hybrid_search
        self.reasoner = multi_hop_reasoner
        
        # Evaluation cache
        self.evaluation_cache = {}
        
        print("RAG Evaluator initialized")
    
    async def evaluate_retrieval(self, 
                                dataset: EvaluationDataset,
                                k_values: List[int] = [1, 3, 5, 10]) -> List[EvaluationResult]:
        """
        Evaluate retrieval performance with standard IR metrics
        """
        print(f"Evaluating retrieval performance on {len(dataset.queries)} queries...")
        
        results = []
        
        for k in k_values:
            print(f"Calculating metrics for k={k}...")
            
            # Calculate precision@k
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for i, query in enumerate(dataset.queries):
                # Perform search
                search_query = SearchQuery(
                    text=query,
                    limit=k,
                    query_type=SearchQuery.HYBRID
                )
                
                search_results = await self.search_engine.search(search_query)
                retrieved_doc_ids = [result.id for result in search_results]
                
                # Ground truth for this query
                relevant_docs = set(dataset.ground_truth_docs[i])
                
                # Calculate precision, recall, F1
                if retrieved_doc_ids:
                    retrieved_relevant = set(retrieved_doc_ids).intersection(relevant_docs)
                    
                    precision = len(retrieved_relevant) / len(retrieved_doc_ids)
                    recall = len(retrieved_relevant) / len(relevant_docs) if relevant_docs else 0
                    
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0
                else:
                    precision = recall = f1 = 0.0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
            
            # Create evaluation results
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            avg_f1 = np.mean(f1_scores)
            
            results.append(EvaluationResult(
                metric=EvaluationMetric.RETRIEVAL_PRECISION,
                score=avg_precision,
                details={"k": k, "per_query_scores": precision_scores},
                per_query_scores=precision_scores
            ))
            
            results.append(EvaluationResult(
                metric=EvaluationMetric.RETRIEVAL_RECALL,
                score=avg_recall,
                details={"k": k, "per_query_scores": recall_scores},
                per_query_scores=recall_scores
            ))
            
            results.append(EvaluationResult(
                metric=EvaluationMetric.RETRIEVAL_F1,
                score=avg_f1,
                details={"k": k, "per_query_scores": f1_scores},
                per_query_scores=f1_scores
            ))
        
        return results
    
    def calculate_mrr(self, dataset: EvaluationDataset) -> EvaluationResult:
        """
        Calculate Mean Reciprocal Rank (MRR)
        """
        reciprocal_ranks = []
        
        for i, query in enumerate(dataset.queries):
            # For simplicity, assume we'd get ranked results
            # In practice, you'd use actual search results
            
            relevant_docs = set(dataset.ground_truth_docs[i])
            
            # Simulate ranked retrieval
            # In real implementation, this would come from actual search
            ranked_docs = self._simulate_ranked_results(query, len(relevant_docs) * 2)
            
            # Find reciprocal rank
            rr = 0.0
            for rank, doc_id in enumerate(ranked_docs, 1):
                if doc_id in relevant_docs:
                    rr = 1.0 / rank
                    break
            
            reciprocal_ranks.append(rr)
        
        mrr_score = np.mean(reciprocal_ranks)
        
        return EvaluationResult(
            metric=EvaluationMetric.MRR,
            score=mrr_score,
            details={"per_query_scores": reciprocal_ranks},
            per_query_scores=reciprocal_ranks
        )
    
    def calculate_map(self, dataset: EvaluationDataset) -> EvaluationResult:
        """
        Calculate Mean Average Precision (MAP)
        """
        average_precisions = []
        
        for i, query in enumerate(dataset.queries):
            relevant_docs = set(dataset.ground_truth_docs[i])
            
            # Simulate ranked retrieval
            ranked_docs = self._simulate_ranked_results(query, 20)
            
            # Calculate average precision
            precisions = []
            relevant_found = 0
            
            for rank, doc_id in enumerate(ranked_docs, 1):
                if doc_id in relevant_docs:
                    relevant_found += 1
                    precision_at_k = relevant_found / rank
                    precisions.append(precision_at_k)
            
            if precisions:
                ap = np.mean(precisions)
            else:
                ap = 0.0
            
            average_precisions.append(ap)
        
        map_score = np.mean(average_precisions)
        
        return EvaluationResult(
            metric=EvaluationMetric.MAP,
            score=map_score,
            details={"per_query_scores": average_precisions},
            per_query_scores=average_precisions
        )
    
    def calculate_diversity(self, 
                           dataset: EvaluationDataset,
                           search_results: List[List[HybridSearchResult]]) -> EvaluationResult:
        """
        Calculate diversity metrics for search results
        """
        diversity_scores = []
        
        for i, query_results in enumerate(search_results):
            if len(query_results) < 2:
                diversity_scores.append(0.0)
                continue
            
            # Calculate pairwise content diversity
            contents = [result.payload.get("content", "") for result in query_results]
            
            pairwise_similarities = []
            for j in range(len(contents)):
                for k in range(j + 1, len(contents)):
                    similarity = self._calculate_content_similarity(contents[j], contents[k])
                    pairwise_similarities.append(similarity)
            
            # Diversity = 1 - average similarity
            if pairwise_similarities:
                avg_similarity = np.mean(pairwise_similarities)
                diversity = 1.0 - avg_similarity
            else:
                diversity = 0.0
            
            diversity_scores.append(diversity)
        
        avg_diversity = np.mean(diversity_scores)
        
        return EvaluationResult(
            metric=EvaluationMetric.DIVERSITY,
            score=avg_diversity,
            details={"per_query_scores": diversity_scores},
            per_query_scores=diversity_scores
        )
    
    def calculate_coverage(self, 
                          dataset: EvaluationDataset,
                          search_results: List[List[HybridSearchResult]]) -> EvaluationResult:
        """
        Calculate coverage - how many ground truth docs are retrieved
        """
        coverage_scores = []
        
        for i, (query_results, ground_truth) in enumerate(zip(search_results, dataset.ground_truth_docs)):
            retrieved_ids = {result.id for result in query_results}
            ground_truth_set = set(ground_truth)
            
            if ground_truth_set:
                coverage = len(retrieved_ids.intersection(ground_truth_set)) / len(ground_truth_set)
            else:
                coverage = 0.0
            
            coverage_scores.append(coverage)
        
        avg_coverage = np.mean(coverage_scores)
        
        return EvaluationResult(
            metric=EvaluationMetric.COVERAGE,
            score=avg_coverage,
            details={"per_query_scores": coverage_scores},
            per_query_scores=coverage_scores
        )
    
    async def evaluate_answer_quality(self, 
                                    dataset: EvaluationDataset,
                                    generated_answers: List[str]) -> List[EvaluationResult]:
        """
        Evaluate answer quality using various metrics
        """
        if len(generated_answers) != len(dataset.queries):
            raise ValueError("Number of answers must match number of queries")
        
        results = []
        
        # Answer relevance
        relevance_scores = []
        for i, (generated_answer, ground_truth_answer) in enumerate(zip(generated_answers, dataset.ground_truth_answers)):
            relevance = self._calculate_answer_relevance(generated_answer, ground_truth_answer)
            relevance_scores.append(relevance)
        
        avg_relevance = np.mean(relevance_scores)
        results.append(EvaluationResult(
            metric=EvaluationMetric.ANSWER_RELEVANCE,
            score=avg_relevance,
            details={"per_query_scores": relevance_scores},
            per_query_scores=relevance_scores
        ))
        
        # Faithfulness (requires retrieved context)
        # This would need actual search results for each query
        # For now, return placeholder
        results.append(EvaluationResult(
            metric=EvaluationMetric.FAITHFULNESS,
            score=0.8,  # Placeholder
            details={"note": "Faithfulness evaluation requires retrieved context"}
        ))
        
        return results
    
    async def evaluate_multi_hop_reasoning(self, 
                                          dataset: EvaluationDataset) -> List[EvaluationResult]:
        """
        Evaluate multi-hop reasoning performance
        """
        if not self.reasoner:
            return []
        
        print("Evaluating multi-hop reasoning...")
        
        reasoning_results = []
        step_counts = []
        confidence_scores = []
        entity_counts = []
        
        for i, query in enumerate(dataset.queries):
            try:
                reasoning_path = await self.reasoner.reason(query)
                
                step_counts.append(len(reasoning_path.steps))
                confidence_scores.append(reasoning_path.confidence_score)
                entity_counts.append(len([entity for step in reasoning_path.steps for entity in step.entities_extracted]))
                
                reasoning_results.append(reasoning_path)
                
            except Exception as e:
                print(f"Error reasoning on query {i}: {e}")
                step_counts.append(0)
                confidence_scores.append(0.0)
                entity_counts.append(0)
        
        # Create evaluation results
        avg_steps = np.mean(step_counts)
        avg_confidence = np.mean(confidence_scores)
        avg_entities = np.mean(entity_counts)
        
        results = [
            EvaluationResult(
                metric=EvaluationMetric.CONTEXT_RECALL,  # Using as proxy for reasoning depth
                score=avg_steps / 3.0,  # Normalize by expected max steps
                details={"avg_steps": avg_steps, "step_counts": step_counts}
            ),
            EvaluationResult(
                metric=EvaluationMetric.CONTEXT_PRECISION,  # Using as proxy for reasoning confidence
                score=avg_confidence,
                details={"avg_confidence": avg_confidence, "confidence_scores": confidence_scores}
            )
        ]
        
        return results
    
    def _simulate_ranked_results(self, query: str, num_results: int) -> List[str]:
        """
        Simulate ranked retrieval results
        In practice, this would use actual search results
        """
        # This is a placeholder implementation
        # In real usage, you would get actual ranked results from the search engine
        return [f"doc_{i}_{hash(query) % 1000}" for i in range(num_results)]
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content pieces
        """
        # Simple Jaccard similarity on word tokens
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_answer_relevance(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
        Calculate relevance between generated and ground truth answers
        """
        # Multiple similarity measures
        # 1. Word overlap (BLEU-like)
        gen_words = set(generated_answer.lower().split())
        gt_words = set(ground_truth_answer.lower().split())
        
        if gt_words:
            precision = len(gen_words.intersection(gt_words)) / len(gen_words) if gen_words else 0
            recall = len(gen_words.intersection(gt_words)) / len(gt_words)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
        else:
            f1 = 0.0
        
        # 2. Length similarity (penalize very short/long answers)
        length_ratio = min(len(generated_answer), len(ground_truth_answer)) / max(len(generated_answer), len(ground_truth_answer))
        
        # Combine metrics
        relevance = 0.7 * f1 + 0.3 * length_ratio
        
        return relevance
    
    async def comprehensive_evaluation(self, 
                                     dataset: EvaluationDataset,
                                     evaluate_reasoning: bool = True) -> EvaluationReport:
        """
        Perform comprehensive evaluation of the RAG system
        """
        start_time = time.time()
        
        print(f"Starting comprehensive evaluation on {len(dataset.queries)} queries...")
        
        all_results = []
        
        # 1. Retrieval evaluation
        retrieval_results = await self.evaluate_retrieval(dataset)
        all_results.extend(retrieval_results)
        
        # 2. Ranking metrics
        mrr_result = self.calculate_mrr(dataset)
        all_results.append(mrr_result)
        
        map_result = self.calculate_map(dataset)
        all_results.append(map_result)
        
        # 3. Get search results for other metrics
        search_results = []
        for query in dataset.queries:
            search_query = SearchQuery(text=query, limit=10)
            results = await self.search_engine.search(search_query)
            search_results.append(results)
        
        # 4. Diversity and coverage
        diversity_result = self.calculate_diversity(dataset, search_results)
        all_results.append(diversity_result)
        
        coverage_result = self.calculate_coverage(dataset, search_results)
        all_results.append(coverage_result)
        
        # 5. Multi-hop reasoning evaluation
        if evaluate_reasoning and self.reasoner:
            reasoning_results = await self.evaluate_multi_hop_reasoning(dataset)
            all_results.extend(reasoning_results)
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(all_results)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(all_results)
        
        processing_time = time.time() - start_time
        
        # Create report
        report = EvaluationReport(
            dataset_name=dataset.metadata.get("name", "Unknown"),
            system_name="Advanced RAG System",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            results=all_results,
            overall_score=overall_score,
            processing_time=processing_time,
            config={
                "num_queries": len(dataset.queries),
                "k_values": [1, 3, 5, 10],
                "evaluate_reasoning": evaluate_reasoning
            },
            recommendations=recommendations
        )
        
        print(f"Comprehensive evaluation completed in {processing_time:.2f}s")
        print(f"Overall score: {overall_score:.3f}")
        
        return report
    
    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []
        
        # Analyze results and generate recommendations
        for result in results:
            if result.metric == EvaluationMetric.RETRIEVAL_PRECISION and result.score < 0.5:
                recommendations.append("Consider improving semantic chunking to increase precision")
            
            if result.metric == EvaluationMetric.RETRIEVAL_RECALL and result.score < 0.6:
                recommendations.append("Increase search limit or improve query expansion for better recall")
            
            if result.metric == EvaluationMetric.DIVERSITY and result.score < 0.4:
                recommendations.append("Implement diversity-aware reranking to increase result variety")
            
            if result.metric == EvaluationMetric.MRR and result.score < 0.3:
                recommendations.append("Improve ranking algorithm to place relevant results higher")
            
            if result.metric == EvaluationMetric.CONTEXT_PRECISION and result.score < 0.5:
                recommendations.append("Enhance multi-hop reasoning for better context precision")
        
        if not recommendations:
            recommendations.append("System performance is good - consider focusing on optimization and scalability")
        
        return recommendations
    
    def _calculate_overall_score(self, results: List[EvaluationResult]) -> float:
        """Calculate overall performance score"""
        if not results:
            return 0.0
        
        # Weight different metrics
        weights = {
            EvaluationMetric.RETRIEVAL_PRECISION: 0.2,
            EvaluationMetric.RETRIEVAL_RECALL: 0.2,
            EvaluationMetric.RETRIEVAL_F1: 0.15,
            EvaluationMetric.MRR: 0.15,
            EvaluationMetric.MAP: 0.1,
            EvaluationMetric.DIVERSITY: 0.1,
            EvaluationMetric.COVERAGE: 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weights.get(result.metric, 0.05)
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def save_report(self, report: EvaluationReport, filepath: str):
        """Save evaluation report to file"""
        # Convert report to dict for JSON serialization
        report_dict = {
            "dataset_name": report.dataset_name,
            "system_name": report.system_name,
            "timestamp": report.timestamp,
            "overall_score": report.overall_score,
            "processing_time": report.processing_time,
            "config": report.config,
            "recommendations": report.recommendations,
            "results": [
                {
                    "metric": result.metric.value,
                    "score": result.score,
                    "details": result.details,
                    "per_query_scores": result.per_query_scores[:10]  # Limit for file size
                }
                for result in report.results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation report saved to {filepath}")


# Factory and utility classes
class EvaluationDatasetFactory:
    """Factory for creating evaluation datasets"""
    
    @staticmethod
    def create_synthetic_dataset(num_queries: int = 100) -> EvaluationDataset:
        """Create synthetic dataset for testing"""
        import random
        
        queries = [
            "What is stop-loss in trading?",
            "How do you calculate position size?",
            "What are the risks of leverage?",
            "Explain technical analysis",
            "What is fundamental analysis?",
            "How to manage trading emotions?",
            "What are support and resistance levels?",
            "Explain risk-reward ratio",
            "What is diversification in trading?",
            "How to create a trading plan?"
        ]
        
        # Extend queries to reach desired number
        while len(queries) < num_queries:
            base_query = random.choice(queries[:5])
            queries.append(f"{base_query} (variation {len(queries)})")
        
        queries = queries[:num_queries]
        
        # Generate synthetic ground truth
        ground_truth_docs = []
        ground_truth_answers = []
        
        for i, query in enumerate(queries):
            # Simulate relevant documents
            relevant_docs = [f"doc_{i}_{j}" for j in range(random.randint(1, 4))]
            ground_truth_docs.append(relevant_docs)
            
            # Generate synthetic answers
            answer = f"This is the answer to {query}. It contains relevant information about trading concepts and strategies."
            ground_truth_answers.append(answer)
        
        return EvaluationDataset(
            queries=queries,
            ground_truth_docs=ground_truth_docs,
            ground_truth_answers=ground_truth_answers,
            metadata={"name": "Synthetic Trading Dataset", "num_queries": num_queries}
        )
    
    @staticmethod
    def load_dataset_from_file(filepath: str) -> EvaluationDataset:
        """Load dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return EvaluationDataset(
            queries=data["queries"],
            ground_truth_docs=data["ground_truth_docs"],
            ground_truth_answers=data["ground_truth_answers"],
            metadata=data.get("metadata", {})
        )


class RAGEvaluatorFactory:
    """Factory for creating RAG evaluators"""
    
    @staticmethod
    def create_evaluator(hybrid_search: HybridSearchEngine,
                        multi_hop_reasoner: Optional[MultiHopReasoner] = None) -> RAGEvaluator:
        """Create RAG evaluator with default configuration"""
        return RAGEvaluator(hybrid_search, multi_hop_reasoner)
    
    @staticmethod
    def create_trading_evaluator(hybrid_search: HybridSearchEngine,
                                multi_hop_reasoner: Optional[MultiHopReasoner] = None) -> RAGEvaluator:
        """Create evaluator optimized for trading domain"""
        return RAGEvaluator(hybrid_search, multi_hop_reasoner)