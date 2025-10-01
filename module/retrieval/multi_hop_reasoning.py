"""
Multi-hop Reasoning and Contextual Retrieval System
Implements graph-based reasoning for complex query answering
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np
from collections import defaultdict, deque
import re

from .hybrid_search import HybridSearchEngine, SearchQuery, SearchStrategy, HybridSearchResult


class ReasoningType(Enum):
    """Types of multi-hop reasoning"""
    ENTITY_RELATION = "entity_relation"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    AGGREGATIVE = "aggregative"
    DECOMPOSITION = "decomposition"


@dataclass
class ReasoningStep:
    """Single step in reasoning process"""
    step_id: str
    query: str
    results: List[HybridSearchResult]
    reasoning_type: ReasoningType
    confidence: float
    entities_extracted: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    next_queries: List[str] = field(default_factory=list)


@dataclass
class ReasoningPath:
    """Complete reasoning path through multiple steps"""
    path_id: str
    initial_query: str
    steps: List[ReasoningStep]
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    supporting_evidence: List[HybridSearchResult] = field(default_factory=list)
    reasoning_graph: Optional[nx.DiGraph] = None


@dataclass
class ReasoningConfig:
    """Configuration for multi-hop reasoning"""
    max_hops: int = 3
    confidence_threshold: float = 0.6
    entity_extraction_enabled: bool = True
    graph_pruning_enabled: bool = True
    parallel_search_enabled: bool = True
    context_window_size: int = 4000
    max_entities_per_step: int = 5


class EntityExtractor:
    """Extract entities and relationships from text"""
    
    def __init__(self):
        # Simple entity patterns - in production, use NER models
        self.entity_patterns = {
            "person": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "organization": r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Company|Trading)\b',
            "trading_term": r'\b(?:stop-loss|take-profit|leverage|margin|pips|spread)\b',
            "numeric": r'\b\d+(?:\.\d+)?(?:%|USD|EUR|JPY)?\b',
            "date": r'\b(?:\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2})\b'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using pattern matching"""
        entities = defaultdict(list)
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return dict(entities)
    
    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Simple relationship extraction based on patterns
        relationship_patterns = [
            (r'(\w+) is (\w+)', "is_a"),
            (r'(\w+) belongs to (\w+)', "belongs_to"),
            (r'(\w+) affects (\w+)', "affects"),
            (r'(\w+) includes (\w+)', "includes")
        ]
        
        for pattern, rel_type in relationship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    relationships.append({
                        "type": rel_type,
                        "subject": match[0],
                        "object": match[1],
                        "confidence": 0.7  # Simple confidence score
                    })
        
        return relationships


class MultiHopReasoner:
    """
    Advanced multi-hop reasoning system for complex query answering
    """
    
    def __init__(self, 
                 hybrid_search: HybridSearchEngine,
                 config: Optional[ReasoningConfig] = None):
        
        self.search_engine = hybrid_search
        self.config = config or ReasoningConfig()
        self.entity_extractor = EntityExtractor()
        
        # Reasoning cache
        self.reasoning_cache = {}
        self.entity_cache = {}
        
        print("Multi-hop Reasoner initialized")
    
    async def reason(self, query: str, reasoning_type: Optional[ReasoningType] = None) -> ReasoningPath:
        """
        Perform multi-hop reasoning on a complex query
        """
        start_time = time.time()
        
        # Generate reasoning path ID
        path_id = f"path_{int(time.time())}"
        
        # Analyze query to determine reasoning approach
        query_analysis = self._analyze_query(query)
        if reasoning_type:
            query_analysis["reasoning_type"] = reasoning_type
        
        # Initialize reasoning path
        reasoning_path = ReasoningPath(
            path_id=path_id,
            initial_query=query,
            steps=[],
            reasoning_graph=nx.DiGraph()
        )
        
        # Step 1: Initial search
        initial_step = await self._perform_reasoning_step(
            query, step_num=1, reasoning_type=query_analysis["reasoning_type"]
        )
        reasoning_path.steps.append(initial_step)
        
        # Add nodes and edges to reasoning graph
        self._update_reasoning_graph(reasoning_path.reasoning_graph, initial_step)
        
        # Generate subsequent queries based on initial results
        next_queries = self._generate_next_queries(initial_step, query_analysis)
        
        # Step 2+: Multi-hop reasoning
        current_hop = 2
        while next_queries and current_hop <= self.config.max_hops:
            # Select best next query
            next_query = self._select_best_next_query(next_queries, reasoning_path)
            
            # Perform reasoning step
            step = await self._perform_reasoning_step(
                next_query, step_num=current_hop, reasoning_type=query_analysis["reasoning_type"]
            )
            
            reasoning_path.steps.append(step)
            self._update_reasoning_graph(reasoning_path.reasoning_graph, step)
            
            # Generate new queries
            new_queries = self._generate_next_queries(step, query_analysis)
            
            # Filter and prioritize queries
            next_queries = self._filter_queries(new_queries, reasoning_path)
            
            current_hop += 1
        
        # Synthesize final answer
        final_answer = self._synthesize_answer(reasoning_path)
        reasoning_path.final_answer = final_answer
        
        # Calculate confidence score
        reasoning_path.confidence_score = self._calculate_path_confidence(reasoning_path)
        
        # Collect supporting evidence
        reasoning_path.supporting_evidence = self._collect_supporting_evidence(reasoning_path)
        
        reasoning_time = time.time() - start_time
        print(f"Multi-hop reasoning completed: {len(reasoning_path.steps)} steps in {reasoning_time:.2f}s")
        
        return reasoning_path
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine reasoning approach"""
        query_lower = query.lower()
        
        analysis = {
            "entities": self.entity_extractor.extract_entities(query),
            "question_words": [],
            "comparison_words": [],
            "temporal_words": [],
            "reasoning_type": ReasoningType.ENTITY_RELATION
        }
        
        # Identify question patterns
        question_patterns = {
            "why": ["why", "what causes", "what leads to"],
            "how": ["how", "how does", "how can"],
            "what": ["what", "what is", "what are"],
            "compare": ["compare", "difference", "vs", "versus"],
            "temporal": ["when", "before", "after", "during", "timeline"],
            "aggregation": ["total", "sum", "average", "count", "list all"]
        }
        
        for q_type, patterns in question_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis["question_words"].append(q_type)
        
        # Determine reasoning type
        if any(word in query_lower for word in ["compare", "difference", "vs"]):
            analysis["reasoning_type"] = ReasoningType.COMPARATIVE
        elif any(word in query_lower for word in ["why", "cause", "lead to", "affect"]):
            analysis["reasoning_type"] = ReasoningType.CAUSAL
        elif any(word in query_lower for word in ["when", "before", "after", "timeline"]):
            analysis["reasoning_type"] = ReasoningType.TEMPORAL
        elif any(word in query_lower for word in ["total", "sum", "average", "count"]):
            analysis["reasoning_type"] = ReasoningType.AGGREGATIVE
        elif "?" in query and len(query.split()) > 10:
            analysis["reasoning_type"] = ReasoningType.DECOMPOSITION
        
        return analysis
    
    async def _perform_reasoning_step(self, 
                                     query: str, 
                                     step_num: int,
                                     reasoning_type: ReasoningType) -> ReasoningStep:
        """Perform a single reasoning step"""
        step_id = f"step_{step_num}_{int(time.time())}"
        
        # Create search query
        search_query = SearchQuery(
            text=query,
            query_type=SearchStrategy.HYBRID,
            limit=10,
            rerank=True,
            expand_query=True
        )
        
        # Perform search
        results = await self.search_engine.search(search_query)
        
        # Extract entities and relationships from results
        entities = []
        relationships = []
        
        if self.config.entity_extraction_enabled:
            all_text = " ".join([r.payload.get("content", "") for r in results])
            entities_dict = self.entity_extractor.extract_entities(all_text)
            entities = list(set([entity for entity_list in entities_dict.values() for entity in entity_list]))
            relationships = self.entity_extractor.extract_relationships(all_text, entities_dict)
        
        # Calculate confidence based on result quality
        confidence = self._calculate_step_confidence(results, query)
        
        # Generate next queries based on reasoning type
        next_queries = self._generate_step_queries(query, results, reasoning_type, entities)
        
        return ReasoningStep(
            step_id=step_id,
            query=query,
            results=results,
            reasoning_type=reasoning_type,
            confidence=confidence,
            entities_extracted=entities[:self.config.max_entities_per_step],
            relationships=relationships,
            next_queries=next_queries
        )
    
    def _generate_next_queries(self, step: ReasoningStep, 
                             query_analysis: Dict[str, Any]) -> List[str]:
        """Generate next queries based on current step results"""
        next_queries = []
        
        # Entity-based expansion
        if step.entities_extracted:
            for entity in step.entities_extracted[:3]:  # Limit to top 3 entities
                # Create entity-focused queries
                entity_query = f"What is {entity} in context of {step.query}"
                next_queries.append(entity_query)
                
                # Relationship queries
                for rel in step.relationships[:2]:
                    if rel.get("subject") == entity or rel.get("object") == entity:
                        rel_query = f"How does {rel.get('subject', '')} {rel.get('type', '')} {rel.get('object', '')}"
                        next_queries.append(rel_query)
        
        # Gap-based queries (what information is missing?)
        if step.results:
            all_content = " ".join([r.payload.get("content", "") for r in step.results])
            missing_info = self._identify_missing_information(step.query, all_content)
            for missing in missing_info[:2]:
                gap_query = f"Tell me more about {missing} related to {step.query}"
                next_queries.append(gap_query)
        
        # Reasoning type specific queries
        if step.reasoning_type == ReasoningType.CAUSAL:
            causal_queries = self._generate_causal_queries(step)
            next_queries.extend(causal_queries)
        elif step.reasoning_type == ReasoningType.COMPARATIVE:
            comp_queries = self._generate_comparative_queries(step)
            next_queries.extend(comp_queries)
        elif step.reasoning_type == ReasoningType.TEMPORAL:
            temporal_queries = self._generate_temporal_queries(step)
            next_queries.extend(temporal_queries)
        
        return next_queries[:5]  # Limit to top 5 queries
    
    def _generate_step_queries(self, 
                              original_query: str,
                              results: List[HybridSearchResult],
                              reasoning_type: ReasoningType,
                              entities: List[str]) -> List[str]:
        """Generate queries specific to reasoning type and current step"""
        queries = []
        
        if reasoning_type == ReasoningType.CAUSAL:
            # Look for causes and effects
            for entity in entities[:2]:
                queries.append(f"What causes {entity}?")
                queries.append(f"What are the effects of {entity}?")
        
        elif reasoning_type == ReasoningType.COMPARATIVE:
            # Look for comparisons
            if len(entities) >= 2:
                queries.append(f"Compare {entities[0]} and {entities[1]}")
        
        elif reasoning_type == ReasoningType.TEMPORAL:
            # Look for temporal relationships
            queries.append(f"What happens before and after {original_query}?")
        
        elif reasoning_type == ReasoningType.AGGREGATIVE:
            # Look for additional instances
            queries.append(f"What are other examples of {original_query}?")
        
        return queries
    
    def _generate_causal_queries(self, step: ReasoningStep) -> List[str]:
        """Generate causal reasoning queries"""
        queries = []
        
        for rel in step.relationships:
            if rel.get("type") == "affects":
                subject = rel.get("subject", "")
                obj = rel.get("object", "")
                if subject and obj:
                    queries.append(f"How does {subject} affect {obj}?")
                    queries.append(f"What are the consequences of {subject} on {obj}?")
        
        return queries
    
    def _generate_comparative_queries(self, step: ReasoningStep) -> List[str]:
        """Generate comparative reasoning queries"""
        queries = []
        
        if len(step.entities_extracted) >= 2:
            entity1, entity2 = step.entities_extracted[0], step.entities_extracted[1]
            queries.extend([
                f"What are the differences between {entity1} and {entity2}?",
                f"What are the similarities between {entity1} and {entity2}?",
                f"Which is better: {entity1} or {entity2}?"
            ])
        
        return queries
    
    def _generate_temporal_queries(self, step: ReasoningStep) -> List[str]:
        """Generate temporal reasoning queries"""
        queries = []
        
        for entity in step.entities_extracted[:2]:
            queries.extend([
                f"When did {entity} happen?",
                f"What happened before {entity}?",
                f"What happened after {entity}?",
                f"What is the timeline of {entity}?"
            ])
        
        return queries
    
    def _select_best_next_query(self, 
                               queries: List[str], 
                               reasoning_path: ReasoningPath) -> str:
        """Select the best next query to pursue"""
        if not queries:
            return ""
        
        # Score queries based on multiple factors
        query_scores = []
        
        for query in queries:
            score = 0.0
            
            # Prefer queries with entities we haven't explored
            query_entities = self.entity_extractor.extract_entities(query)
            explored_entities = set()
            for step in reasoning_path.steps:
                explored_entities.update(step.entities_extracted)
            
            new_entities = set([e for entities in query_entities.values() for e in entities]) - explored_entities
            score += len(new_entities) * 0.3
            
            # Prefer queries that are more specific
            if len(query.split()) > 5:
                score += 0.2
            
            # Avoid queries that are too similar to previous ones
            for step in reasoning_path.steps:
                similarity = self._calculate_query_similarity(query, step.query)
                score -= similarity * 0.4
            
            query_scores.append((query, score))
        
        # Return highest scoring query
        if query_scores:
            query_scores.sort(key=lambda x: x[1], reverse=True)
            return query_scores[0][0]
        
        return queries[0] if queries else ""
    
    def _filter_queries(self, 
                       queries: List[str], 
                       reasoning_path: ReasoningPath) -> List[str]:
        """Filter out redundant and low-quality queries"""
        filtered_queries = []
        
        for query in queries:
            # Skip if too similar to previous queries
            is_similar = False
            for step in reasoning_path.steps:
                similarity = self._calculate_query_similarity(query, step.query)
                if similarity > 0.7:
                    is_similar = True
                    break
            
            # Skip if query is too short or too generic
            if len(query.split()) < 3:
                is_similar = True
            
            if not is_similar:
                filtered_queries.append(query)
        
        return filtered_queries[:3]  # Limit to top 3
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_step_confidence(self, 
                                  results: List[HybridSearchResult], 
                                  query: str) -> float:
        """Calculate confidence score for a reasoning step"""
        if not results:
            return 0.0
        
        # Base confidence from result scores
        avg_score = np.mean([r.combined_score for r in results])
        
        # Boost based on number of relevant results
        result_count_factor = min(1.0, len(results) / 5.0)
        
        # Boost based on content relevance
        relevance_factor = self._calculate_content_relevance(results, query)
        
        confidence = (avg_score * 0.5 + result_count_factor * 0.3 + relevance_factor * 0.2)
        
        return min(1.0, confidence)
    
    def _calculate_content_relevance(self, 
                                   results: List[HybridSearchResult], 
                                   query: str) -> float:
        """Calculate how relevant the content is to the query"""
        query_terms = set(query.lower().split())
        
        if not query_terms:
            return 0.0
        
        total_relevance = 0.0
        for result in results:
            content = result.payload.get("content", "").lower()
            content_terms = set(content.split())
            
            overlap = len(query_terms.intersection(content_terms))
            relevance = overlap / len(query_terms)
            total_relevance += relevance
        
        return total_relevance / len(results) if results else 0.0
    
    def _update_reasoning_graph(self, 
                               graph: nx.DiGraph, 
                               step: ReasoningStep):
        """Update reasoning graph with new step"""
        # Add step node
        graph.add_node(
            step.step_id,
            query=step.query,
            confidence=step.confidence,
            entities=step.entities_extracted,
            reasoning_type=step.reasoning_type.value
        )
        
        # Add entity nodes and relationships
        for entity in step.entities_extracted:
            entity_id = f"entity_{entity.replace(' ', '_')}"
            if not graph.has_node(entity_id):
                graph.add_node(entity_id, type="entity", name=entity)
            
            # Connect step to entities
            graph.add_edge(step.step_id, entity_id, relation="mentions")
        
        # Add relationships between entities
        for rel in step.relationships:
            subject_id = f"entity_{rel.get('subject', '').replace(' ', '_')}"
            object_id = f"entity_{rel.get('object', '').replace(' ', '_')}"
            
            if graph.has_node(subject_id) and graph.has_node(object_id):
                graph.add_edge(subject_id, object_id, 
                             relation=rel.get("type", "related"),
                             confidence=rel.get("confidence", 0.5))
    
    def _synthesize_answer(self, reasoning_path: ReasoningPath) -> str:
        """Synthesize final answer from reasoning path"""
        if not reasoning_path.steps:
            return "Unable to generate answer due to lack of reasoning steps."
        
        # Collect all relevant content
        all_content = []
        for step in reasoning_path.steps:
            for result in step.results[:3]:  # Top 3 results per step
                content = result.payload.get("content", "")
                if content:
                    all_content.append(content)
        
        # Simple synthesis - in production, use LLM for better synthesis
        if reasoning_path.reasoning_type == ReasoningType.CAUSAL:
            return self._synthesize_causal_answer(reasoning_path, all_content)
        elif reasoning_path.reasoning_type == ReasoningType.COMPARATIVE:
            return self._synthesize_comparative_answer(reasoning_path, all_content)
        elif reasoning_path.reasoning_type == ReasoningType.TEMPORAL:
            return self._synthesize_temporal_answer(reasoning_path, all_content)
        else:
            return self._synthesize_general_answer(reasoning_path, all_content)
    
    def _synthesize_causal_answer(self, reasoning_path: ReasoningPath, 
                                 content: List[str]) -> str:
        """Synthesize answer for causal reasoning"""
        answer_parts = [f"Based on the analysis of {reasoning_path.initial_query}:"]
        
        # Extract causes and effects
        causes = []
        effects = []
        
        for step in reasoning_path.steps:
            for rel in step.relationships:
                if rel.get("type") == "affects":
                    causes.append(rel.get("subject", ""))
                    effects.append(rel.get("object", ""))
        
        if causes and effects:
            answer_parts.append(f"Key causes identified: {', '.join(set(causes))}")
            answer_parts.append(f"Effects: {', '.join(set(effects))}")
        
        # Add supporting details
        if content:
            answer_parts.append("Additional details from the analysis:")
            answer_parts.append(content[0][:300] + "..." if len(content[0]) > 300 else content[0])
        
        return " ".join(answer_parts)
    
    def _synthesize_comparative_answer(self, reasoning_path: ReasoningPath, 
                                      content: List[str]) -> str:
        """Synthesize answer for comparative reasoning"""
        entities = []
        for step in reasoning_path.steps:
            entities.extend(step.entities_extracted)
        
        unique_entities = list(set(entities))[:2]  # Top 2 entities
        
        answer = f"Comparison between {unique_entities[0] if unique_entities else 'items'}"
        if len(unique_entities) > 1:
            answer += f" and {unique_entities[1]}"
        
        answer += ":\n"
        
        if content:
            answer += f"Based on the analysis: {content[0][:400]}..."
        
        return answer
    
    def _synthesize_temporal_answer(self, reasoning_path: ReasoningPath, 
                                   content: List[str]) -> str:
        """Synthesize answer for temporal reasoning"""
        answer = f"Timeline analysis for {reasoning_path.initial_query}:\n"
        
        # Extract temporal information
        events = []
        for step in reasoning_path.steps:
            for entity in step.entities_extracted:
                if any(char.isdigit() for char in entity):  # Likely contains date/time
                    events.append(entity)
        
        if events:
            answer += f"Key temporal events: {', '.join(events[:5])}\n"
        
        if content:
            answer += f"Context: {content[0][:300]}..."
        
        return answer
    
    def _synthesize_general_answer(self, reasoning_path: ReasoningPath, 
                                 content: List[str]) -> str:
        """Synthesize general answer"""
        answer = f"Analysis of {reasoning_path.initial_query} based on {len(reasoning_path.steps)} reasoning steps:\n\n"
        
        # Summarize key findings
        for i, step in enumerate(reasoning_path.steps, 1):
            if step.results and step.results[0].payload.get("content"):
                key_finding = step.results[0].payload.get("content", "")[:200]
                answer += f"Step {i}: {key_finding}...\n\n"
        
        # Add conclusion
        if reasoning_path.confidence_score > 0.7:
            answer += "Conclusion: High confidence in the analysis results."
        elif reasoning_path.confidence_score > 0.4:
            answer += "Conclusion: Moderate confidence - additional verification may be needed."
        else:
            answer += "Conclusion: Low confidence - more information needed for definitive answer."
        
        return answer
    
    def _calculate_path_confidence(self, reasoning_path: ReasoningPath) -> float:
        """Calculate overall confidence for the reasoning path"""
        if not reasoning_path.steps:
            return 0.0
        
        # Weight steps by recency (later steps more important)
        step_confidences = []
        for i, step in enumerate(reasoning_path.steps):
            weight = (i + 1) / len(reasoning_path.steps)  # Linear weight
            weighted_confidence = step.confidence * weight
            step_confidences.append(weighted_confidence)
        
        # Calculate weighted average
        overall_confidence = np.mean(step_confidences)
        
        # Adjust based on path completeness
        if len(reasoning_path.steps) >= 3:
            overall_confidence *= 1.1  # Boost for complete reasoning
        elif len(reasoning_path.steps) == 1:
            overall_confidence *= 0.8  # Reduce for single-step reasoning
        
        return min(1.0, overall_confidence)
    
    def _collect_supporting_evidence(self, reasoning_path: ReasoningPath) -> List[HybridSearchResult]:
        """Collect the best supporting evidence from all steps"""
        all_results = []
        
        for step in reasoning_path.steps:
            # Top results from each step
            all_results.extend(step.results[:3])
        
        # Remove duplicates and sort by score
        unique_results = []
        seen_ids = set()
        
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        # Sort by combined score
        unique_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return unique_results[:10]  # Top 10 pieces of evidence
    
    def _identify_missing_information(self, query: str, content: str) -> List[str]:
        """Identify what information is missing from the current content"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated gap detection
        
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        missing_terms = query_terms - content_terms
        
        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        missing_terms = [term for term in missing_terms if term not in stop_words and len(term) > 2]
        
        return missing_terms[:3]


# Factory class
class MultiHopReasonerFactory:
    """Factory for creating multi-hop reasoners"""
    
    @staticmethod
    def create_reasoner(hybrid_search: HybridSearchEngine,
                       config: Optional[ReasoningConfig] = None) -> MultiHopReasoner:
        """Create multi-hop reasoner with default configuration"""
        return MultiHopReasoner(hybrid_search, config)
    
    @staticmethod
    def create_trading_reasoner(hybrid_search: HybridSearchEngine) -> MultiHopReasoner:
        """Create reasoner optimized for trading domain"""
        config = ReasoningConfig(
            max_hops=4,
            confidence_threshold=0.7,
            entity_extraction_enabled=True,
            context_window_size=5000,
            max_entities_per_step=8
        )
        return MultiHopReasoner(hybrid_search, config)