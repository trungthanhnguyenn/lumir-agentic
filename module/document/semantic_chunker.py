"""
Advanced Semantic Chunker with Sentence Transformers
Implements state-of-the-art semantic chunking for better context preservation
"""

import os
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .document_processor import DocumentChunk, DocumentInfo


@dataclass
class SemanticChunk:
    """Enhanced chunk with semantic information"""
    content: str
    semantic_score: float
    cluster_id: int
    boundaries: List[str]  # Boundary types that created this chunk
    topic_coherence: float
    metadata: Dict[str, Any]


@dataclass
class ChunkingConfig:
    """Configuration for advanced chunking"""
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 200
    semantic_threshold: float = 0.3
    clustering_threshold: float = 0.5
    preserve_structure: bool = True
    enable_multi_level: bool = True


class AdvancedSemanticChunker:
    """
    Advanced Semantic Chunker using Sentence Transformers
    Implements multiple chunking strategies with semantic awareness
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 config: Optional[ChunkingConfig] = None):
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence transformers not available. Install: pip install sentence-transformers scikit-learn")
        
        self.model_name = model_name
        self.config = config or ChunkingConfig()
        self.model = None
        self.embedding_cache = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded semantic chunking model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            # Try better alternatives for reasoning tasks
            alternative_models = [
                "all-mpnet-base-v2",  # Better semantic understanding
                "multi-qa-mpnet-base-dot-v1",  # Optimized for Q&A
                "all-MiniLM-L6-v2"  # Last resort
            ]
            
            for model_name in alternative_models:
                try:
                    self.model = SentenceTransformer(model_name)
                    print(f"Successfully loaded alternative model: {model_name}")
                    self.model_name = model_name
                    return
                except Exception:
                    continue
            
            raise RuntimeError(f"Failed to load any sentence transformer model. Tried: {[self.model_name] + alternative_models}")
    
    def create_semantic_chunks(self, 
                              content: str, 
                              file_path: Path,
                              document_structure: Optional[Dict[str, Any]] = None) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Create semantically meaningful chunks using multiple strategies
        
        Args:
            content: Document content
            file_path: Source file path
            document_structure: Pre-analyzed structure
            
        Returns:
            Tuple of (chunks, chunking_metadata)
        """
        
        start_time = time.time()
        
        # Preprocess content
        processed_content = self._preprocess_content(content)
        
        # Extract semantic units
        semantic_units = self._extract_semantic_units(processed_content)
        
        # Create embeddings for units
        unit_embeddings = self._create_unit_embeddings(semantic_units)
        
        # Apply semantic clustering
        clusters = self._semantic_clustering(semantic_units, unit_embeddings)
        
        # Generate chunks with semantic boundaries
        chunks = self._generate_semantic_chunks(
            semantic_units, clusters, file_path, document_structure
        )
        
        # Calculate chunking statistics
        chunking_metadata = self._calculate_chunking_metadata(
            chunks, semantic_units, time.time() - start_time
        )
        
        print(f"Semantic chunking completed: {len(chunks)} chunks in {chunking_metadata['processing_time']:.2f}s")
        
        return chunks, chunking_metadata
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for better semantic analysis"""
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Preserve important separators
        content = re.sub(r'([.!?])\s*', r'\1\n', content)
        
        # Clean up excessive punctuation
        content = re.sub(r'[!]{2,}', '!', content)
        content = re.sub(r'[?]{2,}', '?', content)
        
        return content.strip()
    
    def _extract_semantic_units(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract semantic units with enhanced context for reasoning
        """
        units = []
        
        # Split by paragraphs first for better context preservation
        paragraphs = re.split(r'\n\s*\n', content)
        
        unit_id = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 20:  # Skip very short paragraphs
                continue
            
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
            
            for sent_idx, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                    
                # Include context: previous sentence if available
                context = sentence.strip()
                if sent_idx > 0:
                    # Add previous sentence for context
                    context = sentences[sent_idx-1].strip() + " " + context
                
                unit = {
                    "id": unit_id,
                    "text": context,
                    "original_text": sentence.strip(),
                    "paragraph_id": para_idx,
                    "sentence_id": sent_idx,
                    "start_pos": content.find(sentence),
                    "end_pos": content.find(sentence) + len(sentence),
                    "length": len(context),
                    "type": self._classify_sentence_type(sentence),
                    "is_question": self._is_question(sentence),
                    "has_examples": self._has_examples(sentence)
                }
                
                units.append(unit)
                unit_id += 1
        
        return units
    
    def _is_question(self, text: str) -> bool:
        """Check if text contains a question"""
        return bool(re.search(r'\?|tại sao|như thế nào|làm sao|bao nhiêu|khi nào|đâu|ai|cái gì', text, re.IGNORECASE))
    
    def _has_examples(self, text: str) -> bool:
        """Check if text contains examples"""
        return bool(re.search(r'ví dụ|for example|such as|like:nền tảng', text, re.IGNORECASE))
    
    def _classify_sentence_type(self, sentence: str) -> str:
        """Classify sentence type for better chunking"""
        sentence = sentence.strip().lower()
        
        if any(marker in sentence for marker in ['?', 'tại sao', 'như thế nào']):
            return "question"
        elif any(marker in sentence for marker in [':', 'là', 'gồm', 'bao gồm']):
            return "definition"
        elif any(marker in sentence for marker in ['-', '•', '*']):
            return "list_item"
        elif re.match(r'^\d+\.', sentence):
            return "numbered_item"
        elif sentence.isupper() and len(sentence) > 5:
            return "heading"
        else:
            return "statement"
    
    def _create_unit_embeddings(self, units: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for semantic units with caching"""
        texts = [unit["text"] for unit in units]
        embeddings = []
        
        for i, text in enumerate(texts):
            # Check cache first
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                embedding = self.model.encode(text, convert_to_numpy=True)
                self.embedding_cache[cache_key] = embedding
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _semantic_clustering(self, units: List[Dict[str, Any]], 
                           embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform semantic clustering using agglomerative clustering
        """
        if len(units) <= 1:
            return [{"id": 0, "units": units, "score": 1.0}]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Perform clustering
        n_clusters = max(1, len(units) // 3)  # Adaptive cluster count
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(1 - similarity_matrix)
        
        # Organize clusters
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_units = [units[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if cluster_units:
                cluster_mask = cluster_labels == cluster_id
                cluster_similarities = similarity_matrix[cluster_mask][:, cluster_mask] if np.any(cluster_mask) else np.array([[1.0]])
                avg_similarity = np.mean(cluster_similarities) if cluster_similarities.size > 0 else 1.0
                
                cluster = {
                    "id": cluster_id,
                    "units": cluster_units,
                    "score": self._calculate_cluster_coherence(cluster_units, embeddings[cluster_mask]),
                    "avg_similarity": avg_similarity
                }
                clusters.append(cluster)
        
        # Sort clusters by coherence score
        clusters.sort(key=lambda x: x["score"], reverse=True)
        
        return clusters
    
    def _calculate_cluster_coherence(self, units: List[Dict[str, Any]], 
                                  cluster_embeddings: np.ndarray) -> float:
        """Calculate coherence score for a cluster"""
        if len(cluster_embeddings) <= 1:
            return 1.0
        
        similarities = cosine_similarity(cluster_embeddings)
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        coherence = np.mean(similarities[mask])
        
        return coherence
    
    def _generate_semantic_chunks(self, 
                                 semantic_units: List[Dict[str, Any]],
                                 clusters: List[Dict[str, Any]],
                                 file_path: Path,
                                 document_structure: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Generate final chunks from semantic clusters"""
        
        chunks = []
        chunk_index = 0
        
        for cluster in clusters:
            cluster_units = cluster["units"]
            
            if not cluster_units:
                continue
            
            # Group units into chunks of appropriate size
            current_chunk_units = []
            current_length = 0
            
            for unit in cluster_units:
                unit_length = len(unit["text"])
                
                # Check if adding this unit would exceed chunk size
                if current_length + unit_length > self.config.max_chunk_size and current_chunk_units:
                    # Create chunk from accumulated units
                    chunk = self._create_chunk_from_units(
                        current_chunk_units, chunk_index, file_path, cluster
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    current_chunk_units = current_chunk_units[-2:] if len(current_chunk_units) > 2 else []
                    current_length = sum(len(u["text"]) for u in current_chunk_units)
                
                current_chunk_units.append(unit)
                current_length += unit_length
            
            # Create final chunk for this cluster
            if current_chunk_units:
                chunk = self._create_chunk_from_units(
                    current_chunk_units, chunk_index, file_path, cluster
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _create_chunk_from_units(self, 
                                units: List[Dict[str, Any]],
                                chunk_index: int,
                                file_path: Path,
                                cluster: Dict[str, Any]) -> DocumentChunk:
        """Create DocumentChunk from semantic units"""
        
        # Combine unit texts
        content = " ".join([unit["text"] for unit in units])
        
        # Calculate positions
        start_pos = min(unit["start_pos"] for unit in units)
        end_pos = max(unit["end_pos"] for unit in units)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(units)
        
        # Create metadata
        metadata = {
            "cluster_id": cluster["id"],
            "cluster_coherence": cluster["score"],
            "unit_types": [unit["type"] for unit in units],
            "unit_count": len(units),
            "chunk_strategy": "semantic_advanced",
            "semantic_score": cluster.get("avg_similarity", 0.0)
        }
        
        # Handle both string and Path objects
        source_name = file_path.name if hasattr(file_path, 'name') else str(file_path)
        
        return DocumentChunk(
            content=content,
            metadata=metadata,
            chunk_id=self._generate_chunk_id(file_path, chunk_index),
            source_file=source_name,
            chunk_index=chunk_index,
            start_char=start_pos,
            end_char=end_pos,
            chunk_type=chunk_type,
            chunk_strategy="semantic_advanced",
            token_count=len(content.split()),
            language="vi"  # Default to Vietnamese
        )
    
    def _determine_chunk_type(self, units: List[Dict[str, Any]]) -> str:
        """Determine chunk type based on unit composition"""
        types = [unit["type"] for unit in units]
        
        question_ratio = types.count("question") / len(types)
        definition_ratio = types.count("definition") / len(types)
        
        if question_ratio > 0.3:
            return "qa_section"
        elif definition_ratio > 0.3:
            return "definition_section"
        elif "heading" in types:
            return "heading_section"
        else:
            return "content_section"
    
    def _calculate_chunking_metadata(self, chunks: List[DocumentChunk],
                                   semantic_units: List[Dict[str, Any]],
                                   processing_time: float) -> Dict[str, Any]:
        """Calculate comprehensive chunking metadata"""
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        coherence_scores = [chunk.metadata.get("cluster_coherence", 0) for chunk in chunks]
        
        return {
            "strategy": "semantic_advanced",
            "model_used": self.model_name,
            "total_chunks": len(chunks),
            "total_units": len(semantic_units),
            "processing_time": processing_time,
            "avg_chunk_size": np.mean(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_coherence": np.mean(coherence_scores),
            "config": {
                "max_chunk_size": self.config.max_chunk_size,
                "min_chunk_size": self.config.min_chunk_size,
                "overlap_size": self.config.overlap_size,
                "semantic_threshold": self.config.semantic_threshold
            }
        }
    
    def _generate_chunk_id(self, file_path: Path, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"{file_hash}_semantic_{chunk_index}_{timestamp}"


class MultiLevelChunker:
    """
    Multi-level chunking strategy that combines structural and semantic approaches
    """
    
    def __init__(self, semantic_chunker: AdvancedSemanticChunker):
        self.semantic_chunker = semantic_chunker
    
    def create_multi_level_chunks(self, 
                                 content: str, 
                                 file_path: Path,
                                 document_structure: Optional[Dict[str, Any]] = None) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Create chunks using multi-level approach:
        1. Document structure (headers, sections)
        2. Semantic clusters
        3. Size-based fallback
        """
        
        metadata = {
            "levels_used": [],
            "processing_time": 0,
            "chunk_counts": {}
        }
        
        start_time = time.time()
        
        # Level 1: Structure-based chunking
        if document_structure and document_structure.get("headers"):
            structure_chunks = self._create_structure_chunks(content, file_path, document_structure)
            metadata["levels_used"].append("structure")
            metadata["chunk_counts"]["structure"] = len(structure_chunks)
            
            if structure_chunks:
                # Validate structure chunks
                if self._validate_chunks(structure_chunks):
                    metadata["processing_time"] = time.time() - start_time
                    return structure_chunks, metadata
        
        # Level 2: Semantic chunking
        semantic_chunks, semantic_metadata = self.semantic_chunker.create_semantic_chunks(
            content, file_path, document_structure
        )
        metadata["levels_used"].append("semantic")
        metadata["chunk_counts"]["semantic"] = len(semantic_chunks)
        metadata.update(semantic_metadata)
        
        if self._validate_chunks(semantic_chunks):
            metadata["processing_time"] = time.time() - start_time
            return semantic_chunks, metadata
        
        # No fallback - if semantic chunking fails, raise error
        raise RuntimeError(f"Semantic chunking failed for document {file_path}. Generated {len(semantic_chunks)} chunks but validation failed.")
    
    def _create_structure_chunks(self, content: str, file_path: Path, 
                                structure: Dict[str, Any]) -> List[DocumentChunk]:
        """Create chunks based on document structure"""
        # Implementation similar to existing header-based chunking
        # but enhanced with semantic validation
        chunks = []
        headers = structure.get("headers", [])
        
        for i, header in enumerate(headers):
            start_line = header["line_number"]
            end_line = headers[i + 1]["line_number"] if i + 1 < len(headers) else len(content.split('\n'))
            
            # Extract section content
            lines = content.split('\n')
            section_content = "\n".join(lines[start_line:end_line]).strip()
            
            if len(section_content) > 50:
                chunk = DocumentChunk(
                    content=section_content,
                    metadata={
                        "header_text": header["text"],
                        "header_level": header["level"],
                        "chunk_strategy": "structure_based"
                    },
                    chunk_id=self._generate_chunk_id(file_path, i),
                    source_file=file_path.name,
                    chunk_index=i,
                    start_char=content.find(section_content),
                    end_char=content.find(section_content) + len(section_content),
                    chunk_type="header_section",
                    chunk_strategy="structure_based",
                    header_level=header["level"],
                    token_count=len(section_content.split())
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_fallback_chunks(self, content: str, file_path: Path) -> List[DocumentChunk]:
        """Create simple size-based chunks as fallback"""
        # Simple size-based chunking implementation
        chunk_size = 1000
        overlap = 200
        chunks = []
        
        words = content.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata={"chunk_strategy": "fallback_size_based"},
                chunk_id=self._generate_chunk_id(file_path, i // (chunk_size - overlap)),
                source_file=file_path.name,
                chunk_index=i // (chunk_size - overlap),
                start_char=i * 5,  # Approximate
                end_char=(i + len(chunk_words)) * 5,  # Approximate
                chunk_type="content",
                chunk_strategy="fallback_size_based",
                token_count=len(chunk_words)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _validate_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Validate chunk quality"""
        if not chunks:
            return False
        
        # Check size distribution
        sizes = [len(chunk.content) for chunk in chunks]
        avg_size = np.mean(sizes)
        
        # Check for reasonable size distribution
        if avg_size < 50 or avg_size > 2000:
            return False
        
        # Check for reasonable number of chunks
        if len(chunks) < 1 or len(chunks) > 100:
            return False
        
        return True
    
    def _generate_chunk_id(self, file_path: Path, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"{file_hash}_multi_{chunk_index}_{timestamp}"


# Factory classes
class AdvancedChunkerFactory:
    """Factory for creating advanced chunkers"""
    
    @staticmethod
    def create_semantic_chunker(
        model_name: str = "all-MiniLM-L6-v2",
        config: Optional[ChunkingConfig] = None
    ) -> AdvancedSemanticChunker:
        """Create semantic chunker with specified configuration"""
        return AdvancedSemanticChunker(model_name, config)
    
    @staticmethod
    def create_multilingual_chunker() -> AdvancedSemanticChunker:
        """Create chunker optimized for multilingual content"""
        return AdvancedSemanticChunker(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
    
    @staticmethod
    def create_trading_optimized_chunker() -> AdvancedSemanticChunker:
        """Create chunker optimized for trading documents"""
        config = ChunkingConfig(
            max_chunk_size=800,
            min_chunk_size=100,
            overlap_size=150,
            semantic_threshold=0.4,
            preserve_structure=True
        )
        return AdvancedSemanticChunker(
            model_name="all-MiniLM-L6-v2",
            config=config
        )
    
    @staticmethod
    def create_multi_level_chunker() -> MultiLevelChunker:
        """Create multi-level chunker"""
        semantic_chunker = AdvancedChunkerFactory.create_semantic_chunker()
        return MultiLevelChunker(semantic_chunker)