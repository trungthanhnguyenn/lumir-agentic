import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from dotenv import load_dotenv
import time

load_dotenv()

# Embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

google_genai = None
genai_legacy = None
try:
    # New SDK style
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        # Fallback to older google.generativeai
        import google.generativeai as genai_legacy
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HuggingFaceEmbeddings = None
    HF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EmbeddingManager:
    """
    Manage embedding for RAG system
    Hỗ trợ Gemini embedding-002 và sentence-transformers
    """
    
    def __init__(self, model_name: str = "Qwen3-Embedding-0.6B", provider: str = "hf"):
        self.model_name = model_name
        self.provider = provider
        self.model = None
        self.embedding_dimension = 1024  # Default for Qwen3-0.6B
        self.batch_size = 32
        # Simple rate limit control (requests per minute)
        self.rpm_limit = int(os.getenv("EMBEDDING_RPM_LIMIT", "80"))
        self._window_start = time.time()
        self._requests_in_window = 0
        # HF client
        self._hf_client = None
        
        # Detect optimal device
        self.device = self._detect_device()
        
        # Initialize model
        self._initialize_model()
    
    def _detect_device(self) -> str:
        """Detect optimal device for embedding computation"""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        try:
            import torch
            import platform
            
            # Check if Apple Silicon (M1/M2/M3)
            if platform.system() == "Darwin" and platform.machine() in ["arm64", "arm64e"]:
                if torch.backends.mps.is_available():
                    print("🍎 Apple Silicon GPU (MPS) detected and available")
                    return "mps"
                else:
                    print("🍎 Apple Silicon detected, using CPU")
                    return "cpu"
            
            # Check CUDA availability
            elif torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"🚀 CUDA detected with {gpu_count} GPU(s)")
                return "cuda"
            
            # Default to CPU
            else:
                print("💻 Using CPU for embeddings")
                return "cpu"
                
        except Exception as e:
            print(f"⚠️ Device detection failed, using CPU: {e}")
            return "cpu"
    
    def _initialize_model(self):
        """Initialize embedding model"""
        try:
            if self.provider == "hf" and HF_AVAILABLE:
                # LangChain HuggingFaceEmbeddings for Qwen3-Embedding-0.6B
                # Normalize repo id: need org/model
                model_name = self.model_name or "Qwen/Qwen3-Embedding-0.6B"
                if "/" not in model_name:
                    model_name = f"Qwen/{model_name}"
                # Allow token via HF_TOKEN or HUGGINGFACEHUB_API_TOKEN
                token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                # Some environments auto-pick token; pass if available
                if token:
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
                self.model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        "trust_remote_code": True,
                        "device": self.device  # Use auto-detected device
                    },
                    encode_kwargs={
                        "batch_size": int(os.getenv("HF_EMBED_BATCH", "8"))
                    }
                )
                # Probe to get dimension - no fallback allowed
                probe_vec = self.model.embed_query("probe")
                if not isinstance(probe_vec, list) or len(probe_vec) == 0:
                    raise RuntimeError("Embedding model returned invalid probe vector")
                self.embedding_dimension = len(probe_vec)
                print(f"Verified embedding dimension: {self.embedding_dimension}")
                print(f"HF embeddings model loaded: {model_name}")
                print(f"Embedding dimension: {self.embedding_dimension}")

            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                # Initialize Gemini from .env
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment variables")

                # Prefer new client API
                if google_genai is not None:
                    self._gemini_client = google_genai.Client(api_key=api_key)
                    self.model = "gemini-embedding-001"
                else:
                    # Legacy SDK fallback
                    genai_legacy.configure(api_key=api_key)
                    self._gemini_client = None
                    # legacy embed_content expects "models/embedding-001"
                    self.model = "models/embedding-001"

                # Dimension of Gemini embeddings
                self.embedding_dimension = 768
                print(f"Gemini embedding model loaded: {self.model}")
                print(f"Embedding dimension: {self.embedding_dimension}")
                
            elif self.provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
                # No fallback in HF mode
                raise RuntimeError("SentenceTransformers fallback is disabled. HF provider required.")
            else:
                raise ValueError(f"Provider {self.provider} not available or not supported")
                
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            # No fallback, fail hard
            raise

    def _throttle(self):
        """Simple client-side throttle to respect RPM limits for free tier."""
        try:
            if self.rpm_limit <= 0:
                return
            now = time.time()
            elapsed = now - self._window_start
            # Reset window after 60s
            if elapsed >= 60:
                self._window_start = now
                self._requests_in_window = 0
                return
            # If exceeding limit, sleep until window resets
            if self._requests_in_window >= self.rpm_limit:
                sleep_for = 60 - elapsed + 0.1
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self._window_start = time.time()
                self._requests_in_window = 0
        except Exception:
            # Never block on throttle errors
            pass
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding vector
        """
        try:
            # Throttle per-call
            self._throttle()
            if self.provider == "hf":
                # HuggingFaceEmbeddings via LangChain
                self._throttle()
                try:
                    vec = self.model.embed_query(text)
                    self._requests_in_window += 1
                    return vec
                except Exception as e:
                    print(f"Error getting embedding (HF): {e}")
                    raise RuntimeError(f"Failed to generate embedding with HuggingFace: {e}")

            if self.provider == "gemini":
                # New client first
                if hasattr(self, "_gemini_client") and self._gemini_client is not None:
                    # Retry with exponential backoff on 429
                    backoff = 1.0
                    for attempt in range(4):
                        try:
                            result = self._gemini_client.models.embed_content(
                                model=self.model,
                                content=text
                            )
                            self._requests_in_window += 1
                            break
                        except Exception as e:
                            msg = str(e)
                            if "429" in msg and attempt < 3:
                                time.sleep(backoff)
                                backoff *= 2
                                continue
                            raise
                    # New client returns object with .embedding.values list or dict
                    if hasattr(result, "embedding") and hasattr(result.embedding, "values"):
                        return list(result.embedding.values)
                    if isinstance(result, dict):
                        emb = result.get("embedding")
                        if isinstance(emb, dict) and "values" in emb:
                            return list(emb["values"]) 
                        if isinstance(emb, list):
                            return emb
                    # No fallback - raise error if embedding generation fails
                    raise RuntimeError(f"Failed to generate embedding with Gemini. Invalid response format: {result}")
                else:
                    # Legacy SDK path
                    backoff = 1.0
                    for attempt in range(4):
                        try:
                            response = genai_legacy.embed_content(
                                model=self.model,
                                content=text,
                                task_type="retrieval_document"
                            )
                            self._requests_in_window += 1
                            break
                        except Exception as e:
                            msg = str(e)
                            if "429" in msg and attempt < 3:
                                time.sleep(backoff)
                                backoff *= 2
                                continue
                            raise
                    if isinstance(response, dict):
                        embedding = response.get("embedding")
                        if not embedding:
                            raise RuntimeError("Gemini legacy API returned empty embedding")
                        return embedding
                    embedding = getattr(response, "embedding", None)
                    if not embedding:
                        raise RuntimeError("Gemini legacy API returned no embedding")
                    return embedding
                
            elif self.provider == "sentence_transformers":
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # No fallback - propagate error to caller
            raise RuntimeError(f"Failed to generate embedding for text: {e}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embedding for multiple texts at once (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.provider == "hf":
                try:
                    vecs = self.model.embed_documents(texts)
                    self._requests_in_window += len(texts)
                    return vecs
                except Exception as e:
                    print(f"Error getting batch embeddings (HF): {e}")
                    embeddings = []
                    for text in texts:
                        embeddings.append(self.get_embedding(text))
                    return embeddings

            if self.provider == "gemini":
                embeddings = []
                # Gemini doesn't support native batch → loop through each element + throttle
                for text in texts:
                    embedding = self.get_embedding(text)
                    embeddings.append(embedding)
                return embeddings
                
            elif self.provider == "sentence_transformers":
                embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_tensor=False)
                return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            # No fallback - propagate error to caller
            raise RuntimeError(f"Failed to generate batch embeddings: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding vector dimension"""
        return self.embedding_dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float], method: str = "cosine") -> float:
        """
        Calculate similarity between two embedding vectors
        
        Args:
            embedding1: Vector 1
            embedding2: Vector 2
            method: Similarity calculation method ('cosine', 'euclidean', 'dot_product')
            
        Returns:
            Similarity (0-1 for cosine, smaller for euclidean)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            if method == "cosine":
                # Cosine similarity: 1 = identical, 0 = completely different
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                    
                return dot_product / (norm1 * norm2)
                
            elif method == "euclidean":
                # Euclidean distance: 0 = identical, larger = more different
                distance = np.linalg.norm(vec1 - vec2)
                # Normalize to 0-1 (0 = identical, 1 = completely different)
                max_distance = np.sqrt(len(vec1))  # Assume vector is normalized
                return min(distance / max_distance, 1.0)
                
            elif method == "dot_product":
                # Dot product: larger = more similar
                dot_product = np.dot(vec1, vec2)
                # Normalize to 0-1
                max_dot = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                return max(0, dot_product / max_dot) if max_dot > 0 else 0
                
            else:
                raise ValueError(f"Method {method} not supported")
                
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], candidate_embeddings: List[List[float]], 
                          top_k: int = 5, method: str = "cosine") -> List[tuple]:
        """
        Find top-k embedding vectors most similar
        
        Args:
            query_embedding: Embedding of query
            candidate_embeddings: List of embedding candidates
            top_k: Number of results to return
            method: Similarity calculation method
            
        Returns:
            List of tuples (index, similarity_score)
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate, method)
                similarities.append((i, similarity))
            
            # Sort by similarity (highest first)
            if method == "cosine" or method == "dot_product":
                similarities.sort(key=lambda x: x[1], reverse=True)
            else:  # euclidean
                similarities.sort(key=lambda x: x[1])  # Smaller is better
            
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error finding most similar: {e}")
            return []
    
    def optimize_batch_size(self, texts: List[str]) -> int:
        """
        Optimize batch size based on text length and memory available
        
        Args:
            texts: List of texts to process
            
        Returns:
            Optimal batch size
        """
        try:
            if not TORCH_AVAILABLE:
                return 32  # Default
            
            # Calculate total text length
            total_length = sum(len(text) for text in texts)
            avg_length = total_length / len(texts) if texts else 0
            
            # Check memory available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Estimate memory required for batch
                estimated_memory_per_text = avg_length * 4  # bytes per character
                optimal_batch = min(64, int(gpu_memory / (estimated_memory_per_text * 1000)))
            else:
                # CPU processing
                optimal_batch = min(32, max(8, int(1000 / avg_length)))
            
            return max(1, optimal_batch)
            
        except Exception as e:
            print(f"Error optimizing batch size: {e}")
            return 32
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "batch_size": self.batch_size,
            "available_providers": {
                "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
                "gemini": GEMINI_AVAILABLE,
                "torch": TORCH_AVAILABLE
            }
        }
    
    def preprocess_text_for_trading(self, text: str) -> str:
        """
        Preprocess text for trading documents
        
        Args:
            text: Original text
            
        Returns:
            Text preprocessed
        """
        try:
            # Remove unnecessary special characters
            import re
            
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters like bullet points, arrows
            text = re.sub(r'[•→←↑↓]', ' ', text)
            
            # Normalize punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            
            # Remove consecutive empty lines
            text = re.sub(r'\n\s*\n', '\n', text)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return text


class EmbeddingManagerFactory:
    """Factory to create embedding manager"""
    
    @staticmethod
    def create_manager(model_name: str = "embedding-002", provider: str = "gemini") -> EmbeddingManager:
        """Create embedding manager with specific configuration"""
        return EmbeddingManager(model_name, provider)
    
    @staticmethod
    def create_optimal_manager() -> EmbeddingManager:
        """Create embedding manager optimized based on environment"""
        # Check environment variables
        google_key = os.getenv("GOOGLE_API_KEY")
        hf_token = os.getenv("HF_TOKEN")
        
        if hf_token and HF_AVAILABLE:
            print("Using HF embeddings (Qwen3-Embedding-0.6B)")
            return EmbeddingManager("Qwen3-Embedding-0.6B", "hf")
        elif google_key and GEMINI_AVAILABLE:
            print("Using Gemini embeddings (embedding-002)")
            return EmbeddingManager("embedding-002", "gemini")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Using SentenceTransformers (all-MiniLM-L6-v2)")
            return EmbeddingManager("all-MiniLM-L6-v2", "sentence_transformers")
        else:
            raise RuntimeError("No embedding provider available")
    
    @staticmethod
    def create_multilingual_manager() -> EmbeddingManager:
        """Create embedding manager supporting multiple languages"""
        if GEMINI_AVAILABLE:
            # Gemini supports multiple languages well
            return EmbeddingManager("embedding-002", "gemini")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback to multilingual sentence-transformers
            return EmbeddingManager("paraphrase-multilingual-MiniLM-L12-v2", "sentence_transformers")
        else:
            # Fallback
            return EmbeddingManagerFactory.create_optimal_manager()
    
    @staticmethod
    def create_trading_optimized_manager() -> EmbeddingManager:
        """Create embedding manager optimized for trading documents"""
        if GEMINI_AVAILABLE:
            # Gemini embedding-002 is good for domain-specific content
            return EmbeddingManager("embedding-002", "gemini")
        else:
            # Fallback to sentence-transformers
            return EmbeddingManager("all-MiniLM-L6-v2", "sentence_transformers")


class EmbeddingCache:
    """Cache for embedding to improve performance"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
        self._total_requests = 0
        self._cache_hits = 0
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        self._total_requests += 1
        
        if text in self.cache:
            self.access_count[text] += 1
            self._cache_hits += 1
            return self.cache[text]
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Save embedding to cache"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])
            del self.cache[least_accessed[0]]
            del self.access_count[least_accessed[0]]
        
        self.cache[text] = embedding
        self.access_count[text] = 1
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()
        self._total_requests = 0
        self._cache_hits = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self._calculate_hit_rate(),
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self._total_requests == 0:
            return 0.0
        return self._cache_hits / self._total_requests
