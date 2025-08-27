"""
Embedding Manager Module
Quản lý embedding cho hệ thống RAG - Sử dụng Gemini embedding-002
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from dotenv import load_dotenv
import time

# Load .env from project root explicitly, then fallback to CWD
try:
    project_root_env = Path(__file__).resolve().parents[2] / ".env"
    if project_root_env.exists():
        load_dotenv(dotenv_path=project_root_env)
    else:
        load_dotenv()
except Exception:
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
    Quản lý embedding cho hệ thống RAG
    Hỗ trợ Gemini embedding-002 và sentence-transformers
    """
    
    def __init__(self, model_name: str = "embedding-002", provider: str = "gemini"):
        self.model_name = model_name
        self.provider = provider
        self.model = None
        self.embedding_dimension = None
        self.batch_size = 32
        # Simple rate limit control (requests per minute)
        self.rpm_limit = int(os.getenv("EMBEDDING_RPM_LIMIT", "80"))
        self._window_start = time.time()
        self._requests_in_window = 0
        # HF client
        self._hf_client = None
        
        # Khởi tạo model
        self._initialize_model()
    
    def _initialize_model(self):
        """Khởi tạo embedding model"""
        try:
            if self.provider == "hf" and HF_AVAILABLE:
                # LangChain HuggingFaceEmbeddings for Qwen3-Embedding-0.6B
                # Chuẩn hóa repo id: cần dạng org/model
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
                        "device": os.getenv("HF_DEVICE", "cuda")
                    },
                    encode_kwargs={
                        "batch_size": int(os.getenv("HF_EMBED_BATCH", "8"))
                    }
                )
                # Probe to get dimension
                try:
                    probe_vec = self.model.embed_query("probe")
                    default_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
                    self.embedding_dimension = len(probe_vec) if isinstance(probe_vec, list) else default_dim
                except Exception:
                    self.embedding_dimension = int(os.getenv("EMBEDDING_DIM", "1024"))
                print(f"✅ HF embeddings model loaded: {model_name}")
                print(f"📏 Embedding dimension: {self.embedding_dimension}")

            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                # Khởi tạo Gemini từ .env
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
                print(f"✅ Gemini embedding model loaded: {self.model}")
                print(f"📏 Embedding dimension: {self.embedding_dimension}")
                
            elif self.provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Không dùng fallback trong chế độ HF bắt buộc
                raise RuntimeError("SentenceTransformers fallback is disabled. HF provider required.")
            else:
                raise ValueError(f"Provider {self.provider} not available or not supported")
                
        except Exception as e:
            print(f"❌ Error initializing embedding model: {e}")
            # Không fallback, fail cứng theo yêu cầu
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
        Lấy embedding cho một đoạn text
        
        Args:
            text: Text cần embedding
            
        Returns:
            List embedding vector
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
                    print(f"❌ Error getting embedding (HF): {e}")
                    return [0.0] * (self.embedding_dimension or 0)

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
                    # Fallback: return empty
                    return []
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
                        return response.get("embedding", [])
                    return getattr(response, "embedding", [])
                
            elif self.provider == "sentence_transformers":
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Lấy embedding cho nhiều text cùng lúc (batch processing)
        
        Args:
            texts: List các text cần embedding
            
        Returns:
            List các embedding vector
        """
        try:
            if self.provider == "hf":
                try:
                    vecs = self.model.embed_documents(texts)
                    self._requests_in_window += len(texts)
                    return vecs
                except Exception as e:
                    print(f"❌ Error getting batch embeddings (HF): {e}")
                    embeddings = []
                    for text in texts:
                        embeddings.append(self.get_embedding(text))
                    return embeddings

            if self.provider == "gemini":
                embeddings = []
                # Gemini không hỗ trợ batch native → lặp từng phần tử + throttle
                for text in texts:
                    embedding = self.get_embedding(text)
                    embeddings.append(embedding)
                return embeddings
                
            elif self.provider == "sentence_transformers":
                embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_tensor=False)
                return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                
        except Exception as e:
            print(f"❌ Error getting batch embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dimension] * len(texts)
    
    def get_embedding_dimension(self) -> int:
        """Lấy kích thước embedding vector"""
        return self.embedding_dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float], method: str = "cosine") -> float:
        """
        Tính độ tương tự giữa 2 embedding vectors
        
        Args:
            embedding1: Vector 1
            embedding2: Vector 2
            method: Phương pháp tính ('cosine', 'euclidean', 'dot_product')
            
        Returns:
            Độ tương tự (0-1 cho cosine, càng nhỏ càng tương tự cho euclidean)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            if method == "cosine":
                # Cosine similarity: 1 = giống hệt, 0 = khác biệt hoàn toàn
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                    
                return dot_product / (norm1 * norm2)
                
            elif method == "euclidean":
                # Euclidean distance: 0 = giống hệt, càng lớn càng khác biệt
                distance = np.linalg.norm(vec1 - vec2)
                # Normalize về 0-1 (0 = giống hệt, 1 = khác biệt hoàn toàn)
                max_distance = np.sqrt(len(vec1))  # Giả sử vector được normalize
                return min(distance / max_distance, 1.0)
                
            elif method == "dot_product":
                # Dot product: càng lớn càng tương tự
                dot_product = np.dot(vec1, vec2)
                # Normalize về 0-1
                max_dot = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                return max(0, dot_product / max_dot) if max_dot > 0 else 0
                
            else:
                raise ValueError(f"Method {method} not supported")
                
        except Exception as e:
            print(f"❌ Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], candidate_embeddings: List[List[float]], 
                          top_k: int = 5, method: str = "cosine") -> List[tuple]:
        """
        Tìm top-k embedding vectors tương tự nhất
        
        Args:
            query_embedding: Embedding của query
            candidate_embeddings: List các embedding candidates
            top_k: Số lượng kết quả trả về
            method: Phương pháp tính similarity
            
        Returns:
            List các tuple (index, similarity_score)
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate, method)
                similarities.append((i, similarity))
            
            # Sắp xếp theo similarity (cao nhất trước)
            if method == "cosine" or method == "dot_product":
                similarities.sort(key=lambda x: x[1], reverse=True)
            else:  # euclidean
                similarities.sort(key=lambda x: x[1])  # Càng nhỏ càng tốt
            
            return similarities[:top_k]
            
        except Exception as e:
            print(f"❌ Error finding most similar: {e}")
            return []
    
    def optimize_batch_size(self, texts: List[str]) -> int:
        """
        Tối ưu batch size dựa trên độ dài text và memory available
        
        Args:
            texts: List các text cần xử lý
            
        Returns:
            Batch size tối ưu
        """
        try:
            if not TORCH_AVAILABLE:
                return 32  # Default
            
            # Tính tổng độ dài text
            total_length = sum(len(text) for text in texts)
            avg_length = total_length / len(texts) if texts else 0
            
            # Kiểm tra memory available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Ước tính memory cần thiết cho batch
                estimated_memory_per_text = avg_length * 4  # bytes per character
                optimal_batch = min(64, int(gpu_memory / (estimated_memory_per_text * 1000)))
            else:
                # CPU processing
                optimal_batch = min(32, max(8, int(1000 / avg_length)))
            
            return max(1, optimal_batch)
            
        except Exception as e:
            print(f"⚠️ Error optimizing batch size: {e}")
            return 32
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin về model hiện tại"""
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
        Tiền xử lý text cho trading documents
        
        Args:
            text: Text gốc
            
        Returns:
            Text đã được tiền xử lý
        """
        try:
            # Loại bỏ ký tự đặc biệt không cần thiết
            import re
            
            # Loại bỏ multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Loại bỏ ký tự đặc biệt như bullet points, arrows
            text = re.sub(r'[•→←↑↓]', ' ', text)
            
            # Chuẩn hóa dấu câu
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            
            # Loại bỏ dòng trống liên tiếp
            text = re.sub(r'\n\s*\n', '\n', text)
            
            return text.strip()
            
        except Exception as e:
            print(f"⚠️ Error preprocessing text: {e}")
            return text


class EmbeddingManagerFactory:
    """Factory để tạo embedding manager"""
    
    @staticmethod
    def create_manager(model_name: str = "embedding-002", provider: str = "gemini") -> EmbeddingManager:
        """Tạo embedding manager với cấu hình cụ thể"""
        return EmbeddingManager(model_name, provider)
    
    @staticmethod
    def create_optimal_manager() -> EmbeddingManager:
        """Tạo embedding manager tối ưu dựa trên environment"""
        # Kiểm tra environment variables
        google_key = os.getenv("GOOGLE_API_KEY")
        hf_token = os.getenv("HF_TOKEN")
        
        if hf_token and HF_AVAILABLE:
            print("🚀 Using HF embeddings (Qwen3-Embedding-0.6B)")
            return EmbeddingManager("Qwen3-Embedding-0.6B", "hf")
        elif google_key and GEMINI_AVAILABLE:
            print("🚀 Using Gemini embeddings (embedding-002)")
            return EmbeddingManager("embedding-002", "gemini")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🚀 Using SentenceTransformers (all-MiniLM-L6-v2)")
            return EmbeddingManager("all-MiniLM-L6-v2", "sentence_transformers")
        else:
            raise RuntimeError("No embedding provider available")
    
    @staticmethod
    def create_multilingual_manager() -> EmbeddingManager:
        """Tạo embedding manager hỗ trợ đa ngôn ngữ"""
        if GEMINI_AVAILABLE:
            # Gemini hỗ trợ đa ngôn ngữ tốt
            return EmbeddingManager("embedding-002", "gemini")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback to multilingual sentence-transformers
            return EmbeddingManager("paraphrase-multilingual-MiniLM-L12-v2", "sentence_transformers")
        else:
            # Fallback
            return EmbeddingManagerFactory.create_optimal_manager()
    
    @staticmethod
    def create_trading_optimized_manager() -> EmbeddingManager:
        """Tạo embedding manager tối ưu cho trading documents"""
        if GEMINI_AVAILABLE:
            # Gemini embedding-002 tốt cho domain-specific content
            return EmbeddingManager("embedding-002", "gemini")
        else:
            # Fallback to sentence-transformers
            return EmbeddingManager("all-MiniLM-L6-v2", "sentence_transformers")


class EmbeddingCache:
    """Cache cho embedding để tăng performance"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
        self._total_requests = 0
        self._cache_hits = 0
    
    def get(self, text: str) -> Optional[List[float]]:
        """Lấy embedding từ cache"""
        self._total_requests += 1
        
        if text in self.cache:
            self.access_count[text] += 1
            self._cache_hits += 1
            return self.cache[text]
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Lưu embedding vào cache"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])
            del self.cache[least_accessed[0]]
            del self.access_count[least_accessed[0]]
        
        self.cache[text] = embedding
        self.access_count[text] = 1
    
    def clear(self):
        """Xóa cache"""
        self.cache.clear()
        self.access_count.clear()
        self._total_requests = 0
        self._cache_hits = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self._calculate_hit_rate(),
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits
        }
    
    def _calculate_hit_rate(self) -> float:
        """Tính tỷ lệ hit của cache"""
        if self._total_requests == 0:
            return 0.0
        return self._cache_hits / self._total_requests
