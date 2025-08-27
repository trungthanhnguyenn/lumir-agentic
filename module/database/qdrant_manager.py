"""
Qdrant Manager Module
Quản lý Qdrant vector database cho hệ thống RAG
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid

# Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Numpy for vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class CollectionConfig:
    """Cấu hình cho collection"""
    name: str
    vector_size: int
    distance: str = "Cosine"  # Cosine, Euclidean, Dot
    on_disk: bool = False
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    optimizers_config: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Kết quả tìm kiếm từ Qdrant"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class QdrantManager:
    """
    Quản lý Qdrant vector database
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_prefix: str = "lumir_rag"):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        self.host = host
        self.port = port
        self.collection_prefix = collection_prefix
        self.client = None
        self.collections = {}
        
        # Kết nối đến Qdrant
        self._connect()
    
    def _connect(self):
        """Kết nối đến Qdrant server"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # Test connection
            collections = self.client.get_collections()
            print(f"✅ Connected to Qdrant at {self.host}:{self.port}")
            print(f"📊 Available collections: {len(collections.collections)}")
            
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(self, config: CollectionConfig) -> bool:
        """
        Tạo collection mới
        
        Args:
            config: Cấu hình collection
            
        Returns:
            True nếu thành công
        """
        try:
            collection_name = f"{self.collection_prefix}_{config.name}"
            
            # Kiểm tra collection đã tồn tại chưa
            if self._collection_exists(collection_name):
                print(f"⚠️ Collection {collection_name} already exists")
                return True
            
            # Tạo collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config.vector_size,
                    distance=self._get_distance(config.distance),
                    on_disk=config.on_disk,
                    hnsw_config=models.HnswConfigDiff(
                        m=config.hnsw_m,
                        ef_construct=config.hnsw_ef_construct
                    )
                ),
                optimizers_config=config.optimizers_config or {}
            )
            
            print(f"✅ Collection {collection_name} created successfully")
            self.collections[collection_name] = config
            return True
            
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return False
    
    def _get_distance(self, distance_name: str) -> Distance:
        """Chuyển đổi tên distance sang Distance enum"""
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        return distance_map.get(distance_name, Distance.COSINE)
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Kiểm tra collection có tồn tại không"""
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception:
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Xóa collection"""
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            self.client.delete_collection(full_name)
            print(f"✅ Collection {full_name} deleted")
            
            if full_name in self.collections:
                del self.collections[full_name]
            
            return True
            
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            return False
    
    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """
        Thêm hoặc cập nhật points vào collection
        
        Args:
            collection_name: Tên collection
            points: List các points với format:
                   {
                       'id': str,
                       'vector': List[float],
                       'payload': Dict[str, Any]
                   }
            
        Returns:
            True nếu thành công
        """
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            
            # Kiểm tra collection tồn tại
            if not self._collection_exists(full_name):
                print(f"❌ Collection {full_name} does not exist")
                return False
            
            # Trước khi upsert, bỏ qua các point đã tồn tại (id trùng)
            try:
                full_name = f"{self.collection_prefix}_{collection_name}"
                existing_ids = set()
                # Lấy batch ids hiện có (nếu collection lớn có thể tối ưu hoá bằng filter by ids chunk)
                ids_to_check = [p['id'] for p in points if 'id' in p]
                if ids_to_check:
                    # Qdrant không có API get-by-ids trực tiếp qua client http models, dùng scroll theo filter id
                    # Đơn giản hoá: tiếp tục và để upsert ghi đè (idempotent). Nếu muốn strict-skip, uncomment phần dưới khi có API phù hợp.
                    pass
            except Exception:
                pass

            # Chuyển đổi points sang PointStruct
            qdrant_points = []
            for point in points:
                # Ensure ID is a UUID string or integer as Qdrant expects
                pid = point['id']
                try:
                    import uuid as _uuid
                    # Convert non-UUID string to UUIDv5 stable
                    if isinstance(pid, str):
                        try:
                            _ = _uuid.UUID(pid)
                        except Exception:
                            pid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, pid))
                except Exception:
                    pass
                qdrant_point = PointStruct(
                    id=pid,
                    vector=point['vector'],
                    payload=point['payload']
                )
                qdrant_points.append(qdrant_point)
            
            # Upsert points
            self.client.upsert(
                collection_name=full_name,
                points=qdrant_points
            )
            
            print(f"✅ Upserted {len(points)} points to {full_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error upserting points: {e}")
            return False
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10, score_threshold: float = 0.7,
               with_payload: bool = True, with_vectors: bool = False) -> List[SearchResult]:
        """
        Tìm kiếm trong collection
        
        Args:
            collection_name: Tên collection
            query_vector: Vector query
            limit: Số lượng kết quả tối đa
            score_threshold: Ngưỡng điểm tối thiểu
            with_payload: Có trả về payload không
            with_vectors: Có trả về vectors không
            
        Returns:
            List các SearchResult
        """
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            
            # Kiểm tra collection tồn tại
            if not self._collection_exists(full_name):
                print(f"❌ Collection {full_name} does not exist")
                return []
            
            # Thực hiện tìm kiếm
            search_result = self.client.search(
                collection_name=full_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            # Chuyển đổi kết quả
            results = []
            for point in search_result:
                result = SearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload or {},
                    vector=point.vector if with_vectors else None
                )
                results.append(result)
            
            print(f"🔍 Found {len(results)} results in {full_name}")
            return results
            
        except Exception as e:
            print(f"❌ Error searching collection: {e}")
            return []
    
    def search_with_filter(self, collection_name: str, query_vector: List[float],
                          filter_conditions: Dict[str, Any], limit: int = 10,
                          score_threshold: float = 0.7) -> List[SearchResult]:
        """
        Tìm kiếm với filter conditions
        
        Args:
            collection_name: Tên collection
            query_vector: Vector query
            filter_conditions: Điều kiện filter
            limit: Số lượng kết quả tối đa
            score_threshold: Ngưỡng điểm tối thiểu
            
        Returns:
            List các SearchResult
        """
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            
            # Kiểm tra collection tồn tại
            if not self._collection_exists(full_name):
                print(f"❌ Collection {full_name} does not exist")
                return []
            
            # Tạo filter
            qdrant_filter = self._create_filter(filter_conditions)
            
            # Thực hiện tìm kiếm với filter
            search_result = self.client.search(
                collection_name=full_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Chuyển đổi kết quả
            results = []
            for point in search_result:
                result = SearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload or {}
                )
                results.append(result)
            
            print(f"🔍 Found {len(results)} filtered results in {full_name}")
            return results
            
        except Exception as e:
            print(f"❌ Error searching with filter: {e}")
            return []
    
    def _create_filter(self, conditions: Dict[str, Any]) -> models.Filter:
        """Tạo Qdrant filter từ conditions"""
        must_conditions = []
        should_conditions = []
        must_not_conditions = []
        
        for field, condition in conditions.items():
            if isinstance(condition, dict):
                # Complex condition
                for op, value in condition.items():
                    if op == "eq":
                        must_conditions.append(models.FieldCondition(
                            key=field, match=models.MatchValue(value=value)
                        ))
                    elif op == "ne":
                        must_not_conditions.append(models.FieldCondition(
                            key=field, match=models.MatchValue(value=value)
                        ))
                    elif op == "in":
                        must_conditions.append(models.FieldCondition(
                            key=field, match=models.MatchAny(any=value)
                        ))
                    elif op == "gt":
                        must_conditions.append(models.FieldCondition(
                            key=field, range=models.DatetimeRange(
                                gt=value
                            )
                        ))
                    elif op == "lt":
                        must_conditions.append(models.FieldCondition(
                            key=field, range=models.DatetimeRange(
                                lt=value
                            )
                        ))
            else:
                # Simple equality
                must_conditions.append(models.FieldCondition(
                    key=field, match=models.MatchValue(value=condition)
                ))
        
        return models.Filter(
            must=must_conditions,
            should=should_conditions,
            must_not=must_not_conditions
        )
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin collection"""
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            
            if not self._collection_exists(full_name):
                return None
            
            info = self.client.get_collection(full_name)
            return {
                "name": info.name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
            
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """Liệt kê tất cả collections"""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections 
                   if col.name.startswith(self.collection_prefix)]
        except Exception as e:
            print(f"❌ Error listing collections: {e}")
            return []
    
    def create_payload_index(self, collection_name: str, field_name: str, 
                            field_type: str = "keyword") -> bool:
        """
        Tạo index cho payload field để tăng tốc filter
        
        Args:
            collection_name: Tên collection
            field_name: Tên field
            field_type: Loại field (keyword, integer, float, text)
            
        Returns:
            True nếu thành công
        """
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            
            if not self._collection_exists(full_name):
                print(f"❌ Collection {full_name} does not exist")
                return False
            
            # Tạo payload index
            self.client.create_payload_index(
                collection_name=full_name,
                field_name=field_name,
                field_schema=self._get_field_schema(field_type)
            )
            
            print(f"✅ Created payload index for {field_name} in {full_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating payload index: {e}")
            return False
    
    def _get_field_schema(self, field_type: str) -> "models.PayloadSchemaType":
        """Lấy schema cho field type (Qdrant >= v1.7 uses PayloadSchemaType)."""
        try:
            schema_type = models.PayloadSchemaType
        except AttributeError:
            # Backward fallback to older name if present
            schema_type = getattr(models, "PayloadFieldSchema", None)
            if schema_type is None:
                raise
        mapping = {
            "keyword": getattr(schema_type, "KEYWORD"),
            "integer": getattr(schema_type, "INTEGER"),
            "float": getattr(schema_type, "FLOAT"),
            "text": getattr(schema_type, "TEXT"),
        }
        return mapping.get(field_type, getattr(schema_type, "KEYWORD"))
    
    def batch_upsert(self, collection_name: str, points: List[Dict[str, Any]], 
                     batch_size: int = 100) -> bool:
        """
        Upsert points theo batch để tối ưu performance
        
        Args:
            collection_name: Tên collection
            points: List các points
            batch_size: Kích thước batch
            
        Returns:
            True nếu thành công
        """
        try:
            total_points = len(points)
            success_count = 0
            
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                
                if self.upsert_points(collection_name, batch):
                    success_count += len(batch)
                    print(f"📦 Processed batch {i//batch_size + 1}: {len(batch)} points")
                else:
                    print(f"❌ Failed to process batch {i//batch_size + 1}")
            
            print(f"✅ Batch upsert completed: {success_count}/{total_points} points")
            return success_count == total_points
            
        except Exception as e:
            print(f"❌ Error in batch upsert: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Lấy thống kê collection"""
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            
            if not self._collection_exists(full_name):
                return None
            
            info = self.client.get_collection(full_name)
            name = getattr(info, "name", full_name)
            config = getattr(info, "config", None)
            params = getattr(config, "params", None) if config else None
            vectors = getattr(params, "vectors", None) if params else None
            size = getattr(vectors, "size", None)
            distance = getattr(vectors, "distance", None)
            on_disk = getattr(vectors, "on_disk", None)
            points_count = getattr(info, "points_count", None)
            segments_count = getattr(info, "segments_count", None)
            status = getattr(info, "status", None)
            return {
                "name": name,
                "points_count": points_count,
                "segments_count": segments_count,
                "config": {
                    "vector_size": size,
                    "distance": str(distance) if distance is not None else None,
                    "on_disk": on_disk
                },
                "status": str(status) if status is not None else None
            }
            
        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return None


class QdrantManagerFactory:
    """Factory để tạo Qdrant manager"""
    
    @staticmethod
    def create_manager(host: str = "localhost", port: int = 6333, 
                      collection_prefix: str = "lumir_rag") -> QdrantManager:
        """Tạo Qdrant manager với cấu hình cụ thể"""
        return QdrantManager(host, port, collection_prefix)
    
    @staticmethod
    def create_local_manager(collection_prefix: str = "lumir_rag") -> QdrantManager:
        """Tạo Qdrant manager cho local development"""
        return QdrantManager("localhost", 6333, collection_prefix)
    
    @staticmethod
    def create_cloud_manager(url: str, api_key: str, 
                           collection_prefix: str = "lumir_rag") -> QdrantManager:
        """Tạo Qdrant manager cho cloud deployment"""
        # Parse URL để lấy host và port
        if url.startswith("http"):
            url = url.replace("http://", "").replace("https://", "")
        
        if ":" in url:
            host, port_str = url.split(":")
            port = int(port_str)
        else:
            host = url
            port = 6333
        
        manager = QdrantManager(host, port, collection_prefix)
        # Set API key if needed
        if hasattr(manager.client, 'set_api_key'):
            manager.client.set_api_key(api_key)
        
        return manager


class RAGCollectionManager:
    """Quản lý collections cho hệ thống RAG LUMIR - Tối ưu cho 2 collections chính"""
    
    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant = qdrant_manager
        self.collections_config = self._get_optimized_collections()
    
    def _get_optimized_collections(self) -> Dict[str, CollectionConfig]:
        """
        Tạo cấu hình collections tối ưu dựa trên đặc điểm documents:
        
        1. FAQ Collection: Gộp FAQ behavior + training (cùng loại Q&A)
        2. Knowledge Base: Gộp Handbook + Presentation (cùng loại kiến thức)
        """
        return {
            "faq": CollectionConfig(
                name="faq",
                vector_size=1024,  # Qwen3-Embedding-0.6B dimension
                distance="Cosine",
                on_disk=True,
                hnsw_m=16,
                hnsw_ef_construct=100,
                optimizers_config={
                    "default_segment_number": 2,
                    "memmap_threshold": 20000
                }
            ),
            "knowledge_base": CollectionConfig(
                name="knowledge_base", 
                vector_size=1024,  # Qwen3-Embedding-0.6B dimension
                distance="Cosine",
                on_disk=True,
                hnsw_m=16,
                hnsw_ef_construct=100,
                optimizers_config={
                    "default_segment_number": 2,
                    "memmap_threshold": 20000
                }
            )
        }
    
    def setup_collections(self) -> bool:
        """Thiết lập tất cả collections cần thiết"""
        try:
            success_count = 0
            
            for name, config in self.collections_config.items():
                if self.qdrant.create_collection(config):
                    success_count += 1
                    
                    # Tạo payload indexes cho các field quan trọng
                    self._create_payload_indexes(name)
                else:
                    print(f"❌ Failed to create collection: {name}")
            
            print(f"✅ Setup collections: {success_count}/{len(self.collections_config)} successful")
            return success_count == len(self.collections_config)
            
        except Exception as e:
            print(f"❌ Error setting up collections: {e}")
            return False
    
    def _create_payload_indexes(self, collection_name: str):
        """Tạo payload indexes cho collection"""
        try:
            # Index cho các field thường dùng để filter
            index_fields = [
                ("document_type", "keyword"),      # faq_behavior, faq_training, handbook, presentation
                ("chunk_type", "keyword"),         # header, content, qa_pair, list_item
                ("language", "keyword"),           # vi, en
                ("source_file", "keyword"),        # Tên file gốc
                ("chunk_strategy", "keyword"),     # header_based, qa_based, size_based, semantic_based
                ("header_level", "integer"),       # Cấp độ header (0, 1, 2, 3)
                ("qa_count", "integer"),           # Số lượng Q&A trong chunk
                ("token_count", "integer")         # Số lượng tokens
            ]
            
            for field_name, field_type in index_fields:
                self.qdrant.create_payload_index(collection_name, field_name, field_type)
                
        except Exception as e:
            print(f"⚠️ Error creating payload indexes for {collection_name}: {e}")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Lấy trạng thái tất cả collections"""
        status = {}
        
        for name in self.collections_config.keys():
            stats = self.qdrant.get_collection_stats(name)
            status[name] = stats if stats else {"status": "not_found"}
        
        return status
    
    def get_collection_for_document(self, filename: str) -> str:
        """
        Xác định collection phù hợp cho document dựa trên tên file
        
        Args:
            filename: Tên file document
            
        Returns:
            Tên collection: 'faq' hoặc 'knowledge_base'
        """
        filename_lower = filename.lower()
        
        # FAQ collection: Chứa các câu hỏi và trả lời
        if any(keyword in filename_lower for keyword in ['faq', 'behavior', 'training']):
            return "faq"
        
        # Knowledge base: Chứa kiến thức tổng hợp, hướng dẫn
        elif any(keyword in filename_lower for keyword in ['handbook', 'present', 'guide', 'manual']):
            return "knowledge_base"
        
        # Default: Đặt vào knowledge_base
        else:
            return "knowledge_base"
    
    def get_collection_description(self) -> Dict[str, str]:
        """Lấy mô tả chi tiết về mục đích của từng collection"""
        return {
            "faq": "Chứa các câu hỏi thường gặp, ví dụ hỏi đáp, lộ trình huấn luyện. Tối ưu cho semantic search về Q&A.",
            "knowledge_base": "Chứa kiến thức tổng hợp, hướng dẫn sử dụng, tính năng, khái niệm. Tối ưu cho comprehensive search."
        }
