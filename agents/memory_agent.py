import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

from config import get_openai_llm


class MemoryEntry(BaseModel):
    """One entry in memory cache"""
    key: str = Field(description="Key to search (question or topic)")
    summary: str = Field("Summary of important knowledge")
    context: str = Field("Context related")
    timestamp: str = Field("Creation time")
    confidence: float = Field("Confidence of information (0-1)")
    tags: List[str] = Field("Tags to classify")
    source_turns: List[int] = Field("Source turns that created this knowledge")


class MemoryQuery(BaseModel):
    """Result of memory query"""
    can_answer: bool = Field("Can answer from cache")
    relevant_entries: List[MemoryEntry] = Field("Relevant entries")
    suggested_response: str = Field("Suggested response from cache")
    confidence: float = Field("Overall confidence")
    needs_refresh: bool = Field("Need to refresh cache")


class MemoryAgent:
    """
    Memory Agent smart to manage cache for multi-turn conversations
    
    Features:
    1. Summarize conversation history into knowledge cache
    2. Query cache to answer follow-up questions quickly
    3. Manage cache per user with UUID
    4. Automatically clean up and refresh cache
    """
    
    def __init__(self, cache_dir: str = ".memory_cache"):
        try:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.llm = get_openai_llm()
            
            # Cache size limit
            self.max_cache_size = 100  # entries per user
            self.max_cache_age_days = 1  # days
            
            print(f"✅ Memory Agent initialized with cache directory: {self.cache_dir}")
        except Exception as e:
            print(f"❌ Error initializing Memory Agent: {e}")
            # Fallback initialization
            self.cache_dir = Path(".memory_cache")
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.llm = None
            self.max_cache_size = 100
            self.max_cache_age_days = 1
        
    def _generate_user_uuid(self, user_name: str, birthday: str, username: str) -> str:
        """Create unique UUID for user based on personal information"""
        try:
            # Ensure parameters are not None and have values
            user_name = str(user_name) if user_name else "unknown"
            birthday = str(birthday) if birthday else "unknown"
            username = str(username) if username else "unknown"
            
            user_string = f"{user_name}_{birthday}_{username}"
            return hashlib.md5(user_string.encode('utf-8')).hexdigest()
        except Exception as e:
            print(f"❌ Error generating UUID: {e}")
            # Fallback UUID
            return hashlib.md5("unknown_user".encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, user_uuid: str) -> Path:
        """Get cache file path for user"""
        try:
            # Ensure user_uuid is a valid string
            if not user_uuid or not isinstance(user_uuid, str):
                user_uuid = "unknown_user"
            
            # Sanitize user_uuid to avoid invalid characters in file name
            safe_uuid = "".join(c for c in user_uuid if c.isalnum() or c in '_-')
            if not safe_uuid:
                safe_uuid = "unknown_user"
                
            return self.cache_dir / f"user_{safe_uuid}_memory.json"
        except Exception as e:
            print(f"❌ Error getting cache file path: {e}")
            return self.cache_dir / "user_unknown_memory.json"
    
    def _load_user_cache(self, user_uuid: str) -> List[MemoryEntry]:
        """Load cache from file for user"""
        cache_file = self._get_cache_file_path(user_uuid)
        if not cache_file.exists():
            return []
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Validate and create MemoryEntry objects
                entries = []
                for entry in data:
                    try:
                        # Ensure required fields are present
                        if isinstance(entry, dict) and 'key' in entry:
                            # Set default values for missing fields
                            entry_data = {
                                'key': entry.get('key', 'unknown'),
                                'summary': entry.get('summary', ''),
                                'context': entry.get('context', ''),
                                'timestamp': entry.get('timestamp', datetime.now().isoformat()),
                                'confidence': float(entry.get('confidence', 0.5)),
                                'tags': entry.get('tags', []),
                                'source_turns': entry.get('source_turns', [])
                            }
                            
                            memory_entry = MemoryEntry(**entry_data)
                            entries.append(memory_entry)
                        else:
                            print(f"⚠️ Invalid cache entry format: {entry}")
                    except Exception as e:
                        print(f"⚠️ Error creating MemoryEntry from {entry}: {e}")
                        continue
                
                print(f"✅ Cache loaded successfully: {len(entries)} entries")
                return entries
                
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return []
    
    def _save_user_cache(self, user_uuid: str, cache: List[MemoryEntry]):
        """Save cache for user to file"""
        cache_file = self._get_cache_file_path(user_uuid)
        try:
            # Convert MemoryEntry objects to dictionaries
            cache_data = []
            for entry in cache:
                if hasattr(entry, 'dict'):
                    entry_dict = entry.dict()
                else:
                    # Fallback if entry is not a MemoryEntry object
                    entry_dict = {
                        'key': getattr(entry, 'key', 'unknown'),
                        'summary': getattr(entry, 'summary', ''),
                        'context': getattr(entry, 'context', ''),
                        'timestamp': getattr(entry, 'timestamp', datetime.now().isoformat()),
                        'confidence': getattr(entry, 'confidence', 0.5),
                        'tags': getattr(entry, 'tags', []),
                        'source_turns': getattr(entry, 'source_turns', [])
                    }
                cache_data.append(entry_dict)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Cache saved successfully: {len(cache_data)} entries")
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
            print(f"❌ Cache data: {cache}")
    
    def _cleanup_old_entries(self, cache: List[MemoryEntry]) -> List[MemoryEntry]:
        """Delete old and expired entries"""
        if not cache:
            return []
            
        now = datetime.now()
        cutoff_date = now - timedelta(days=self.max_cache_age_days)
        
        # Filter by time
        fresh_cache = []
        for entry in cache:
            try:
                if hasattr(entry, 'timestamp') and entry.timestamp:
                    entry_date = datetime.fromisoformat(entry.timestamp)
                    if entry_date > cutoff_date:
                        fresh_cache.append(entry)
                else:
                    # If no timestamp, keep entry
                    fresh_cache.append(entry)
            except Exception as e:
                print(f"⚠️ Error parsing timestamp for entry {entry.key}: {e}")
                # Keep entry if there's an error parsing timestamp
                fresh_cache.append(entry)
        
        # Limit cache size
        if len(fresh_cache) > self.max_cache_size:
            # Sort by confidence and timestamp, keep the best entries
            try:
                fresh_cache.sort(key=lambda x: (getattr(x, 'confidence', 0.0), getattr(x, 'timestamp', '')))
                fresh_cache = fresh_cache[-self.max_cache_size:]  # Keep the newest entries
            except Exception as e:
                print(f"⚠️ Error sorting cache: {e}")
                # If sorting fails, keep the last entries
                fresh_cache = fresh_cache[-self.max_cache_size:]
        
        print(f"🧹 Cache cleanup: {len(cache)} → {len(fresh_cache)} entries")
        return fresh_cache
    
    def _extract_knowledge_from_conversation(
        self, 
        user_question: str, 
        lumir_response: str, 
        turn_number: int,
        language: str = "vi"
    ) -> List[MemoryEntry]:
        """
        Use LLM to extract important knowledge from conversation
        """
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Bạn là Memory Agent chuyên trích xuất kiến thức quan trọng từ conversation.

## 🌐 LANGUAGE INSTRUCTION
**IMPORTANT**: You MUST respond in the language specified by the user. If the user's language is "vi" (Vietnamese), respond in Vietnamese. If the user's language is "en" (English), respond in English. If the user's language is "zh" (Chinese), respond in Chinese. Maintain this language consistency throughout your response.

**CURRENT LANGUAGE**: {language}

Nhiệm vụ: Tóm tắt conversation thành các knowledge entries có thể tái sử dụng.

## QUY TẮC TRÍCH XUẤT:
1. **Chỉ lưu kiến thức có giá trị cao** - không lưu chào hỏi, thời tiết
2. **Tập trung vào insights về trading, tính cách, tâm lý**
3. **Tạo key ngắn gọn, dễ tìm kiếm**
4. **Đánh giá confidence dựa trên độ rõ ràng của thông tin**

## FORMAT JSON BẮT BUỘC:
```json
[
  {{
    "key": "trading_style",
    "summary": "User prefers swing trading with medium risk tolerance",
    "context": "Based on personality analysis and trading patterns",
    "confidence": 0.9,
    "tags": ["trading", "personality", "risk_tolerance"]
  }},
  {{
    "key": "emotional_issue",
    "summary": "User experiences FOMO when seeing price increases",
    "context": "User mentioned feeling fear of missing out",
    "confidence": 0.8,
    "tags": ["emotion", "fomo", "trading_psychology"]
  }}
]

## VÍ DỤ:
- "Tôi phù hợp với swing trading" → Key: "trading_style", Confidence: 0.9
- "Tôi thường bị FOMO" → Key: "emotional_issue", Confidence: 0.8
- "Xin chào" → Không lưu (không có kiến thức)

**QUAN TRỌNG**: Trả về JSON chính xác theo format trên, không thêm text khác."""),
                ("human", """
Conversation Turn {turn_number}:
User: {user_question}
LUMIR: {lumir_response}

Trích xuất kiến thức quan trọng:
""")
            ])
        except Exception as e:
            print(f"❌ Error creating prompt: {e}")
            return []
        
        try:
            # Kiểm tra LLM có sẵn không
            if not self.llm:
                print("⚠️ LLM not available, skipping knowledge extraction")
                return []
                
            # Gọi LLM trực tiếp để tránh vấn đề với parser
            try:
                response = self.llm.invoke(prompt.format_messages(
                    turn_number=turn_number,
                    user_question=user_question,
                    lumir_response=lumir_response,
                    language=language
                ))
            except Exception as e:
                print(f"❌ Error invoking LLM: {e}")
                return []
            
            # Kiểm tra response
            if not response or not hasattr(response, 'content'):
                print("⚠️ Invalid LLM response")
                return []
                
            # Parse response thủ công
            content = response.content
            if not content:
                print("⚠️ Empty LLM response")
                return []
                
            print(f"🔍 LLM Response: {content[:200]}...")
            
            # Tìm JSON trong response - cải thiện regex
            import re
            import json
            
            # Tìm JSON array
            json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if not json_match:
                # Thử tìm JSON object đơn lẻ
                json_match = re.search(r'\{\s*"key".*\}', content, re.DOTALL)
                if json_match:
                    # Wrap trong array
                    json_str = f"[{json_match.group()}]"
                else:
                    print("⚠️ No JSON found in LLM response")
                    return []
            else:
                json_str = json_match.group()
            
            try:
                # Clean JSON string
                json_str = json_str.strip()
                # Remove markdown code blocks if present
                json_str = re.sub(r'```json\s*', '', json_str)
                json_str = re.sub(r'```\s*$', '', json_str)
                
                parsed_data = json.loads(json_str)
                
                # Đảm bảo parsed_data là list
                if not isinstance(parsed_data, list):
                    parsed_data = [parsed_data]
                
                # Tạo MemoryEntry objects
                entries = []
                for item in parsed_data:
                    try:
                        # Validate required fields
                        if not isinstance(item, dict) or 'key' not in item:
                            print(f"⚠️ Invalid item format: {item}")
                            continue
                            
                        entry = MemoryEntry(
                            key=item.get('key', 'unknown'),
                            summary=item.get('summary', ''),
                            context=item.get('context', ''),
                            timestamp=datetime.now().isoformat(),
                            confidence=float(item.get('confidence', 0.5)),
                            tags=item.get('tags', []),
                            source_turns=[turn_number]
                        )
                        entries.append(entry)
                        print(f"✅ Created memory entry: {entry.key} (confidence: {entry.confidence})")
                    except Exception as e:
                        print(f"⚠️ Error creating entry from {item}: {e}")
                        continue
                
                return entries
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"❌ Raw JSON string: {json_str}")
                return []
                
        except Exception as e:
            print(f"❌ Error extracting knowledge: {e}")
            return []
    
    def update_memory(
        self, 
        user_uuid: str,
        user_question: str,
        lumir_response: str,
        turn_number: int,
        language: str = "vi"
    ) -> None:
        """
        Cập nhật memory cache với conversation mới
        """
        
        # Load cache hiện tại
        cache = self._load_user_cache(user_uuid)
        
        # Trích xuất kiến thức mới
        if not self.llm:
            print("⚠️ LLM not available, skipping memory update")
            return
            
        new_entries = self._extract_knowledge_from_conversation(
            user_question, lumir_response, turn_number, language
        )
        
        # Thêm source turn info - đảm bảo tất cả entries đều có source_turns
        for i, entry in enumerate(new_entries):
            if not hasattr(entry, 'source_turns') or not entry.source_turns:
                # Tạo entry mới với source_turns
                entry_dict = entry.dict() if hasattr(entry, 'dict') else entry
                entry_dict['source_turns'] = [turn_number]
                new_entries[i] = MemoryEntry(**entry_dict)
            else:
                # Đảm bảo turn_number có trong source_turns
                if turn_number not in entry.source_turns:
                    entry.source_turns.append(turn_number)
        
        # Merge với cache cũ (tránh duplicate)
        for new_entry in new_entries:
            # Tìm entry tương tự trong cache
            similar_entry = None
            for old_entry in cache:
                if (new_entry.key == old_entry.key and 
                    abs(new_entry.confidence - old_entry.confidence) < 0.1):
                    similar_entry = old_entry
                    break
            
            if similar_entry:
                # Update entry cũ
                similar_entry.summary = new_entry.summary
                similar_entry.confidence = max(similar_entry.confidence, new_entry.confidence)
                similar_entry.timestamp = new_entry.timestamp
                if hasattr(similar_entry, 'source_turns') and turn_number not in similar_entry.source_turns:
                    similar_entry.source_turns.append(turn_number)
            else:
                # Thêm entry mới
                cache.append(new_entry)
        
        # Cleanup và lưu
        cache = self._cleanup_old_entries(cache)
        self._save_user_cache(user_uuid, cache)
        
        print(f"✅ Memory updated: {len(new_entries)} new entries, total: {len(cache)}")
    
    def query_memory(
        self, 
        user_uuid: str, 
        current_question: str,
        conversation_history: List[Dict[str, Any]],
        language: str = "vi"
    ) -> MemoryQuery:
        """
        Query memory cache để xem có thể trả lời nhanh không
        """
        
        # Load cache
        cache = self._load_user_cache(user_uuid)
        if not cache:
            return MemoryQuery(
                can_answer=False,
                relevant_entries=[],
                suggested_response="",
                confidence=0.0,
                needs_refresh=False
            )
        
        # Kiểm tra LLM có sẵn không
        if not self.llm:
            print("⚠️ LLM not available, returning default memory query")
            return MemoryQuery(
                can_answer=False,
                relevant_entries=[],
                suggested_response="",
                confidence=0.0,
                needs_refresh=False
            )
            
        # Chuẩn bị data trước khi tạo prompt
        cache_entries = "\n".join([
            f"- {entry.key}: {entry.summary} (confidence: {entry.confidence})"
            for entry in cache
        ])
        
        recent_history = "\n".join([
            f"Turn {i+1}: {turn.get('user_question', '')[:100]}..."
            for i, turn in enumerate(conversation_history[-3:])
        ])
        
        try:
            # Sử dụng LLM để đánh giá relevance
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Bạn là Memory Query Agent. Nhiệm vụ: Đánh giá xem cache có thể trả lời câu hỏi hiện tại không.

## 🌐 LANGUAGE INSTRUCTION
**IMPORTANT**: You MUST respond in the language specified by the user. If the user's language is "vi" (Vietnamese), respond in Vietnamese. If the user's language is "en" (English), respond in English. If the user's language is "zh" (Chinese), respond in Chinese. Maintain this language consistency throughout your response.

**CURRENT LANGUAGE**: {language}

## QUY TẮC ĐÁNH GIÁ:
1. **Có thể trả lời**: Cache chứa thông tin đủ để trả lời câu hỏi
2. **Không thể trả lời**: Cache thiếu thông tin hoặc câu hỏi quá khác biệt
3. **Cần refresh**: Cache cũ hoặc không chính xác

## FORMAT JSON BẮT BUỘC:
```json
{{
  "can_answer": true,
  "confidence": 0.8,
  "needs_refresh": false,
  "suggested_response": "Dựa trên thông tin trong cache, bạn phù hợp với swing trading và cần kiểm soát FOMO."
}}
```

## OUTPUT FIELDS:
- can_answer: true/false
- confidence: 0.0-1.0 (độ tin cậy)
- needs_refresh: true nếu cache cũ hoặc không chính xác
- suggested_response: câu trả lời từ cache (nếu can_answer=true)

**QUAN TRỌNG**: Trả về JSON chính xác theo format trên, không thêm text khác."""),
                ("human", """
Câu hỏi hiện tại: {current_question}

Cache entries:
{cache_entries}

Conversation history (recent 3 turns):
{recent_history}

Đánh giá xem cache có thể trả lời câu hỏi không:
""")
            ])
        except Exception as e:
            print(f"❌ Error creating prompt in query_memory: {e}")
            return MemoryQuery(
                can_answer=False,
                relevant_entries=[],
                suggested_response="",
                confidence=0.0,
                needs_refresh=False
            )
        
        # Gọi LLM trực tiếp để tránh vấn đề với parser
        try:
            response = self.llm.invoke(prompt.format_messages(
                current_question=current_question,
                cache_entries=cache_entries,
                recent_history=recent_history,
                language=language
            ))
        except Exception as e:
            print(f"❌ Error invoking LLM in query_memory: {e}")
            return MemoryQuery(
                can_answer=False,
                relevant_entries=[],
                suggested_response="",
                confidence=0.0,
                needs_refresh=False
            )
        
        # Kiểm tra response
        if not response or not hasattr(response, 'content'):
            print("⚠️ Invalid LLM response in query_memory")
            return MemoryQuery(
                can_answer=False,
                relevant_entries=[],
                suggested_response="",
                confidence=0.0,
                needs_refresh=False
            )
            
        # Parse response thủ công
        content = response.content
        if not content:
            print("⚠️ Empty LLM response in query_memory")
            return MemoryQuery(
                can_answer=False,
                relevant_entries=[],
                suggested_response="",
                confidence=0.0,
                needs_refresh=False
            )
            
        print(f"🔍 Memory Query LLM Response: {content[:200]}...")
        
        # Tìm JSON trong response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # Clean JSON string
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            
            try:
                parsed_data = json.loads(json_str)
                
                # Tìm các entry liên quan trước
                relevant_entries = [
                    entry for entry in cache 
                    if entry.key.lower() in current_question.lower() or
                       any(tag.lower() in current_question.lower() for tag in entry.tags)
                ]
                
                # Tạo MemoryQuery object với relevant_entries đã có
                result = MemoryQuery(
                    can_answer=parsed_data.get('can_answer', False),
                    confidence=float(parsed_data.get('confidence', 0.0)),
                    needs_refresh=parsed_data.get('needs_refresh', False),
                    suggested_response=parsed_data.get('suggested_response', ''),
                    relevant_entries=relevant_entries
                )
                
                print(f"✅ Memory query successful: can_answer={result.can_answer}, confidence={result.confidence}")
                return result
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error in query_memory: {e}")
                print(f"❌ Raw JSON string: {json_str}")
        else:
            print("⚠️ No JSON found in memory query response")
        
        # Fallback nếu parsing thất bại
        return MemoryQuery(
            can_answer=False,
            relevant_entries=[],
            suggested_response="",
            confidence=0.0,
            needs_refresh=False
        )
    
    def get_memory_summary(self, user_uuid: str) -> Dict[str, Any]:
        """Lấy summary của memory cache"""
        cache = self._load_user_cache(user_uuid)
        
        if not cache:
            return {"status": "empty", "count": 0}
        
        try:
            # Thống kê
            total_entries = len(cache)
            
            # Tính average confidence an toàn
            confidence_values = []
            for entry in cache:
                if hasattr(entry, 'confidence') and entry.confidence is not None:
                    try:
                        confidence_values.append(float(entry.confidence))
                    except (ValueError, TypeError):
                        confidence_values.append(0.5)  # Default value
                else:
                    confidence_values.append(0.5)
            
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
            
            # Top knowledge areas
            key_counts = {}
            for entry in cache:
                if hasattr(entry, 'key') and entry.key:
                    key_counts[entry.key] = key_counts.get(entry.key, 0) + 1
            
            top_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Last updated timestamp
            last_updated = None
            try:
                timestamps = []
                for entry in cache:
                    if hasattr(entry, 'timestamp') and entry.timestamp:
                        try:
                            timestamps.append(entry.timestamp)
                        except:
                            continue
                
                if timestamps:
                    last_updated = max(timestamps)
            except Exception as e:
                print(f"⚠️ Error getting last updated: {e}")
            
            return {
                "status": "active",
                "total_entries": total_entries,
                "average_confidence": round(avg_confidence, 2),
                "top_knowledge_areas": top_keys,
                "last_updated": last_updated
            }
            
        except Exception as e:
            print(f"❌ Error generating memory summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "count": len(cache) if cache else 0
            }
    
    def add_memory_entry(
        self,
        user_uuid: str,
        entry_data: Dict[str, Any],
        language: str = "vi"
    ) -> bool:
        """
        Thêm entry mới vào memory cache
        
        Args:
            user_uuid: UUID của user
            entry_data: Dữ liệu entry (phải có 'key' field)
            language: Ngôn ngữ
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Validate entry_data
            if "key" not in entry_data:
                print("❌ Entry data must contain 'key' field")
                return False
            
            # Load cache hiện tại
            cache = self._load_user_cache(user_uuid)
            
            # Tạo MemoryEntry object
            entry = MemoryEntry(
                key=entry_data["key"],
                summary=entry_data.get("summary", ""),
                context=entry_data.get("context", ""),
                timestamp=datetime.now().isoformat(),
                confidence=entry_data.get("confidence", 0.8),
                tags=entry_data.get("tags", []),
                source_turns=entry_data.get("source_turns", [1])  # Default to turn 1
            )
            
            # Kiểm tra xem key đã tồn tại chưa
            existing_entry = None
            for old_entry in cache:
                if old_entry.key == entry.key:
                    existing_entry = old_entry
                    break
            
            if existing_entry:
                # Update existing entry
                existing_entry.summary = entry.summary
                existing_entry.context = entry.context
                existing_entry.timestamp = entry.timestamp
                existing_entry.confidence = max(existing_entry.confidence, entry.confidence)
                existing_entry.tags = list(set(existing_entry.tags + entry.tags))
                if 1 not in existing_entry.source_turns:
                    existing_entry.source_turns.append(1)
                print(f"✅ Updated existing entry: {entry.key}")
            else:
                # Add new entry
                cache.append(entry)
                print(f"✅ Added new entry: {entry.key}")
            
            # Cleanup và lưu
            cache = self._cleanup_old_entries(cache)
            self._save_user_cache(user_uuid, cache)
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding memory entry: {e}")
            return False
    
    def update_memory_entry(
        self,
        user_uuid: str,
        entry_key: str,
        entry_data: Dict[str, Any],
        language: str = "vi"
    ) -> bool:
        """
        Cập nhật entry hiện có trong memory cache
        
        Args:
            user_uuid: UUID của user
            entry_key: Key của entry cần cập nhật
            entry_data: Dữ liệu mới để cập nhật
            language: Ngôn ngữ
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Load cache hiện tại
            cache = self._load_user_cache(user_uuid)
            
            # Tìm entry cần cập nhật
            target_entry = None
            for entry in cache:
                if entry.key == entry_key:
                    target_entry = entry
                    break
            
            if not target_entry:
                print(f"❌ Entry with key '{entry_key}' not found")
                return False
            
            # Cập nhật các field được cung cấp
            if "summary" in entry_data:
                target_entry.summary = entry_data["summary"]
            if "context" in entry_data:
                target_entry.context = entry_data["context"]
            if "confidence" in entry_data:
                target_entry.confidence = entry_data["confidence"]
            if "tags" in entry_data:
                target_entry.tags = entry_data["tags"]
            if "source_turns" in entry_data:
                target_entry.source_turns = entry_data["source_turns"]
            
            # Cập nhật timestamp
            target_entry.timestamp = datetime.now().isoformat()
            
            # Lưu cache
            self._save_user_cache(user_uuid, cache)
            print(f"✅ Updated entry: {entry_key}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating memory entry: {e}")
            return False
    
    def remove_memory_entry(
        self,
        user_uuid: str,
        entry_key: str
    ) -> bool:
        """
        Xóa entry cụ thể khỏi memory cache
        
        Args:
            user_uuid: UUID của user
            entry_key: Key của entry cần xóa
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Load cache hiện tại
            cache = self._load_user_cache(user_uuid)
            
            # Tìm và xóa entry
            original_count = len(cache)
            cache = [entry for entry in cache if entry.key != entry_key]
            
            if len(cache) == original_count:
                print(f"❌ Entry with key '{entry_key}' not found")
                return False
            
            # Lưu cache đã cập nhật
            self._save_user_cache(user_uuid, cache)
            print(f"✅ Removed entry: {entry_key}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error removing memory entry: {e}")
            return False

    def clear_user_memory(self, user_uuid: str) -> None:
        """Xóa toàn bộ memory của user"""
        try:
            cache_file = self._get_cache_file_path(user_uuid)
            if cache_file.exists():
                cache_file.unlink()
                print(f"✅ Memory cleared for user {user_uuid}")
            else:
                print(f"ℹ️ No memory file found for user {user_uuid}")
        except Exception as e:
            print(f"❌ Error clearing memory for user {user_uuid}: {e}")


def build_memory_agent(cache_dir: str = ".memory_cache") -> MemoryAgent:
    """Factory function để tạo memory agent"""
    try:
        agent = MemoryAgent(cache_dir)
        print(f"✅ Memory Agent built successfully with cache dir: {cache_dir}")
        return agent
    except Exception as e:
        print(f"❌ Error building Memory Agent: {e}")
        # Fallback: tạo agent với cache directory mặc định
        return MemoryAgent(".memory_cache")
