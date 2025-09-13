import json
import os
from pathlib import Path
from fastapi import UploadFile, File
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import hashlib

# Add root directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agents.question_decomposition_agent import build_question_decomposition_agent
from agents.tbi_agent import build_tbi_agent
from agents.trading_agent import build_trading_agent
from agents.lumir_synthesis_agent import build_lumir_synthesis_agent
from agents.memory_agent import build_memory_agent
from tools.data_validator_tool import DataValidator


class LUMIRAPIEndpoints:
    """
    API Endpoints for each agent
    Each agent can operate independently or combined
    """
    
    def __init__(self):
        """Initialize agents"""
        self.question_decomposer = build_question_decomposition_agent()
        self.tbi_agent = build_tbi_agent()
        self.trading_agent = build_trading_agent()
        self.lumir_agent = build_lumir_synthesis_agent()
        self.memory_agent = build_memory_agent()
        # Chatbot lazy init
        self._chatbot = None
        
        # Conversation history for multi-turn (in-memory cache)
        self.conversation_history = {}
        
        # Directory to persist conversation history on disk
        self._history_dir = Path(__file__).parent / ".memory_cache"
        self._history_dir.mkdir(exist_ok=True, parents=True)
        
        print("LUMIR-AI API Endpoints initialized successfully!")
    
    def _generate_user_uuid(self, user_name: str, birthday: str, username: str) -> str:
        """Generate a unique UUID for user"""
        try:
            user_name = str(user_name) if user_name else "unknown"
            birthday = str(birthday) if birthday else "unknown"
            username = str(username) if username else "unknown"
            
            user_string = f"{user_name}_{birthday}_{username}"
            return hashlib.md5(user_string.encode('utf-8')).hexdigest()
        except Exception as e:
            print(f"Error generating UUID: {e}")
            return hashlib.md5("unknown_user".encode('utf-8')).hexdigest()
    
    def _get_history_file_path(self, user_uuid: str) -> Path:
        """Path to persist history for user."""
        safe_uuid = "".join(c for c in user_uuid if c.isalnum())
        if not safe_uuid:
            safe_uuid = "unknown"
        return self._history_dir / f"user_{safe_uuid}_history.json"
    
    def _load_history_from_disk(self, user_uuid: str) -> List[Dict[str, Any]]:
        """Read conversation history from file if exists."""
        try:
            history_file = self._get_history_file_path(user_uuid)
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # Ensure each turn has basic fields
                    normalized: List[Dict[str, Any]] = []
                    for turn in data:
                        if isinstance(turn, dict):
                            normalized.append({
                                "user_question": turn.get("user_question", ""),
                                "lumir_response": turn.get("lumir_response", ""),
                                "timestamp": turn.get("timestamp", datetime.now().isoformat())
                            })
                    return normalized
        except Exception as e:
            print(f"Failed to load history from disk for {user_uuid}: {e}")
        return []
    
    def _save_history_to_disk(self, user_uuid: str, history: List[Dict[str, Any]]):
        """Save conversation history to file."""
        try:
            history_file = self._get_history_file_path(user_uuid)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save history to disk for {user_uuid}: {e}")
    
    def _get_user_conversation_history(self, user_uuid: str) -> List[Dict[str, Any]]:
        """Get conversation history of user (RAM); if not exists, load from disk."""
        if user_uuid not in self.conversation_history:
            # Lazy-load from persist file if exists
            self.conversation_history[user_uuid] = self._load_history_from_disk(user_uuid)
        return self.conversation_history.get(user_uuid, [])
    
    def _update_user_conversation_history(self, user_uuid: str, question: str, response: str):
        """Update conversation history of user (RAM + persist)."""
        if user_uuid not in self.conversation_history:
            self.conversation_history[user_uuid] = self._load_history_from_disk(user_uuid)
        
        turn_info = {
            "user_question": question,
            "lumir_response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history[user_uuid].append(turn_info)
        
        # Limit history to avoid overload
        if len(self.conversation_history[user_uuid]) > 100:
            self.conversation_history[user_uuid] = self.conversation_history[user_uuid][-100:]
        
        # Persist to disk after each update
        try:
            self._save_history_to_disk(user_uuid, self.conversation_history[user_uuid])
        except Exception as e:
            print(f"Persist conversation history failed for {user_uuid}: {e}")


# ============================================================================
# ENDPOINT 1: MEMORY AGENT
# ============================================================================

    def memory_history_endpoint(
        self,
        user_name: str,
        birthday: str,
        username: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Return conversation history for user (to pass to synthesize)
        
        Args:
            user_name: User name
            birthday: Birthday
            username: Username
            limit: Number of recent turns (optional)
        """
        try:
            user_uuid = self._generate_user_uuid(user_name, birthday, username)
            history = self._get_user_conversation_history(user_uuid)

            # Fallback: if history in RAM is empty, read from .memory_cache
            if not history:
                try:
                    cache_dir = Path(__file__).parent / ".memory_cache"
                    cache_file = cache_dir / f"user_{user_uuid}_memory.json"
                    if cache_file.exists():
                        with open(cache_file, "r", encoding="utf-8") as f:
                            cache_entries = json.load(f)
                        # Convert MemoryEntry → pseudo conversation turns
                        pseudo_history: List[Dict[str, Any]] = []
                        for entry in cache_entries:
                            # Protect missing fields
                            key = entry.get("key", "")
                            summary = entry.get("summary", "")
                            context = entry.get("context", "")
                            timestamp = entry.get("timestamp", datetime.now().isoformat())
                            confidence = entry.get("confidence", 0.0)
                            tags = entry.get("tags", [])
                            # Create a readable turn
                            pseudo_history.append({
                                "user_question": f"[cache:{key}] {summary}",
                                "lumir_response": context,
                                "timestamp": timestamp,
                                "confidence": confidence,
                                "tags": tags
                            })
                        # Sort by timestamp if exists
                        try:
                            pseudo_history.sort(key=lambda t: t.get("timestamp", ""))
                        except Exception:
                            pass
                        history = pseudo_history
                except Exception as e:
                    print(f"Fallback read from .memory_cache failed: {e}")
                    # Keep history empty if error
                    history = []

            # Apply limit if exists
            if limit is not None and isinstance(limit, int) and limit > 0:
                history = history[-limit:]

            return {
                "endpoint": "memory_history",
                "success": True,
                "user_uuid": user_uuid,
                "count": len(history),
                "conversation_history": history,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "endpoint": "memory_history",
                "success": False,
                "error": f"Failed to get conversation history: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def memory_check_endpoint(
        self,
        question: str,
        user_name: str,
        birthday: str,
        username: str,
        language: str = "vi"
    ) -> Dict[str, Any]:
        """
        Endpoint 1: Check memory cache
        
        Args:
            question: User's question
            user_name: User name
            birthday: Birthday
            username: Username
            language: Language
            
        Returns:
            Dict containing memory check result
        """
        
        try:
            print(f"Memory Check Endpoint - Question: {question}")
            
            # Create user UUID
            user_uuid = self._generate_user_uuid(user_name, birthday, username)
            
            # Get conversation history
            conversation_history = self._get_user_conversation_history(user_uuid)
            
            # Query memory cache for answer
            memory_query = self.memory_agent.query_memory(
                user_uuid, question, conversation_history, language
            )
            
            result = {
                "endpoint": "memory_check",
                "success": True,
                "question": question,
                "user_uuid": user_uuid,
                "can_answer_from_cache": memory_query.can_answer,
                "cache_confidence": memory_query.confidence,
                "suggested_response": memory_query.suggested_response,
                "needs_refresh": memory_query.needs_refresh,
                "relevant_entries": [
                    {
                        "key": entry.key,
                        "summary": entry.summary,
                        "confidence": entry.confidence
                    }
                    for entry in memory_query.relevant_entries
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Memory Check completed - Cache hit: {memory_query.can_answer}")
            return result
            
        except Exception as e:
            error_msg = f"Memory check failed: {str(e)}"
            print(f"{error_msg}")
            return {
                "endpoint": "memory_check",
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# ENDPOINT 2: QUESTION DECOMPOSITION AGENT
# ============================================================================

    def question_decomposition_endpoint(
        self,
        question: str,
        user_name: Optional[str] = None,
        birthday: Optional[str] = None,
        excel_path: Optional[UploadFile] = File(None),
        language: str = "vi",
        username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Endpoint 2: Analyze question and decide routing
        
        Args:
            question: User's question
            user_name: User name (optional)
            birthday: Birthday (optional)
            excel_path: Excel file path (optional)
            language: Language
            username: Username (optional)
            
        Returns:
            Dict containing question analysis result
        """
        
        try:
            print(f"Question Decomposition Endpoint - Question: {question}")
            
            # Validate trading data if exists
            has_valid_trading_data = False
            if excel_path:
                validator = DataValidator()
                if os.path.exists(excel_path):
                    try:
                        import pandas as pd
                        df = pd.read_excel(excel_path)
                        validation = validator.validate_excel_dataframe(df)
                        has_valid_trading_data = validation["is_valid"]
                    except Exception:
                        has_valid_trading_data = False
            
            # Call question decomposition agent
            decomposition_result = self.question_decomposer(
                question=question,
                user_name=user_name,
                birthday=birthday,
                excel_path=excel_path,
                language=language,
                username=username,
                conversation_history=[]
            )
            
            # Update has_valid_trading_data
            decomposition_result["has_valid_trading_data"] = has_valid_trading_data
            
            result = {
                "endpoint": "question_decomposition",
                "success": True,
                "question": question,
                "decomposition_result": decomposition_result,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Question Decomposition completed - Type: {decomposition_result.get('question_type')}")
            return result
            
        except Exception as e:
            error_msg = f"Question decomposition failed: {str(e)}"
            print(f"{error_msg}")
            return {
                "endpoint": "question_decomposition",
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# ENDPOINT 3: TBI AGENT
# ============================================================================

    def tbi_endpoint(
        self,
        question: str,
        user_name: str,
        birthday: str,
        language: str = "vi"
    ) -> Dict[str, Any]:
        """
        Endpoint 3: Analyze TBI (Trading Behavior Intelligence)
        
        Args:
            question: Question about TBI
            user_name: User name
            birthday: Birthday
            language: Language
            
        Returns:
            Dict containing TBI analysis result
        """
        
        try:
            print(f"TBI Endpoint - Question: {question}")
            
            # Handle None or empty question - return empty response instead of error
            if question is None or question.strip() == "":
                print("TBI question is None or empty - returning empty response")
                return {
                    "endpoint": "tbi",
                    "success": True,
                    "question": question,
                    "user_name": user_name,
                    "birthday": birthday,
                    "tbi_response": "",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert birthday format to dd/mm/yyyy if needed
            converted_birthday = self._convert_birthday_format(birthday)
            
            # Call TBI agent
            inputs = {
                "question": question,
                "user_name": user_name,
                "birthday": converted_birthday,
                "language": language
            }
            
            tbi_response = self.tbi_agent.invoke(inputs)
            
            return {
                "endpoint": "tbi",
                "success": True,
                "question": question,
                "user_name": user_name,
                "birthday": birthday,
                "tbi_response": str(tbi_response),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"TBI Endpoint Error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "endpoint": "tbi",
                "success": False,
                "error": str(e),
                "question": question,
                "user_name": user_name,
                "birthday": birthday,
                "tbi_response": "",
                "timestamp": datetime.now().isoformat()
            }

# ENDPOINT 4: NUMEROLOGY AGENT (DEPRECATED)
# ============================================================================

    def numerology_endpoint(
        self,
        question: str,
        user_name: str,
        birthday: str,
        language: str = "vi"
    ) -> Dict[str, Any]:
        """
        Endpoint 3: Analyze numerology
        
        Args:
            question: Question about numerology
            user_name: User name
            birthday: Birthday
            language: Language
            
        Returns:
            Dict containing numerology analysis result
        """
        
        try:
            print(f"Numerology Endpoint - Question: {question}")
            
            # Handle None or empty question - return empty response instead of error
            if question is None or question.strip() == "":
                print("Numerology question is None or empty - returning empty response")
                return {
                    "endpoint": "numerology",
                    "success": True,
                    "question": question,
                    "user_name": user_name,
                    "birthday": birthday,
                    "numerology_response": "",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert birthday format to dd/mm/yyyy if needed
            converted_birthday = self._convert_birthday_format(birthday)
            
            # Call numerology agent
            inputs = {
                "question": question,
                "user_name": user_name,
                "birthday": converted_birthday,
                "language": language
            }
            
            result = self.tbi_agent.invoke(inputs)
            numerology_response = str(result) if result else ""
            
            api_result = {
                "endpoint": "numerology",
                "success": True,
                "question": question,
                "user_name": user_name,
                "birthday": converted_birthday,
                "numerology_response": numerology_response,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Numerology analysis completed - Response length: {len(numerology_response)}")
            return api_result
            
        except Exception as e:
            error_msg = f"Numerology analysis failed: {str(e)}"
            print(f"{error_msg}")
            return {
                "endpoint": "numerology",
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _convert_birthday_format(self, birthday: str) -> str:
        """
        Convert birthday from various formats to dd/mm/yyyy format expected by CalNum
        
        Args:
            birthday: Birthday string in various formats
            
        Returns:
            Birthday string in dd/mm/yyyy format
        """
        if not birthday or birthday == "Unknown":
            return "01/01/1990"  # Default fallback
            
        try:
            # Handle different date formats
            if "-" in birthday:
                # ISO format (1990-05-15) or dash format (15-05-1990)
                parts = birthday.split("-")
                if len(parts) == 3:
                    if len(parts[0]) == 4:  # ISO format: 1990-05-15
                        year, month, day = parts[0], parts[1], parts[2]
                    else:  # Dash format: 15-05-1990
                        day, month, year = parts[0], parts[1], parts[2]
                    return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
                    
            elif "/" in birthday:
                # Slash format: could be 1990/05/15 or 15/05/1990
                parts = birthday.split("/")
                if len(parts) == 3:
                    if len(parts[0]) == 4:  # 1990/05/15
                        year, month, day = parts[0], parts[1], parts[2]
                    else:  # 15/05/1990
                        day, month, year = parts[0], parts[1], parts[2]
                    return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
                    
            # If no separators found, assume it's already in correct format
            return birthday
            
        except Exception as e:
            print(f"Warning: Could not convert birthday format '{birthday}': {e}")
            return "01/01/1990"  # Default fallback


# ============================================================================
# ENDPOINT 4: TRADING AGENT
# ============================================================================

    def trading_endpoint(
        self,
        question: str,
        excel_path: str,
        language: str = "vi",
        has_trading_data: bool = False
    ) -> Dict[str, Any]:
        """
        Endpoint 4: Analyze trading data
        
        Args:
            question: Question about trading
            excel_path: Excel file path
            language: Language
            has_trading_data: Whether user has valid trading data
            
        Returns:
            Dict containing trading analysis result
        """
        
        try:
            print(f"Trading Endpoint - Question: {question}")
            print(f"Has Trading Data: {has_trading_data}")
            
            # Handle None or empty question - return empty response instead of error
            if question is None or question.strip() == "":
                print("Trading question is None or empty - returning empty response")
                return {
                    "endpoint": "trading",
                    "success": True,
                    "question": question,
                    "excel_path": excel_path,
                    "has_trading_data": has_trading_data,
                    "trading_response": "",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Handle trading data path logic
            if has_trading_data and not excel_path:
                # If has_trading_data is True but no excel_path provided, use sample data
                excel_path = "trading_data/test_sample.xlsx"
            elif not has_trading_data:
                # If has_trading_data is False, clear excel_path
                excel_path = ""
            
            # Call trading agent with has_trading_data flag
            inputs = {
                "question": question,
                "excel_path": excel_path,
                "language": language,
                "has_trading_data": has_trading_data
            }
            
            result = self.trading_agent.invoke(inputs)  # Use .invoke() method for Langchain
            trading_response = str(result) if result else ""
            
            api_result = {
                "endpoint": "trading",
                "success": True,
                "question": question,
                "excel_path": excel_path,
                "has_trading_data": has_trading_data,
                "trading_response": trading_response,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Trading analysis completed - Response length: {len(trading_response)}")
            return api_result
            
        except Exception as e:
            error_msg = f"Trading analysis failed: {str(e)}"
            print(f"{error_msg}")
            return {
                "endpoint": "trading",
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# ENDPOINT 5: LUMIR-AI SYNTHESIS AGENT
# ============================================================================

    def lumir_synthesis_endpoint(
        self,
        question: str,
        question_type: str = "general_chat",
        tbi_context: str = "",
        trading_context: str = "",
        user_name: str = "",
        username: str = "",
        language: str = "vi",
        has_trading_data: bool = False,
        focus_areas: Optional[List[str]] = None,
        needs_user_info: bool = False,
        suggested_questions: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Endpoint 5: Synthesize and create final response
        
        Args:
            question: Original question
            question_type: Question type
            tbi_context: Context from TBI agent
            trading_context: Context from trading agent
            user_name: User name
            username: Username
            language: Language
            has_trading_data: Has trading data
            focus_areas: Focus areas
            needs_user_info: Need user info
            suggested_questions: Suggested questions
            conversation_history: Conversation history
            
        Returns:
            Dict containing final response from LUMIR-AI
        """
        
        try:
            print(f"LUMIR-AI Synthesis Endpoint - Question: {question}")
            
            # Prepare input dictionary for the chain
            inputs = {
                "question": question,
                "question_type": question_type,
                "tbi_context": tbi_context,
                "trading_context": trading_context,
                "user_name": user_name,
                "username": username,
                "language": language,
                "has_trading_data": has_trading_data,
                "focus_areas": focus_areas or [],
                "needs_user_info": needs_user_info,
                "suggested_questions": suggested_questions or [],
                "conversation_history": conversation_history or []
            }
            
            # Call LUMIR-AI agent using invoke method
            lumir_response = self.lumir_agent.invoke(inputs)
            
            api_result = {
                "endpoint": "lumir_synthesis",
                "success": True,
                "question": question,
                "question_type": question_type,
                "tbi_available": bool(tbi_context),
                "trading_available": bool(trading_context),
                "lumir_response": lumir_response,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"LUMIR-AI synthesis completed - Response length: {len(lumir_response)}")
            return api_result
            
        except Exception as e:
            error_msg = f"LUMIR-AI synthesis failed: {str(e)}"
            print(f"{error_msg}")
            return {
                "endpoint": "lumir_synthesis",
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# ENDPOINT 6: COMPLETE PIPELINE (Mô phỏng test_infer)
# ============================================================================

    def complete_pipeline_endpoint(
        self,
        question: str,
        user_name: Optional[str] = None,
        birthday: Optional[str] = None,
        excel_path: Optional[str] = None,
        language: str = "vi",
        username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Endpoint 6: Complete pipeline (mô phỏng logic test_infer)
        
        Args:
            question: User's question
            user_name: User name
            birthday: Birthday
            excel_path: Excel file path
            language: Language
            username: Username
            
        Returns:
            Dict containing complete pipeline result
        """
        
        start_time = datetime.now()
        
        try:
            print(f"Complete Pipeline Endpoint - Question: {question}")
            
            # Step 1: Memory Check
            print("Step 1: Memory Check...")
            memory_result = self.memory_check_endpoint(
                question, user_name or "Unknown", birthday or "Unknown", username or "Unknown", language
            )
            
            if memory_result["success"] and memory_result["can_answer_from_cache"] and memory_result["cache_confidence"] > 0.7:
                print(f"Cache hit - Confidence: {memory_result['cache_confidence']:.2f}")
                
                # Update conversation history
                user_uuid = memory_result["user_uuid"]
                self._update_user_conversation_history(user_uuid, question, memory_result["suggested_response"])
                
                return {
                    "endpoint": "complete_pipeline",
                    "success": True,
                    "question": question,
                    "response": memory_result["suggested_response"],
                    "processing_time": 0.1,
                    "question_type": "cached_response",
                    "source": "memory_cache",
                    "cache_confidence": memory_result["cache_confidence"],
                    "timestamp": datetime.now().isoformat()
                }
            
            # Step 2: Question Decomposition
            print("Step 2: Question Decomposition...")
            decomposition_result = self.question_decomposition_endpoint(
                question, user_name, birthday, excel_path, language, username
            )
            
            if not decomposition_result["success"]:
                raise Exception(f"Question decomposition failed: {decomposition_result['error']}")
            
            decomposition_data = decomposition_result["decomposition_result"]
            question_type = decomposition_data.get("question_type", "general_chat")
            should_call_agents = decomposition_data.get("should_call_agents", False)
            
            # Step 3: Execute Specialized Agents (if needed)
            tbi_context = ""
            trading_context = ""
            
            if should_call_agents:
                print("Step 3: Executing Specialized Agents...")
                
                # TBI Agent
                if decomposition_data.get("tbi_question") and user_name and birthday:
                    print("Calling TBI Agent...")
                    tbi_result = self.tbi_endpoint(
                        decomposition_data["tbi_question"], user_name, birthday, language
                    )
                    if tbi_result["success"]:
                        tbi_context = tbi_result["tbi_response"]
                
                # Trading Agent
                if (decomposition_data.get("trading_question") and 
                    decomposition_data.get("has_valid_trading_data") and excel_path):
                    print("Calling Trading Agent...")
                    trading_result = self.trading_endpoint(
                        decomposition_data["trading_question"], excel_path, language, decomposition_data.get("has_valid_trading_data", False)
                    )
                    if trading_result["success"]:
                        trading_context = trading_result["trading_response"]
            
            # Step 4: LUMIR-AI Synthesis
            print("Step 4: LUMIR-AI Synthesis...")
            
            # Lấy conversation history nếu có user info
            conversation_history = []
            if user_name and birthday and username:
                user_uuid = self._generate_user_uuid(user_name, birthday, username)
                conversation_history = self._get_user_conversation_history(user_uuid)
            
            lumir_result = self.lumir_synthesis_endpoint(
                question=question,
                question_type=question_type,
                tbi_context=tbi_context,
                trading_context=trading_context,
                user_name=user_name or "",
                username=username or "",
                language=language,
                has_trading_data=decomposition_data.get("has_valid_trading_data", False),
                focus_areas=decomposition_data.get("focus_areas", []),
                needs_user_info=decomposition_data.get("needs_user_info", False),
                suggested_questions=decomposition_data.get("suggested_questions", []),
                conversation_history=conversation_history
            )
            
            if not lumir_result["success"]:
                raise Exception(f"LUMIR-AI synthesis failed: {lumir_result['error']}")
            
            # Step 5: Update Memory Cache
            if user_name and birthday and username:
                print("Step 5: Updating Memory Cache...")
                user_uuid = self._generate_user_uuid(user_name, birthday, username)
                turn_number = len(conversation_history) + 1
                
                try:
                    self.memory_agent.update_memory(
                        user_uuid, question, lumir_result["lumir_response"], turn_number, language
                    )
                except Exception as e:
                    print(f"Memory update failed: {e}")
                
                # Update conversation history
                self._update_user_conversation_history(user_uuid, question, lumir_result["lumir_response"])
            
            # Tính thời gian xử lý
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "endpoint": "complete_pipeline",
                "success": True,
                "question": question,
                "response": lumir_result["lumir_response"],
                "processing_time": processing_time,
                "question_type": question_type,
                "decomposition_result": decomposition_data,
                "context_summary": {
                    "tbi_available": bool(tbi_context),
                    "trading_available": bool(trading_context),
                    "response_type": "comprehensive" if (tbi_context and trading_context) else "partial",
                    "needs_user_info": decomposition_data.get("needs_user_info", False),
                    "suggested_questions": decomposition_data.get("suggested_questions", [])
                },
                "timestamp": end_time.isoformat()
            }
            
            print(f"Complete Pipeline completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            error_msg = f"Complete pipeline failed: {str(e)}"
            print(f"{error_msg}")
            
            return {
                "endpoint": "complete_pipeline",
                "success": False,
                "error": error_msg,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }

    # =========================================================================
    # ENDPOINT 8: CHATBOT RAG (retrieve → rerank → LLM)
    # =========================================================================
    def chatbot_endpoint(self, question: str, user_name: str = "", user_birthday: str = "", username: str = "", trading_data: bool = False, language: str = "vi") -> Dict[str, Any]:
        """
        Endpoint: RAG chatbot using chatbot.py pipeline with intelligent user context analysis.
        
        Args:
            question: User question
            user_name: User's full name (optional)
            user_birthday: User's birthday in DD/MM/YYYY format (optional)
            username: Username in app (optional)
            trading_data: Whether user has trading data (boolean)
            language: Language (default "vi")
            
        Returns:
            Dict containing chatbot response with user context analysis
        """
        try:
            from chatbot import build_chatbot
            from module.rag_orchestrator import RAGOrchestratorFactory
            
            if self._chatbot is None:
                orch = RAGOrchestratorFactory.create_optimal_orchestrator()
                self._chatbot = build_chatbot(orch)
            
            # Call chatbot with all user context parameters
            result = self._chatbot.answer(
                question=question,
                user_name=user_name,
                user_birthday=user_birthday,
                username=username,
                trading_data=trading_data,
                language=language
            )
            
            # Extract user context for response
            user_context = result.get("user_context", {})
            suggestions = result.get("suggestions", "")
            
            return {
                "endpoint": "chatbot",
                "success": True,
                "question": question,
                "user_context": {
                    "is_logged_in": user_context.get("is_logged_in", False),
                    "has_personal_info": user_context.get("has_personal_info", False),
                    "has_trading_info": user_context.get("has_trading_info", False),
                    "question_type": user_context.get("question_type", "general"),
                    "info_sufficient": user_context.get("info_sufficient", True),
                    "missing_info": user_context.get("missing_info", [])
                },
                "answer": result.get("answer", ""),
                "suggestions": suggestions,
                "retrieved": result.get("retrieved", 0),
                "reranked": result.get("reranked", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "endpoint": "chatbot",
                "success": False,
                "error": str(e),
                "question": question,
                "user_context": {
                    "is_logged_in": bool(username),
                    "has_personal_info": bool(user_name and user_birthday),
                    "has_trading_info": trading_data,
                    "question_type": "unknown",
                    "info_sufficient": False,
                    "missing_info": ["system_error"]
                },
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# ENDPOINT 7: MEMORY MANAGEMENT
# ============================================================================

    def memory_management_endpoint(
        self,
        action: str,
        user_name: str,
        birthday: str,
        username: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Endpoint 7: Manage memory cache
        
        Args:
            action: Action ('get_status', 'clear', 'get_summary')
            user_name: User name
            birthday: Birthday
            username: Username
            **kwargs: Other parameters depending on action
            
        Returns:
            Dict containing memory management result
        """
        
        try:
            print(f"Memory Management Endpoint - Action: {action}")
            
            user_uuid = self._generate_user_uuid(user_name, birthday, username)
            
            if action == "get_status":
                result = self.memory_agent.get_memory_summary(user_uuid)
                return {
                    "endpoint": "memory_management",
                    "action": action,
                    "success": True,
                    "user_uuid": user_uuid,
                    "memory_status": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            elif action == "clear":
                self.memory_agent.clear_user_memory(user_uuid)
                return {
                    "endpoint": "memory_management",
                    "action": action,
                    "success": True,
                    "user_uuid": user_uuid,
                    "message": f"Memory cleared for user {user_uuid}",
                    "timestamp": datetime.now().isoformat()
                }
                
            elif action == "get_summary":
                result = self.memory_agent.get_memory_summary(user_uuid)
                return {
                    "endpoint": "memory_management",
                    "action": action,
                    "success": True,
                    "user_uuid": user_uuid,
                    "memory_summary": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            else:
                return {
                    "endpoint": "memory_management",
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            error_msg = f"Memory management failed: {str(e)}"
            print(f"{error_msg}")
            return {
                "endpoint": "memory_management",
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

    def memory_update_endpoint(
        self,
        action: str,
        user_name: str,
        birthday: str,
        username: str,
        entry_key: Optional[str] = None,
        entry_data: Optional[Dict[str, Any]] = None,
        entries: Optional[List[Dict[str, Any]]] = None,
        language: str = "vi"
    ) -> Dict[str, Any]:
        """
        Update memory cache for specific user
        
        Supported actions:
        - add_entry: Add new entry
        - update_entry: Update existing entry
        - bulk_update: Update multiple entries at once
        - remove_entry: Remove specific entry
        """
        try:
            # Generate user UUID
            user_uuid = self._generate_user_uuid(user_name, birthday, username)
            
            if action == "add_entry":
                if not entry_data:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "entry_data is required for add_entry action",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Validate entry_data
                if "key" not in entry_data:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "entry_data must contain 'key' field",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Add single entry
                success = self.memory_agent.add_memory_entry(
                    user_uuid=user_uuid,
                    entry_data=entry_data,
                    language=language
                )
                
                if success:
                    return {
                        "endpoint": "memory_update",
                        "success": True,
                        "action": "add_entry",
                        "user_uuid": user_uuid,
                        "message": "Entry added successfully",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "Failed to add entry",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            elif action == "update_entry":
                if not entry_key or not entry_data:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "entry_key and entry_data are required for update_entry action",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Update existing entry
                success = self.memory_agent.update_memory_entry(
                    user_uuid=user_uuid,
                    entry_key=entry_key,
                    entry_data=entry_data,
                    language=language
                )
                
                if success:
                    return {
                        "endpoint": "memory_update",
                        "success": True,
                        "action": "update_entry",
                        "user_uuid": user_uuid,
                        "entry_key": entry_key,
                        "message": "Entry updated successfully",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "Failed to update entry",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            elif action == "bulk_update":
                if not entries:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "entries is required for bulk_update action",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Validate entries
                for entry in entries:
                    if "key" not in entry:
                        return {
                            "endpoint": "memory_update",
                            "success": False,
                            "error": "All entries must contain 'key' field",
                            "timestamp": datetime.now().isoformat()
                        }
                
                # Bulk update entries
                success_count = 0
                for entry in entries:
                    if self.memory_agent.add_memory_entry(
                        user_uuid=user_uuid,
                        entry_data=entry,
                        language=language
                    ):
                        success_count += 1
                
                return {
                    "endpoint": "memory_update",
                    "success": True,
                    "action": "bulk_update",
                    "user_uuid": user_uuid,
                    "entries_count": len(entries),
                    "success_count": success_count,
                    "message": f"Bulk updated {success_count}/{len(entries)} entries successfully",
                    "timestamp": datetime.now().isoformat()
                }
                
            elif action == "remove_entry":
                if not entry_key:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "entry_key is required for remove_entry action",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Remove entry
                success = self.memory_agent.remove_memory_entry(
                    user_uuid=user_uuid,
                    entry_key=entry_key
                )
                
                if success:
                    return {
                        "endpoint": "memory_update",
                        "success": True,
                        "action": "remove_entry",
                        "user_uuid": user_uuid,
                        "entry_key": entry_key,
                        "message": "Entry removed successfully",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "endpoint": "memory_update",
                        "success": False,
                        "error": "Failed to remove entry",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            else:
                return {
                    "endpoint": "memory_update",
                    "success": False,
                    "error": f"Unsupported action: {action}. Supported actions: add_entry, update_entry, bulk_update, remove_entry",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "endpoint": "memory_update",
                "success": False,
                "error": f"Memory update failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def build_lumir_api_endpoints() -> LUMIRAPIEndpoints:
    """
    Factory function to create LUMIR API Endpoints
    
    Returns:
        LUMIRAPIEndpoints instance
    """
    return LUMIRAPIEndpoints()


# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("LUMIR-AI API Endpoints Demo")
    print("=" * 80)
    
    # Initialize API endpoints
    api = build_lumir_api_endpoints()
    
    # Demo endpoints
    print("\nAvailable Endpoints:")
    print("1. memory_check_endpoint() - Check memory cache")
    print("2. question_decomposition_endpoint() - Analyze question")
    print("3. numerology_endpoint() - Analyze numerology")
    print("4. trading_endpoint() - Analyze trading data")
    print("5. lumir_synthesis_endpoint() - Synthesize LUMIR-AI")
    print("6. complete_pipeline_endpoint() - Complete pipeline")
    print("7. memory_management_endpoint() - Manage memory")
    
    print("\nUsage Examples:")
    print("api.memory_check_endpoint(question, user_name, birthday, username, language)")
    print("api.complete_pipeline_endpoint(question, user_name, birthday, excel_path, language, username)")
    
    print("\nAPI Endpoints ready for use!")
