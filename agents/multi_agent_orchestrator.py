import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from agents.question_decomposition_agent import build_question_decomposition_agent
from agents.numerology_agent import build_numerology_agent
from agents.trading_agent import build_trading_agent
from agents.lumir_synthesis_agent import build_lumir_with_memory
from agents.memory_agent import build_memory_agent

# Import LUMIRChatbot for general questions
from chatbot import build_chatbot
from module.rag_orchestrator import RAGOrchestratorFactory


class MultiAgentOrchestrator:
    """
    Multi-Agent Orchestrator - Điều phối hệ thống LUMIR-AI thông minh
    
    Hệ thống bao gồm:
    1. Question Decomposition Agent - Phân tích câu hỏi thông minh
    2. Intelligent Routing - Quyết định có gọi agent hay không
    3. LUMIR-AI Synthesis Agent - Tổng hợp và trả lời tự nhiên
    4. Multi-turn Memory - Ghi nhớ context hội thoại
    """
    
    def __init__(self):
        self.question_decomposer = build_question_decomposition_agent()
        self.numerology_agent = build_numerology_agent()
        self.trading_agent = build_trading_agent()
        self.lumir_agent = build_lumir_with_memory()
        self.memory_agent = build_memory_agent()
        
        # Khởi tạo LUMIRChatbot cho general questions
        try:
            self.rag_orchestrator = RAGOrchestratorFactory.create_optimal_orchestrator()
            self.lumir_chatbot = build_chatbot(self.rag_orchestrator)
            print("✅ LUMIRChatbot initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: LUMIRChatbot initialization failed: {e}")
            self.lumir_chatbot = None
        
        # Memory cho multi-turn chat
        self.conversation_history = []
        
    def process_user_question(
        self,
        question: str,
        user_name: Optional[str] = None,
        birthday: Optional[str] = None,
        excel_path: Optional[str] = None,
        language: str = "vi",
        username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Xử lý câu hỏi của user qua hệ thống multi-agent thông minh
        
        Args:
            question: Câu hỏi của user
            user_name: Tên user
            birthday: Ngày sinh
            excel_path: Đường dẫn file Excel
            language: Ngôn ngữ
            username: Username
            
        Returns:
            Dict chứa kết quả xử lý
        """
        
        start_time = datetime.now()
        
        try:
            print(f"🚀 Bắt đầu xử lý câu hỏi: {question}")
            
            # Kiểm tra memory cache trước
            user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
            memory_query = self.memory_agent.query_memory(
                user_uuid, question, self.conversation_history, language
            )
            
            if memory_query.can_answer and memory_query.confidence > 0.7:
                print(f"✅ Trả lời từ cache (confidence: {memory_query.confidence:.2f})")
                
                # Cập nhật conversation history
                self._update_conversation_history(question, memory_query.suggested_response, user_name, birthday, username, language)
                
                return {
                    "success": True,
                    "question": question,
                    "response": memory_query.suggested_response,
                    "processing_time": 0.1,  # Cache response nhanh
                    "question_type": "cached_response",
                    "decomposition_result": {"source": "memory_cache"},
                    "context_summary": {
                        "numerology_available": False,
                        "trading_available": False,
                        "response_type": "cached",
                        "needs_user_info": False,
                        "suggested_questions": []
                    },
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": True,
                    "cache_confidence": memory_query.confidence
                }
            else:
                print(f"❌ Cache không đủ thông tin (confidence: {memory_query.confidence:.2f})")
            
            # Bước 1: Phân tích câu hỏi một cách thông minh
            print("🔍 Bước 1: Phân tích câu hỏi thông minh...")
            decomposition_result = self.question_decomposer(
                question=question,
                user_name=user_name,
                birthday=birthday,
                excel_path=excel_path,
                language=language,
                username=username,
                conversation_history=self.conversation_history
            )
            
            print(f"✅ Phân tích hoàn thành: {decomposition_result}")
            
            # Bước 2: Quyết định routing thông minh
            question_type = decomposition_result.get("question_type", "general_chat")
            should_call_agents = decomposition_result.get("should_call_agents", False)
            
            print(f"🎯 Loại câu hỏi: {question_type}")
            print(f"🔀 Có gọi agent: {should_call_agents}")
            
            # Bước 3: Xử lý theo loại câu hỏi
            if should_call_agents:
                # Gọi các agent chuyên biệt
                print("🔄 Bước 3: Gọi các agent chuyên biệt...")
                numerology_context, trading_context = self._execute_specialized_agents(
                    decomposition_result, user_name, birthday, excel_path, language
                )
            else:
                # Không gọi agent - xử lý trực tiếp
                print("💬 Bước 3: Xử lý trực tiếp (không gọi agent)...")
                numerology_context = ""
                trading_context = ""
                
                # Xử lý general_chat với LUMIRChatbot nếu có
                if question_type == "general_chat" and self.lumir_chatbot:
                    print("🤖 Sử dụng LUMIRChatbot cho general question...")
                    try:
                        chatbot_result = self.lumir_chatbot.answer(question)
                        if chatbot_result.get("success"):
                            # Trả lời trực tiếp từ chatbot, không cần synthesis
                            chatbot_response = chatbot_result.get("answer", "")
                            
                            # Cập nhật conversation history
                            self._update_conversation_history(question, chatbot_response, user_name, birthday, username, language)
                            
                            # Tạo kết quả cuối cùng
                            end_time = datetime.now()
                            processing_time = (end_time - start_time).total_seconds()
                            
                            result = {
                                "success": True,
                                "question": question,
                                "response": chatbot_response,
                                "processing_time": processing_time,
                                "question_type": question_type,
                                "decomposition_result": decomposition_result,
                                "context_summary": {
                                    "numerology_available": False,
                                    "trading_available": False,
                                    "response_type": "general_chat_direct",
                                    "needs_user_info": False,
                                    "suggested_questions": [],
                                    "source": "lumir_chatbot"
                                },
                                "timestamp": end_time.isoformat(),
                                "chatbot_result": chatbot_result
                            }
                            
                            print(f"✅ Xử lý general_chat hoàn thành trong {processing_time:.2f}s")
                            return result
                        else:
                            print(f"⚠️ LUMIRChatbot không thể trả lời: {chatbot_result.get('reason', 'unknown')}")
                            # Fallback to normal synthesis flow
                    except Exception as e:
                        print(f"❌ LUMIRChatbot error: {e}")
                        # Fallback to normal synthesis flow
            
            # Bước 4: Tổng hợp với LUMIR-AI
            print("🤖 Bước 4: Tổng hợp với LUMIR-AI...")
            
            lumir_response = self.lumir_agent(
                question=question,
                question_type=question_type,
                numerology_context=numerology_context,
                trading_context=trading_context,
                user_name=user_name,
                username=username,
                language=language,
                has_trading_data=decomposition_result.get("has_valid_trading_data", False),
                focus_areas=decomposition_result.get("focus_areas", []),
                needs_user_info=decomposition_result.get("needs_user_info", False),
                suggested_questions=decomposition_result.get("suggested_questions", []),
                conversation_history=self.conversation_history
            )
            
            # Cập nhật conversation history
            self._update_conversation_history(question, lumir_response, user_name, birthday, username, language)
            
            # Tạo kết quả cuối cùng
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "question": question,
                "response": lumir_response,
                "processing_time": processing_time,
                "question_type": question_type,
                "decomposition_result": decomposition_result,
                "context_summary": {
                    "numerology_available": bool(numerology_context),
                    "trading_available": bool(trading_context),
                    "response_type": self._determine_response_type(question_type, numerology_context, trading_context),
                    "needs_user_info": decomposition_result.get("needs_user_info", False),
                    "suggested_questions": decomposition_result.get("suggested_questions", [])
                },
                "timestamp": end_time.isoformat()
            }
            
            print(f"✅ Xử lý hoàn thành trong {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Lỗi xử lý: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Fallback response
            fallback_response = self._create_fallback_response(
                question, user_name, username, language
            )
            
            return {
                "success": False,
                "question": question,
                "response": fallback_response,
                "error": error_msg,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_specialized_agents(
        self, 
        decomposition_result: Dict[str, Any], 
        user_name: str, 
        birthday: str, 
        excel_path: str,
        language: str
    ) -> tuple[str, str]:
        """Thực thi các agent chuyên biệt song song"""
        
        numerology_context = ""
        trading_context = ""
        
        # Tạo tasks cho parallel execution
        tasks_to_run = []
        
        # Numerology task
        numerology_question = decomposition_result.get("numerology_question")
        if numerology_question and numerology_question.strip() and user_name and birthday:
            tasks_to_run.append(
                ("numerology", self._execute_numerology_agent, 
                 numerology_question, user_name, birthday, language)
            )
        
        # Trading task
        trading_question = decomposition_result.get("trading_question")
        if (trading_question and trading_question.strip() and 
            decomposition_result.get("has_valid_trading_data") and excel_path):
            tasks_to_run.append(
                ("trading", self._execute_trading_agent, 
                 trading_question, excel_path, language)
            )
        
        # Thực thi song song
        if tasks_to_run:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_task_name = {}
                for task_name, func, *args in tasks_to_run:
                    future = executor.submit(func, *args)
                    future_to_task_name[future] = task_name
                
                for future in concurrent.futures.as_completed(future_to_task_name):
                    task_name = future_to_task_name[future]
                    try:
                        result = future.result()
                        if task_name == "numerology":
                            numerology_context = result
                        elif task_name == "trading":
                            trading_context = result
                    except Exception as e:
                        print(f"❌ {task_name} agent failed: {e}")
        
        return numerology_context, trading_context
    
    def _execute_numerology_agent(self, question: str, user_name: str, birthday: str, language: str = "vi") -> str:
        """Thực thi numerology agent"""
        try:
            inputs = {
                "question": question,
                "user_name": user_name,
                "birthday": birthday,
                "language": language
            }
            result = self.numerology_agent.invoke(inputs)
            return str(result) if result else ""
        except Exception as e:
            print(f"❌ Numerology agent error: {e}")
            return ""
    
    def _execute_trading_agent(self, question: str, excel_path: str, language: str = "vi") -> str:
        """Thực thi trading agent"""
        try:
            inputs = {
                "question": question,
                "excel_path": excel_path,
                "language": language
            }
            result = self.trading_agent.invoke(inputs)
            return str(result) if result else ""
        except Exception as e:
            print(f"❌ Trading agent error: {e}")
            return ""
    
    def _update_conversation_history(self, question: str, response: str, user_name: str = None, birthday: str = None, username: str = None, language: str = "vi"):
        """Cập nhật conversation history và memory cache"""
        turn_info = {
            "user_question": question,
            "lumir_response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(turn_info)
        
        # Giới hạn history để tránh quá tải
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Cập nhật memory cache nếu có thông tin user
        if user_name and birthday and username:
            try:
                user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
                turn_number = len(self.conversation_history)
                
                # Cập nhật memory cache
                self.memory_agent.update_memory(
                    user_uuid, question, response, turn_number, language
                )
                
                print(f"🧠 Memory cache updated for user {user_uuid}")
            except Exception as e:
                print(f"❌ Error updating memory cache: {e}")
    
    def _determine_response_type(self, question_type: str, numerology_context: str, trading_context: str) -> str:
        """Xác định loại response dựa trên question_type"""
        if question_type == "general_chat":
            return "general_chat"
        elif question_type == "needs_more_info":
            return "needs_more_info"
        elif numerology_context and trading_context:
            return "comprehensive"
        elif numerology_context:
            return "numerology_only"
        elif trading_context:
            return "trading_only"
        else:
            return "general"
    
    def _create_fallback_response(self, question: str, user_name: str, username: str, language: str) -> str:
        """Tạo fallback response khi có lỗi"""
        
        if language == "en":
            return f"""I apologize, {username}. I'm experiencing some technical difficulties right now.

**General Advice:**
• Focus on **risk management** in trading
• Maintain **emotional control** during market fluctuations
• Continue **learning and improving** your skills

**For Personalized Advice:**
Please try again later or contact support for assistance. I'm here to help you with your trading journey!"""
        else:
            return f"""Xin lỗi {username}, tôi đang gặp một số vấn đề kỹ thuật.

**Lời khuyên chung:**
• Tập trung vào **quản lý rủi ro** khi giao dịch
• Duy trì **kiểm soát cảm xúc** trong biến động thị trường
• Tiếp tục **học hỏi và cải thiện** kỹ năng

**Để có tư vấn cá nhân hóa:**
Vui lòng thử lại sau hoặc liên hệ hỗ trợ để được trợ giúp. Tôi luôn sẵn sàng hỗ trợ bạn trong hành trình trading!"""
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Lấy conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Xóa conversation history"""
        self.conversation_history = []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hệ thống"""
        return {
            "status": "operational",
            "agents": {
                "question_decomposer": "ready",
                "numerology_agent": "ready", 
                "trading_agent": "ready",
                "lumir_agent": "ready",
                "memory_agent": "ready"
            },
            "conversation_history_length": len(self.conversation_history),
            "last_activity": datetime.now().isoformat()
        }
    
    def get_memory_status(self, user_name: str, birthday: str, username: str) -> Dict[str, Any]:
        """Lấy trạng thái memory cache của user"""
        if not all([user_name, birthday, username]):
            return {"error": "Missing user information"}
        
        try:
            user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
            return self.memory_agent.get_memory_summary(user_uuid)
        except Exception as e:
            return {"error": str(e)}
    
    def clear_user_memory(self, user_name: str, birthday: str, username: str) -> Dict[str, Any]:
        """Xóa memory cache của user"""
        if not all([user_name, birthday, username]):
            return {"error": "Missing user information"}
        
        try:
            user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
            self.memory_agent.clear_user_memory(user_uuid)
            return {"success": True, "message": f"Memory cleared for user {user_uuid}"}
        except Exception as e:
            return {"error": str(e)}


# Factory function để tạo orchestrator
def build_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """
    Build multi-agent orchestrator
    
    Returns:
        MultiAgentOrchestrator instance
    """
    return MultiAgentOrchestrator()
