import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from agents.question_decomposition_agent import build_question_decomposition_agent
# from agents.numerology_agent import build_numerology_agent
from agents.tbi_agent import build_tbi_agent
from agents.trading_agent import build_trading_agent
from agents.lumir_synthesis_agent import build_lumir_with_memory
from agents.memory_agent import build_memory_agent

# Import LUMIRChatbot for general questions
from chatbot import build_chatbot
from module.rag_orchestrator import RAGOrchestratorFactory


class MultiAgentOrchestrator:
    """
    Multi-Agent Orchestrator - Orchestrate LUMIR-AI smart multi-agent
    
    System includes:
    1. Question Decomposition Agent - Smart question decomposition
    2. Intelligent Routing - Determine if agent should be called
    3. LUMIR-AI Synthesis Agent - Synthesize and answer naturally
    4. Multi-turn Memory - Remember conversation context
    """
    
    def __init__(self):
        self.question_decomposer = build_question_decomposition_agent()
        # self.numerology_agent = build_numerology_agent()
        self.tbi_agent = build_tbi_agent()
        self.trading_agent = build_trading_agent()
        self.lumir_agent = build_lumir_with_memory()
        self.memory_agent = build_memory_agent()
        
        # Initialize LUMIRChatbot for general questions
        try:
            self.rag_orchestrator = RAGOrchestratorFactory.create_optimal_orchestrator()
            self.lumir_chatbot = build_chatbot(self.rag_orchestrator)
            print("LUMIRChatbot initialized successfully")
        except Exception as e:
            print(f"Warning: LUMIRChatbot initialization failed: {e}")
            self.lumir_chatbot = None
        
        # Memory for multi-turn chat
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
        Process user question through smart multi-agent system
        
        Args:
            question: User question
            user_name: User name
            birthday: Birthday
            excel_path: Excel file path
            language: Language
            username: Username
            
        Returns:
            Dict containing the result of processing
        """
        
        start_time = datetime.now()
        
        try:
            print(f"Start processing question: {question}")
            
            # Check memory cache first
            user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
            memory_query = self.memory_agent.query_memory(
                user_uuid, question, self.conversation_history, language
            )
            
            if memory_query.can_answer and memory_query.confidence > 0.7:
                print(f"Answer from cache (confidence: {memory_query.confidence:.2f})")
                
                # Update conversation history
                self._update_conversation_history(question, memory_query.suggested_response, user_name, birthday, username, language)
                
                return {
                    "success": True,
                    "question": question,
                    "response": memory_query.suggested_response,
                    "processing_time": 0.1,  # Cache response quickly
                    "question_type": "cached_response",
                    "decomposition_result": {"source": "memory_cache"},
                    "context_summary": {
                        "tbi_available": False,
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
                print(f"Cache không đủ thông tin (confidence: {memory_query.confidence:.2f})")
            
            # Bước 1: Smart question decomposition
            print("Step 1: Smart question decomposition...")
            decomposition_result = self.question_decomposer(
                question=question,
                user_name=user_name,
                birthday=birthday,
                excel_path=excel_path,
                language=language,
                username=username,
                conversation_history=self.conversation_history
            )
            
            print(f"Analysis completed: {decomposition_result}")
            
            # Step 2: Smart routing decision
            question_type = decomposition_result.get("question_type", "general_chat")
            should_call_agents = decomposition_result.get("should_call_agents", False)
            
            print(f"Question type: {question_type}")
            print(f"Call agent: {should_call_agents}")
            
            # Bước 3: Xử lý theo loại câu hỏi
            if should_call_agents:
                # Call specialized agents
                print("Step 3: Call specialized agents...")
                tbi_context, trading_context = self._execute_specialized_agents(
                    decomposition_result, user_name, birthday, excel_path, language
                )
            else:
                # No agent - process directly
                print("Step 3: Process directly (no agent)...")
                # numerology_context = ""
                tbi_context = ""
                trading_context = ""
                
                # Process general_chat with LUMIRChatbot if available
                if question_type == "general_chat" and self.lumir_chatbot:
                    print("Using LUMIRChatbot for general question...")
                    try:
                        chatbot_result = self.lumir_chatbot.answer(question)
                        if chatbot_result.get("success"):
                            # Direct response from chatbot, no synthesis
                            chatbot_response = chatbot_result.get("answer", "")
                            
                            # Update conversation history
                            self._update_conversation_history(question, chatbot_response, user_name, birthday, username, language)
                            
                            # Create final result
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
                                    "tbi_available": False,
                                    "trading_available": False,
                                    "response_type": "general_chat_direct",
                                    "needs_user_info": False,
                                    "suggested_questions": [],
                                    "source": "lumir_chatbot"
                                },
                                "timestamp": end_time.isoformat(),
                                "chatbot_result": chatbot_result
                            }
                            
                            print(f"General_chat completed in {processing_time:.2f}s")
                            return result
                        else:
                            print(f"LUMIRChatbot cannot answer: {chatbot_result.get('reason', 'unknown')}")
                            # Fallback to normal synthesis flow
                    except Exception as e:
                        print(f"LUMIRChatbot error: {e}")
                        # Fallback to normal synthesis flow
            
            # Step 4: Synthesize with LUMIR-AI
            print("Step 4: Synthesize with LUMIR-AI...")
            
            lumir_response = self.lumir_agent(
                question=question,
                question_type=question_type,
                # numerology_context=numerology_context,
                tbi_context=tbi_context,
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
            
            # Update conversation history
            self._update_conversation_history(question, lumir_response, user_name, birthday, username, language)
            
            # Create final result
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
                    # "numerology_available": bool(numerology_context),
                    "tbi_available": bool(tbi_context),
                    "trading_available": bool(trading_context),
                    "response_type": self._determine_response_type(question_type, tbi_context, trading_context),
                    "needs_user_info": decomposition_result.get("needs_user_info", False),
                    "suggested_questions": decomposition_result.get("suggested_questions", [])
                },
                "timestamp": end_time.isoformat()
            }
            
            print(f"Processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(f"{error_msg}")
            
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
        """Execute specialized agents in parallel"""
        
        # numerology_context = ""
        tbi_context = ""
        trading_context = ""
        
        # Create tasks for parallel execution
        tasks_to_run = []
        
        # Numerology task
        # numerology_question = decomposition_result.get("numerology_question")
        # if numerology_question and numerology_question.strip() and user_name and birthday:
        #     tasks_to_run.append(
        #         ("numerology", self._execute_numerology_agent, 
        #          numerology_question, user_name, birthday, language)
        #     )
        
        # TBI task
        tbi_question = decomposition_result.get("tbi_question")
        if tbi_question and tbi_question.strip() and user_name and birthday:
            tasks_to_run.append(
                ("tbi", self._execute_tbi_agent, 
                 tbi_question, user_name, birthday, language)
            )
        
        # Trading task
        trading_question = decomposition_result.get("trading_question")
        if (trading_question and trading_question.strip() and 
            decomposition_result.get("has_valid_trading_data") and excel_path):
            tasks_to_run.append(
                ("trading", self._execute_trading_agent, 
                 trading_question, excel_path, language)
            )
        
        # Execute in parallel
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
                        # if task_name == "numerology":
                        #     numerology_context = result
                        # elif task_name == "tbi":
                        if task_name == "tbi":
                            tbi_context = result
                        elif task_name == "trading":
                            trading_context = result
                    except Exception as e:
                        print(f"{task_name} agent failed: {e}")
        
        return tbi_context, trading_context
    
    # def _execute_numerology_agent(self, question: str, user_name: str, birthday: str, language: str = "vi") -> str:
    #     """Execute numerology agent"""
    #     try:
    #         inputs = {
    #             "question": question,
    #             "user_name": user_name,
    #             "birthday": birthday,
    #             "language": language
    #         }
    #         result = self.numerology_agent.invoke(inputs)
    #         return str(result) if result else ""
    #     except Exception as e:
    #         print(f"Numerology agent error: {e}")
    #         return ""
    
    def _execute_tbi_agent(self, question: str, user_name: str, birthday: str, language: str = "vi") -> str:
        """Execute TBI agent"""
        try:
            inputs = {
                "question": question,
                "user_name": user_name,
                "birthday": birthday,
                "language": language
            }
            result = self.tbi_agent.invoke(inputs)
            return str(result) if result else ""
        except Exception as e:
            print(f"TBI agent error: {e}")
            return ""
    
    def _execute_trading_agent(self, question: str, excel_path: str, language: str = "vi") -> str:
        """Execute trading agent"""
        try:
            inputs = {
                "question": question,
                "excel_path": excel_path,
                "language": language
            }
            result = self.trading_agent.invoke(inputs)
            return str(result) if result else ""
        except Exception as e:
            print(f"Trading agent error: {e}")
            return ""
    
    def _update_conversation_history(self, question: str, response: str, user_name: str = None, birthday: str = None, username: str = None, language: str = "vi"):
        """Update conversation history and memory cache"""
        turn_info = {
            "user_question": question,
            "lumir_response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(turn_info)
        
        # Limit history to avoid overloading
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Update memory cache if user information is available
        if user_name and birthday and username:
            try:
                user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
                turn_number = len(self.conversation_history)
                
                # Update memory cache
                self.memory_agent.update_memory(
                    user_uuid, question, response, turn_number, language
                )
                
                print(f"Memory cache updated for user {user_uuid}")
            except Exception as e:
                print(f"Error updating memory cache: {e}")
    
    def _determine_response_type(self, question_type: str, tbi_context: str, trading_context: str) -> str:
        """Determine response type based on question_type"""
        if question_type == "general_chat":
            return "general_chat"
        elif question_type == "needs_more_info":
            return "needs_more_info"
        elif tbi_context and trading_context:
            return "comprehensive"
        elif tbi_context:
            return "tbi_only"
        elif trading_context:
            return "trading_only"
        else:
            return "general"
    
    def _create_fallback_response(self, question: str, user_name: str, username: str, language: str) -> str:
        """Create fallback response when error occurs"""
        
        if language == "en":
            return f"""I apologize, {username}. I'm experiencing some technical difficulties right now.

**General Advice:**
• Focus on **risk management** in trading
• Maintain **emotional control** during market fluctuations
• Continue **learning and improving** your skills

**For Personalized Advice:**
Please try again later or contact support for assistance. I'm here to help you with your trading journey!"""
        else:
            return f"""I apologize {username}, I'm experiencing some technical issues.

**Lời khuyên chung:**
• Tập trung vào **quản lý rủi ro** khi giao dịch
• Duy trì **kiểm soát cảm xúc** trong biến động thị trường
• Tiếp tục **học hỏi và cải thiện** kỹ năng

**Để có tư vấn cá nhân hóa:**
Vui lòng thử lại sau hoặc liên hệ hỗ trợ để được trợ giúp. Tôi luôn sẵn sàng hỗ trợ bạn trong hành trình trading!"""
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "status": "operational",
            "agents": {
                "question_decomposer": "ready",
                "tbi_agent": "ready", 
                "trading_agent": "ready",
                "lumir_agent": "ready",
                "memory_agent": "ready"
            },
            "conversation_history_length": len(self.conversation_history),
            "last_activity": datetime.now().isoformat()
        }
    
    def get_memory_status(self, user_name: str, birthday: str, username: str) -> Dict[str, Any]:
        """Get memory cache status of user"""
        if not all([user_name, birthday, username]):
            return {"error": "Missing user information"}
        
        try:
            user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
            return self.memory_agent.get_memory_summary(user_uuid)
        except Exception as e:
            return {"error": str(e)}
    
    def clear_user_memory(self, user_name: str, birthday: str, username: str) -> Dict[str, Any]:
        """Clear memory cache of user"""
        if not all([user_name, birthday, username]):
            return {"error": "Missing user information"}
        
        try:
            user_uuid = self.memory_agent._generate_user_uuid(user_name, birthday, username)
            self.memory_agent.clear_user_memory(user_uuid)
            return {"success": True, "message": f"Memory cleared for user {user_uuid}"}
        except Exception as e:
            return {"error": str(e)}


# Factory function to build orchestrator
def build_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """
    Build multi-agent orchestrator
    
    Returns:
        MultiAgentOrchestrator instance
    """
    return MultiAgentOrchestrator()
