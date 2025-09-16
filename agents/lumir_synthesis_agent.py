from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from config import get_openai_llm


def _read_prompt() -> str:
    base_dir = Path(__file__).resolve().parents[1]
    prompt_path = base_dir / "prompts" / "lumir_synthesis_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def _detect_abnormal_behavior(tbi_context: str, trading_context: str, question: str) -> List[str]:
    """
    Detect abnormal behavior from context
    
    Args:
        tbi_context: Context from tbi agent
        trading_context: Context from trading agent
        question: Original question from user
        
    Returns:
        List of abnormal behaviors detected
    """
    abnormal_behaviors = []
    
    # Extract FOMO from question
    fomo_keywords = ["fomo", "sợ bỏ lỡ", "vào lệnh vội", "chạy theo đám đông", "mua đỉnh"]
    if any(keyword in question.lower() for keyword in fomo_keywords):
        abnormal_behaviors.append("FOMO - Sợ bỏ lỡ cơ hội")
    
    # Extract revenge trading from question
    revenge_keywords = ["trả thù", "gỡ gạc", "lấy lại", "bù lỗ", "revenge"]
    if any(keyword in question.lower() for keyword in revenge_keywords):
        abnormal_behaviors.append("Revenge Trading - Giao dịch trả thù")
    
    # Extract overtrading from trading context
    if trading_context and "rapid_fire" in trading_context.lower():
        abnormal_behaviors.append("Overtrading - Giao dịch quá nhiều")
    
    # Extract emotional trading from question
    emotion_keywords = ["stress", "lo lắng", "sợ hãi", "tham lam", "tức giận"]
    if any(keyword in question.lower() for keyword in emotion_keywords):
        abnormal_behaviors.append("Emotional Trading - Giao dịch theo cảm xúc")
    
    return abnormal_behaviors


def _prepare_synthesis_data(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for synthesis agent
    
    Args:
        input_dict: Input from user
        question: Question
        question_type: Type of question
        tbi_context: Context from tbi agent
        trading_context: Context from trading agent
        user_name: User name
        username: Username
        language: Language
        has_trading_data: Whether trading data is valid
        focus_areas: Focus areas if any
        needs_user_info: Whether to ask for more user info
        suggested_questions: Suggested questions to ask user if needed
        conversation_history: Conversation history for context
        
    Returns:
        Dict containing prepared data
        "question": Question
        "question_type": Type of question
        "tbi_context": Context from tbi agent
        "trading_context": Context from trading agent
        "user_name": User name
        "username": Username
        "language": Language
        "has_trading_data": Whether trading data is valid
        "focus_areas": Focus areas if any
        "abnormal_behaviors": Abnormal behaviors if any
        "response_type": Response type
        "context_summary": Context summary
        "needs_user_info": Whether to ask for more user info
        "suggested_questions": Suggested questions to ask user if needed
        "conversation_history": Conversation history for context
        "timestamp": Timestamp
    """
    question = input_dict.get("question", "")
    question_type = input_dict.get("question_type", "general_chat")
    tbi_context = input_dict.get("tbi_context", "")
    birthday = input_dict.get("birthday", "")
    trading_context = input_dict.get("trading_context", "")
    user_name = input_dict.get("user_name", "")
    username = input_dict.get("username", "")
    language = input_dict.get("language", "vi")
    has_trading_data = input_dict.get("has_trading_data", False)
    focus_areas = input_dict.get("focus_areas", [])
    needs_user_info = input_dict.get("needs_user_info", False)
    suggested_questions = input_dict.get("suggested_questions", [])
    conversation_history = input_dict.get("conversation_history", [])
    response_type = input_dict.get("task")
    reasoning = input_dict.get("reasoning")

    # Detect abnormal behavior
    abnormal_behaviors = _detect_abnormal_behavior(tbi_context, trading_context, question)
    
    # Prepare intelligent context summary based on language
    context_summary = []
    if tbi_context and tbi_context.strip():
        if language == "en":
            context_summary.append("TBI Personality & Psychology Analysis Available")
        else:  # Vietnamese default
            context_summary.append("Có phân tích tính cách và tâm lý TBI")
            
    if trading_context and trading_context.strip():
        if language == "en":
            context_summary.append("Trading Performance Data Available")
        else:  # Vietnamese default
            context_summary.append("Có phân tích dữ liệu giao dịch")

    if question_type == "trading_related" and not has_trading_data:
        if language == "en":
            context_summary.append("No trading data - Cannot provide trading-specific advice")
        else:
            context_summary.append("Chưa có dữ liệu giao dịch - Không thể tư vấn chuyên sâu")

    if question_type == "tbi_related" and (not user_name or not birthday):
        if language == "en":
            context_summary.append("Missing user info - Cannot provide TBI-specific advice")
        else:
            context_summary.append("Chưa có thông tin người dùng - Không thể tư vấn chuyên sâu về cảm xúc hành vi trong giao dịch")
            
    if not context_summary:
        if language == "en":
            context_summary.append("Limited data - General guidance available")
        else:  # Vietnamese default
            context_summary.append("Chưa có dữ liệu phân tích chuyên sâu")
    
    # Format conversation history for display
    conversation_context = ""
    if conversation_history:
        conversation_context = create_memory_context(conversation_history)
    
    return {
        "question": question,
        "question_type": question_type,
        "tbi_context": tbi_context,
        "trading_context": trading_context,
        "user_name": user_name,
        "username": username,
        "language": language,
        "has_trading_data": has_trading_data,
        "focus_areas": focus_areas,
        "abnormal_behaviors": abnormal_behaviors,
        "response_type": response_type,
        "context_summary": context_summary,
        "needs_user_info": needs_user_info,
        "suggested_questions": suggested_questions,
        "conversation_history": conversation_context,
        "analysis_reasoning": reasoning,
        "timestamp": datetime.now().isoformat()
    }


def build_lumir_synthesis_agent():
    """
    Build LUMIR-AI synthesis agent
    
    Returns:
        Chain that can synthesize responses from multiple agents
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", _read_prompt()),
        ("human", """
USER: {username}
QUESTION: {question}
LANGUAGE REQUIRED: {language}

=== CONTEXT ANALYSIS ===
QUESTION TYPE: {question_type}
CONTEXT SUMMARY: {context_summary}
TASK: {response_type}

=== AVAILABLE DATA SOURCES ===

TBI ANALYSIS (Trading Behavioral Index - Personality & Psychology in Trading):
{tbi_context}

TRADING DATA ANALYSIS:
{trading_context}

ABNORMAL BEHAVIORS DETECTED:
{abnormal_behaviors}

CONVERSATION HISTORY:
{conversation_history}

### DECOMPOSITION INSIGHTS
Reasoning: {analysis_reasoning}
Focus Areas: {focus_areas}

=== SYNTHESIS INSTRUCTION ===
Based on the question and available context above:
1. Analyze what the user is really asking for
2. Determine which context sources are relevant to answer
3. Synthesize a natural, helpful response that addresses their specific need
4. Use the specified language: {language}
5. Be intelligent and adaptive - not rigid or mechanical

Remember: You are LUMIR-AI, a smart trading psychology advisor. Use the context intelligently to provide personalized, helpful responses.""")
    ])
    
    llm = get_openai_llm()
    
    # Create the chain
    prepare_data = RunnableLambda(_prepare_synthesis_data)
    
    chain = (
        prepare_data | 
        prompt | 
        llm | 
        StrOutputParser()
    )
    
    return chain


def create_memory_context(conversation_history: List[Dict[str, Any]]) -> str:
    """
    Create context from conversation history for multi-turn chat
    
    Args:
        conversation_history: Conversation history
        
    Returns:
        String containing context from history
    """
    if not conversation_history:
        return ""
    
    context_parts = []
    context_parts.append("LỊCH SỬ HỘI THOẠI:")
    
    for i, turn in enumerate(conversation_history[-5:], 1):  # Get last 5 turns
        user_question = turn.get("user_question", "")
        lumir_response = turn.get("lumir_response", "")
        
        if user_question and lumir_response:
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {user_question}")
            context_parts.append(f"LUMIR: {lumir_response}")
            context_parts.append("")
    
    return "\n".join(context_parts)


def build_lumir_with_memory():
    """
    Build LUMIR-AI agent with memory support for multi-turn conversations
    
    Returns:
        Function that can handle multi-turn conversations
    """
    
    def lumir_with_memory(
        question: str,
        task: str,
        reasoning: str,
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
    ) -> str:
        """
        LUMIR-AI with memory for multi-turn chat
        
        Args:
            question: Current question
            question_type: Question type
            tbi_context: Context from TBI agent
            trading_context: Context from trading agent
            user_name: Tên user
            username: Username
            language: Language
            has_trading_data: Has trading data
            focus_areas: Focus areas
            needs_user_info: Needs user info
            suggested_questions: Suggested questions
            conversation_history: Conversation history
            
        Returns:
            Response from LUMIR-AI
        """
        
        # Create memory context
        memory_context = create_memory_context(conversation_history or [])
        
        # Prepare input
        input_data = {
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
            "context_summary": memory_context,
            "response_type": task,
            "analysis_reasoning": reasoning,
            "conversation_history": conversation_history or []
        }
        
        # Call synthesis agent
        agent = build_lumir_synthesis_agent()
        response = agent.invoke(input_data)
        
        return response
    
    return lumir_with_memory
