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

def _sanitize_numerology_context(numerology_context: str) -> str:
    """
    Remove sensitive numerology terms and transform to safe alternatives
    """
    if not numerology_context:
        return ""
    
    # Dictionary of transformations
    transformations = {
        # Direct term replacements
        "thần số học": "phân tích tính cách",
        "numerology": "phân tích tính cách",
        "con số cá nhân": "đặc điểm cá nhân",
        "số định mệnh": "đặc điểm bẩm sinh",
        "chỉ số": "đặc điểm",
        
        # Pattern replacements for numbered references
        r"ngày cá nhân số \d+": "hôm nay",
        r"đường đời số \d+": "đặc điểm tính cách",
        r"số \d+ trong": "đặc điểm trong",
        
        # Remove specific number references
        r"là số \d+": "có đặc điểm",
        r"thuộc nhóm \d+": "có xu hướng",
    }
    
    sanitized = numerology_context
    for old, new in transformations.items():
        if old.startswith('r"'):  # regex pattern
            import re
            sanitized = re.sub(old[2:-1], new, sanitized)
        else:  # direct replacement
            sanitized = sanitized.replace(old, new)
    
    return sanitized

def _detect_abnormal_behavior(numerology_context: str, trading_context: str, question: str) -> List[str]:
    """
    Detect abnormal behavior from context
    
    Args:
        numerology_context: Context from numerology agent
        trading_context: Context from trading agent
        question: Original question from user
        
    Returns:
        List of abnormal behaviors detected
    """
    abnormal_behaviors = []
    
    # Phát hiện FOMO
    fomo_keywords = ["fomo", "sợ bỏ lỡ", "vào lệnh vội", "chạy theo đám đông", "mua đỉnh"]
    if any(keyword in question.lower() for keyword in fomo_keywords):
        abnormal_behaviors.append("FOMO - Sợ bỏ lỡ cơ hội")
    
    # Phát hiện revenge trading
    revenge_keywords = ["trả thù", "gỡ gạc", "lấy lại", "bù lỗ", "revenge"]
    if any(keyword in question.lower() for keyword in revenge_keywords):
        abnormal_behaviors.append("Revenge Trading - Giao dịch trả thù")
    
    # Phát hiện overtrading
    if trading_context and "rapid_fire" in trading_context.lower():
        abnormal_behaviors.append("Overtrading - Giao dịch quá nhiều")
    
    # Phát hiện emotional trading
    emotion_keywords = ["stress", "lo lắng", "sợ hãi", "tham lam", "tức giận"]
    if any(keyword in question.lower() for keyword in emotion_keywords):
        abnormal_behaviors.append("Emotional Trading - Giao dịch theo cảm xúc")
    
    return abnormal_behaviors


def _prepare_synthesis_data(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn bị dữ liệu cho synthesis agent
    
    Args:
        input_dict: Input từ user
        
    Returns:
        Dict chứa dữ liệu đã chuẩn bị
    """
    question = input_dict.get("question", "")
    question_type = input_dict.get("question_type", "general_chat")
    numerology_context = input_dict.get("numerology_context", "")
    trading_context = input_dict.get("trading_context", "")
    user_name = input_dict.get("user_name", "")
    username = input_dict.get("username", "")
    language = input_dict.get("language", "vi")
    has_trading_data = input_dict.get("has_trading_data", False)
    focus_areas = input_dict.get("focus_areas", [])
    needs_user_info = input_dict.get("needs_user_info", False)
    suggested_questions = input_dict.get("suggested_questions", [])
    conversation_history = input_dict.get("conversation_history", [])
    
    # Detect abnormal behavior
    abnormal_behaviors = _detect_abnormal_behavior(numerology_context, trading_context, question)
    
    # Determine response type
    response_type = "comprehensive"
    if question_type == "general_chat":
        response_type = "general_chat"
    elif question_type == "needs_more_info":
        response_type = "needs_more_info"
    elif not numerology_context and not trading_context:
        response_type = "general"
    elif not trading_context and has_trading_data:
        response_type = "missing_trading_data"
    elif not numerology_context:
        response_type = "missing_numerology_data"
    
    # Prepare context summary
    context_summary = []
    if numerology_context:
        numerology_context = _sanitize_numerology_context(numerology_context)
        context_summary.append(" Có phân tích tính cách và tâm lý")
    if trading_context:
        context_summary.append(" Có phân tích dữ liệu giao dịch")
    if not context_summary:
        context_summary.append(" Chưa có dữ liệu phân tích chuyên sâu")
    
    # Format conversation history for display
    conversation_context = ""
    if conversation_history:
        conversation_context = create_memory_context(conversation_history)
    
    return {
        "question": question,
        "question_type": question_type,
        "numerology_context": numerology_context,
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
Câu hỏi của {username}: {question}

LOẠI CÂU HỎI: {question_type}
TÓM TẮT NGỮ CẢNH:
{context_summary}

THÔNG TIN TÍNH CÁCH VÀ TÂM LÝ CỦA NGƯỜI DÙNG:
- Nếu trong nội dung có nhắc đến các **con số cá nhân** hoặc các **cụm từ liên quan đến thần số học** thì bạn cần tránh nhắc trực tiếp mà sử dụng các thuật ngữ chung thay thế.
{numerology_context}

PHÂN TÍCH DỮ LIỆU GIAO DỊCH:
{trading_context}

HÀNH VI LỆCH CHUẨN PHÁT HIỆN:
{abnormal_behaviors}

LỊCH SỬ HỘI THOẠI:
{conversation_history}

Từ những thông tin trên hãy đưa ra tư vấn hữu ích và thân thiện với vai trò LUMIR-AI:
- Nếu ngữ cảnh có nhắc đến các **con số cá nhân** hoặc các **cụm từ liên quan đến thần số học** thì bạn cần tránh nhắc trực tiếp mà sử dụng các thuật ngữ chung thay thế.
- Nếu không có thông về hành vi lệch chuẩn, hoặc lịch sử hội thoại thì không cần trả về.
""")
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
    context_parts.append("📝 LỊCH SỬ HỘI THOẠI:")
    
    for i, turn in enumerate(conversation_history[-5:], 1):  # Get last 5 turns
        user_question = turn.get("user_question", "")
        lumir_response = turn.get("lumir_response", "")
        
        if user_question and lumir_response:
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {user_question}")
            context_parts.append(f"LUMIR: {lumir_response[:200]}...")  # Limit length
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
        question_type: str = "general_chat",
        numerology_context: str = "",
        trading_context: str = "",
        user_name: str = "",
        username: str = "",
        language: str = "vi",
        has_trading_data: bool = False,
        focus_areas: List[str] = None,
        needs_user_info: bool = False,
        suggested_questions: List[str] = None,
        conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        """
        LUMIR-AI with memory for multi-turn chat
        
        Args:
            question: Current question
            question_type: Question type
            numerology_context: Context from numerology agent
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
            "numerology_context": numerology_context,
            "trading_context": trading_context,
            "user_name": user_name,
            "username": username,
            "language": language,
            "has_trading_data": has_trading_data,
            "focus_areas": focus_areas or [],
            "needs_user_info": needs_user_info,
            "suggested_questions": suggested_questions or [],
            "memory_context": memory_context
        }
        
        # Call synthesis agent
        agent = build_lumir_synthesis_agent()
        response = agent.invoke(input_data)
        
        return response
    
    return lumir_with_memory
