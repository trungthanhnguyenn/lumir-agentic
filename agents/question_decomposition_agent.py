from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import get_openai_llm
from tools.data_validator_tool import DataValidator


class QuestionDecomposition(BaseModel):
    """Kết quả phân tích và tạo câu hỏi con"""
    question_type: str = Field(description="Loại câu hỏi: 'trading_related', 'numerology_related', 'general_chat', 'needs_more_info'")
    should_call_agents: bool = Field(description="Có nên gọi các agent chuyên biệt không")
    numerology_question: Optional[str] = Field(description="Câu hỏi cho numerology agent (null nếu không cần)")
    trading_question: Optional[str] = Field(description="Câu hỏi cho trading agent (null nếu không cần)")
    reasoning: str = Field(description="Lý do cho việc phân tích và routing")
    focus_areas: List[str] = Field(description="Các lĩnh vực tập trung nếu có")
    needs_user_info: bool = Field(description="Có cần hỏi thêm thông tin từ user không")
    suggested_questions: List[str] = Field(description="Các câu hỏi gợi ý để hỏi user nếu cần thêm thông tin")


def _read_prompt() -> str:
    base_dir = Path(__file__).resolve().parents[1]
    prompt_path = base_dir / "prompts" / "question_decomposition_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def analyze_and_decompose_question(
    question: str, 
    user_name: Optional[str] = None,
    birthday: Optional[str] = None,
    excel_path: Optional[str] = None,
    language: str = "vi",
    username: Optional[str] = None,
    conversation_history: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Phân tích câu hỏi một cách thông minh và tự nhiên
    
    Args:
        question: Câu hỏi gốc của user
        user_name: Tên user
        birthday: Ngày sinh
        excel_path: Đường dẫn file Excel trading
        language: Ngôn ngữ (vi, en, etc.)
        username: Username để gọi tên
        conversation_history: Lịch sử hội thoại để context
        
    Returns:
        Dict chứa kết quả phân tích thông minh
    """
    
    try:
        # Validate trading data nếu có
        has_valid_trading_data = False
        if excel_path:
            validator = DataValidator()
            import os
            if os.path.exists(excel_path):
                try:
                    import pandas as pd
                    df = pd.read_excel(excel_path)
                    validation = validator.validate_excel_dataframe(df)
                    has_valid_trading_data = validation["is_valid"]
                except Exception:
                    has_valid_trading_data = False
        
        # Sử dụng LLM để phân tích thông minh
        llm = get_openai_llm()
        parser = JsonOutputParser(pydantic_object=QuestionDecomposition)
        
        # Tạo context từ conversation history
        context_from_history = ""
        if conversation_history:
            recent_turns = conversation_history[-3:]  # Lấy 3 turn gần nhất
            context_parts = []
            for turn in recent_turns:
                user_q = turn.get("user_question", "")[:100]  # Giới hạn độ dài
                if user_q:
                    context_parts.append(f"User: {user_q}")
            if context_parts:
                context_from_history = "\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", _read_prompt()),
            ("human", """
Câu hỏi hiện tại: {question}

Thông tin người dùng:
- Tên: {user_name}
- Ngày sinh: {birthday}
- File Excel: {excel_path}
- Có dữ liệu trading hợp lệ: {has_valid_trading_data}
- Ngôn ngữ: {language}
- Username: {username}

Lịch sử hội thoại gần đây:
{context_from_history}

Hãy phân tích một cách THÔNG MINH và TỰ NHIÊN như ChatGPT/Claude:
1. Hiểu ý định thực sự của user
2. Quyết định có cần gọi agent chuyên biệt không
3. Nếu cần thêm thông tin, gợi ý câu hỏi phù hợp
4. Không cứng nhắc, hãy tự nhiên như con người
""")
        ])
        
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "question": question,
            "user_name": user_name or "Không có",
            "birthday": birthday or "Không có",
            "excel_path": excel_path or "Không có",
            "has_valid_trading_data": has_valid_trading_data,
            "language": language,
            "username": username or "Không có",
            "context_from_history": context_from_history or "Không có"
        })
        
        # Cập nhật has_valid_trading_data từ validation thực tế
        result["has_valid_trading_data"] = has_valid_trading_data
        
        print(f"🔍 Question Decomposition Result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Question decomposition failed: {e}")
        # Fallback analysis - trả về general chat
        return {
            "question_type": "general_chat",
            "should_call_agents": False,
            "numerology_question": None,
            "trading_question": None,
            "reasoning": f"Fallback analysis due to error: {str(e)}",
            "focus_areas": [],
            "needs_user_info": False,
            "suggested_questions": [],
            "has_valid_trading_data": False
        }


def build_question_decomposition_agent():
    """
    Build question decomposition agent
    
    Returns:
        Function that can be called with user input
    """
    return analyze_and_decompose_question
