from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import get_openai_llm
from tools.data_validator_tool import DataValidator


class QuestionDecomposition(BaseModel):
    """Result of analysis and create sub-question"""
    question_type: str = Field(description="Question type: 'trading_related', 'tbi_related', 'general_chat', 'needs_more_info'")
    should_call_agents: bool = Field(description="Should call specialized agents")
    # numerology_question: Optional[str] = Field(description="Question for numerology agent (null if not needed)")
    tbi_question: Optional[str] = Field(description="Question for TBI agent (null if not needed)")
    trading_question: Optional[str] = Field(description="Question for trading agent (null if not needed)")
    reasoning: str = Field(description="Reasoning for analysis and routing")
    focus_areas: List[str] = Field(description="Focus areas if any")
    needs_user_info: bool = Field(description="Whether to ask for more user info")
    suggested_questions: List[str] = Field(description="Suggested questions to ask user if needed")


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
    Analyze and decompose question in a smart and natural way
    
    Args:
        question: Original user question
        user_name: User name
        birthday: Birthday
        excel_path: Path to Excel file containing trading data
        language: Language (vi, en, etc.)
        username: Username to call name
        conversation_history: Conversation history for context
        
    Returns:
        Dict containing smart analysis result
        "question_type": Type of question
        "should_call_agents": Whether to call specialized agents
        "tbi_question": Question for tbi agent
        "trading_question": Question for trading agent
        "reasoning": Reasoning for analysis and routing
        "focus_areas": Focus areas if any
        "needs_user_info": Whether to ask for more user info
        "suggested_questions": Suggested questions to ask user if needed
        "has_valid_trading_data": Whether trading data is valid
    """
    
    try:
        # Validate trading data if exists
        has_valid_trading_data = False
        if excel_path and excel_path.strip():
            validator = DataValidator()
            import os
            if os.path.exists(excel_path):
                try:
                    import pandas as pd
                    df = pd.read_excel(excel_path)
                    validation = validator.validate_excel_dataframe(df)
                    has_valid_trading_data = validation["is_valid"]
                except Exception as e:
                    print(f"Error validating trading data: {e}")
                    has_valid_trading_data = False
            else:
                # File does not exist
                has_valid_trading_data = False
        else:
            # No file path provided
            has_valid_trading_data = False
        
        # Ensure has_valid_trading_data is always a boolean
        has_valid_trading_data = bool(has_valid_trading_data)
        
        # Check if user is logged in
        user_logged_in = bool(user_name and birthday and username)
        
        # Use LLM to analyze smartly
        llm = get_openai_llm()
        parser = JsonOutputParser(pydantic_object=QuestionDecomposition)
        
        # Create context from conversation history
        context_from_history = ""
        if conversation_history:
            recent_turns = conversation_history[-3:]  # Last 3 turns
            context_parts = []
            for turn in recent_turns:
                user_q = turn.get("user_question", "")
                if user_q:
                    context_parts.append(f"User: {user_q}")
            if context_parts:
                context_from_history = "\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", _read_prompt()),
            ("human", """
## ANALYSIS REQUEST

**Original Question**: "{question}"

**User Context**:
- Name: {user_name}
- Birthday: {birthday}
- Excel File: {excel_path}
- Valid Trading Data: {has_valid_trading_data}
- Logged In: {user_logged_in} 
- Language: {language}

**Conversation History**:
{context_from_history}

---

## ANALYSIS FRAMEWORK

### 1. Smart question decompose
- Create specific sub-questions for relevant agents that match which the original question intent
- Ensure sub-questions are clear, concise, and contextually relevant
-  ALWAYS make the most helpul sub-questions in order to help user get the best answer

### 2. Intent Analysis
**Question Classification Examples**:
- "Hello" → `general_chat` (greeting/social)
- "What can you help with?" → `lumir_related` (system inquiry)
- "I feel anxious when trading" → `tbi_related` (psychology/emotions)
- "Should I buy stocks now?" → `trading_related` (market/strategy)
- "I'm confused" → `needs_more_info` (unclear intent)

### 3. Smart Routing Decision
Based on your analysis:
- Determine the most appropriate question category
- Generate specific, actionable sub-questions for relevant agents
- Consider whether multiple agents should be engaged
- Identify any missing information needed
- Choose task type wisely based on question and user context

### 4. For Ambiguous Emotional Expressions:
**ALWAYS classify as `needs_more_info`** until you can confirm:
- Is this related to trading psychology? → `tbi_related`
- Is this about trading performance? → `trading_related`
- Is this about Lumir's capabilities? → `lumir_related` 
- Is this completely unrelated? → `general_chat`
**If a question can be misunderstood between `trading_related` `tbi_related` or `lumir_related` it should also be put in the `needs_user_info` type**
Example: - "How can you help me to improve my trading performance?" -> Does user asked about Lumir ability or want trading agent and tbi agent analyze thier trading data?

**Your task**: Analyze the user's question intelligently and provide precise routing with well-crafted sub-questions that will help deliver the most valuable response to the user.
## IMPORTANT: QUESTION TYPE MUST BE ONE OF: `trading_related`, `tbi_related`, `general_chat`, `needs_more_info`, `lumir_related` dont use any other type.
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
            "user_logged_in": user_logged_in,
            "context_from_history": context_from_history or "Không có"
        })
        
        # Update has_valid_trading_data from actual validation
        result["has_valid_trading_data"] = has_valid_trading_data
        
        print(f"Question Decomposition Result: {result}")
        return result
        
    except Exception as e:
        print(f"Question decomposition failed: {e}")
        # Fallback analysis - return general chat if user is not logged in
        fallback_type = "general_chat" if not all([user_name, birthday, username]) else "needs_more_info"
        return {
            "question_type": fallback_type,
            "should_call_agents": False,
            # "numerology_question": None,
            "tbi_question": None,
            "trading_question": None,
            "reasoning": f"Fallback analysis due to error: {str(e)}",
            "focus_areas": [],
            "needs_user_info": not all([user_name, birthday, username]),
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
