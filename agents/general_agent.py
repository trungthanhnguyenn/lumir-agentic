from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from config import get_openai_llm
from tools.data_validator_tool import DataValidator
from pathlib import Path

def _read_prompt() -> str:
    """Read the prompt from the user"""
    base_dir = Path(__file__).resolve().parents[1]
    prompt_path = base_dir / "prompts" / "general_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")

def answer(question: str, language: str = "vi", user_name: str = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Answer general questions"""
    if language == "en":
        language = "English"
    elif language == "vi":
        language = "Vietnamese"
    try:
        llm = get_openai_llm()
        prompt_template = _read_prompt()

        # Create context from conversation history
        context_from_history = ""
        if conversation_history:
            recent_turns = conversation_history[-5:]  # Last 5 turns
            context_parts = []
            for turn in recent_turns:
                role = turn.get("role", "")
                content = turn.get("content", "")

                if role and content:
                    if role == "user":
                        context_parts.append(f"User: {content}")
                    elif role == "assistant":
                        context_parts.append(f"Lumir: {content}")
            if context_parts:
                context_from_history = "\n".join(context_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("user", """
    ## FOLLOW THESE INSTRUCTIONS CAREFULLY
    - You are a helpful assistant that always provides accurate and useful information.
    - You must obey the instructions in the question below and respond in {language}.

    Context from recent conversation history (if any):
    {context_from_history}

    User name: {user_name}
    Question: {question}
    """)
        ])

        chain = prompt | llm
        response = chain.invoke({"question": question, "user_name": user_name or "User", "language": language, "context_from_history": context_from_history})
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def build_general_agent(question: str, language: str, user_name: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build the general agent"""
    return answer(question, language, user_name, conversation_history=conversation_history)
