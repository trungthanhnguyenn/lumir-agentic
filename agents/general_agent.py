from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import get_openai_llm
from tools.data_validator_tool import DataValidator
from pathlib import Path

def _read_prompt() -> str:
    """Read the prompt from the user"""
    base_dir = Path(__file__).resolve().parents[1]
    prompt_path = base_dir / "prompts" / "general_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")

def answer(question: str, language: str = "vi", user_name: str = None) -> str:
    """Answer general questions"""
    try:
        llm = get_openai_llm()
        prompt_template = _read_prompt()
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("user", """
    ## FOLLOW THESE INSTRUCTIONS CAREFULLY
    - You are a helpful assistant that always provides accurate and useful information.
    - Answer the question bellow and response in {language}.:

    User name: {user_name}\nQuestion: {question}
    """)
        ])

        chain = prompt | llm
        response = chain.invoke({"question": question, "user_name": user_name or "User", "language": language})
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def build_general_agent(question: str, language: str, user_name: str) -> str:
    """Build the general agent"""
    return answer(question, language, user_name)
