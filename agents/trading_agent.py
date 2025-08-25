from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.trading_tool import (
    calculate_trade_index,
    analyze_trading_data,
    get_trading_data_from_excel,
)
from tools.data_validator_tool import DataValidator

from config import get_openai_llm


def _read_prompt() -> str:
    base_dir = Path(__file__).resolve().parents[1]
    prompt_path = base_dir / "prompts" / "trading_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def _prepare_trading_data(input_dict: dict) -> dict:
    """
    Prepare trading data for analysis based on user input.
    
    Args:
        input_dict: Dictionary containing question and optional excel_path
        
    Returns:
        Dictionary with prepared data for the agent
    """
    question = input_dict.get("question", "")
    excel_path = input_dict.get("excel_path") or input_dict.get("file_path")  # Support both keys
    
    try:
        # Validate profile lightly if provided
        validator = DataValidator()
        if input_dict.get("profile"):
            prof = input_dict["profile"]
            validator.validate_profile(prof.get("user_name"), prof.get("birthday"))

        # Sử dụng trading tool mới để phân tích
        trading_analysis = analyze_trading_data(excel_path, question)
        
        if not trading_analysis.get("success"):
            return {
                "question": question,
                "error": trading_analysis.get("error", "Unknown error"),
                "error_type": "trading_analysis_error",
                "success": False,
            }
        
        # Lấy kết quả từ trading tool
        trading_result = trading_analysis["full_result"]
        analysis_result = trading_analysis["analysis_result"]
        comprehensive_report = trading_analysis["report"]
        
        # Create a comprehensive summary for the agent (structured JSON)
        data_summary = {
            "total_trades": trading_result["trades"],
            "total_profit": trading_result["net_profit"],
            "win_rate": trading_result["win_rate_pct"],
            "avg_profit_per_trade": trading_result["avg_profit_per_trade"],
            "profit_factor": trading_result["profit_factor"],
            "max_drawdown": trading_result["max_drawdown_pct"],
            "max_consecutive_losses": trading_result["max_consecutive_losses"],
            "best_trade": trading_result["best_trade"],
            "worst_trade": trading_result["worst_trade"],
            "available_columns": list(trading_analysis["data"].columns) if hasattr(trading_analysis["data"], 'columns') else [],
            "symbols_traded": list(trading_analysis["data"]['symbol'].unique()) if 'symbol' in trading_analysis["data"].columns else [],
            "time_range": {
                "earliest": trading_analysis["data"]['close_time'].min() if 'close_time' in trading_analysis["data"].columns else None,
                "latest": trading_analysis["data"]['close_time'].max() if 'close_time' in trading_analysis["data"].columns else None
            } if 'close_time' in trading_analysis["data"].columns else None
        }
        
        # Flatten the data structure for the prompt template
        prompt_data = {
            "question": question,
            "total_trades": data_summary["total_trades"],
            "total_profit": data_summary["total_profit"],
            "win_rate": data_summary["win_rate"],
            "avg_profit_per_trade": data_summary["avg_profit_per_trade"],
            "profit_factor": data_summary["profit_factor"],
            "max_drawdown": data_summary["max_drawdown"],
            "max_consecutive_losses": data_summary["max_consecutive_losses"],
            "comprehensive_report": comprehensive_report,
            "trading_data": trading_result,  # structured JSON payload
            "analysis_result": analysis_result,
            "data_summary": data_summary,
            "success": True,
            "language": input_dict.get("language", "vi")  # Thêm language vào prompt data
        }
        
        return prompt_data
        
    except Exception as e:
        return {
            "question": question,
            "error": str(e),
            "error_type": "data_processing_error",
            "success": False,
            "language": input_dict.get("language", "vi"),  # Ensure language is always provided
            # Provide default values for all expected variables
            "total_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_consecutive_losses": 0,
            "comprehensive_report": "Không có dữ liệu để phân tích",
            "trading_data": {},
            "analysis_result": {},
            "data_summary": {}
        }


def build_trading_agent():
    """
    Build a trading agent that can intelligently analyze trading data.
    
    The agent will:
    1. Read Excel files with trading history
    2. Analyze user questions to determine what they want to know
    3. Generate focused reports based on user needs
    4. Handle errors gracefully and explain limitations
    """
    
    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", _read_prompt()),
        ("human", """
Câu hỏi: {question}

Dữ liệu giao dịch:
{data_summary}

Báo cáo chi tiết:
{comprehensive_report}

**QUAN TRỌNG**: Trả lời bằng ngôn ngữ {language}. Nếu language="vi" thì trả lời tiếng Việt, nếu language="en" thì trả lời tiếng Anh.
        """),
    ])
    
    llm = get_openai_llm()
    
    # Create a simple chain
    chain = prompt | llm | StrOutputParser()
    
    # Create a wrapper function that prepares data first
    def trading_agent_wrapper(inputs):
        # Prepare the data first
        prepared_data = _prepare_trading_data(inputs)
        
        # Create a simplified data structure for the prompt
        prompt_data = {
            "question": prepared_data["question"],
            "data_summary": f"""
📊 TÓM TẮT DỮ LIỆU:
- Tổng số lệnh: {prepared_data.get('total_trades', 0)}
- Tổng lợi nhuận: {prepared_data.get('total_profit', 0):,.2f}
- Tỷ lệ thắng: {prepared_data.get('win_rate', 0)}%
- Lợi nhuận trung bình/lệnh: {prepared_data.get('avg_profit_per_trade', 0):,.2f}
- Hệ số lợi nhuận: {prepared_data.get('profit_factor', 0)}
- Sụt giảm tối đa: {prepared_data.get('max_drawdown', 0)}%
- Số lệnh thua liên tiếp tối đa: {prepared_data.get('max_consecutive_losses', 0)}
            """,
            "comprehensive_report": prepared_data.get('comprehensive_report', 'Không có dữ liệu để phân tích'),
            "language": prepared_data.get('language', 'vi')
        }
        
        # Call the chain with prepared data
        result = chain.invoke(prompt_data)
        return result
    
    return trading_agent_wrapper


def build_simple_trading_agent():
    """
    Build a simpler trading agent for basic queries.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", _read_prompt()),
        ("human", "Câu hỏi: {question}\n\nDữ liệu giao dịch: {data}"),
    ])
    
    llm = get_openai_llm()
    return prompt | llm | StrOutputParser()


