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
    language = input_dict.get("language", "vi")
    
    try:
        # Validate profile lightly if provided
        validator = DataValidator()
        if input_dict.get("profile"):
            prof = input_dict["profile"]
            validator.validate_profile(prof.get("user_name"), prof.get("birthday"))

        # Kiểm tra xem có file_path không
        if not excel_path:
            print("⚠️ Không có file trading data được cung cấp")
            return {
                "question": question,
                "language": language,
                "total_trades": 0,
                "total_profit": 0.0,
                "win_rate": 0.0,
                "avg_profit_per_trade": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "max_consecutive_losses": 0,
                "comprehensive_report": "Không có dữ liệu trading để phân tích. Vui lòng cung cấp file Excel chứa dữ liệu giao dịch.",
                "trading_data": {},
                "analysis_result": {},
                "data_summary": {},
                "has_trading_data": False,
                "success": False,
                "error": "no_trading_data",
                "error_type": "missing_data"
            }

        # Sử dụng trading tool mới để phân tích - hỗ trợ cả file path và trading data
        trading_analysis = analyze_trading_data(
            file_path=excel_path, 
            question=question,
            trading_data=input_dict.get("trading_data"),  # Dữ liệu từ API endpoint
            excel_path=excel_path
        )
        
        if not trading_analysis.get("success"):
            return {
                "question": question,
                "error": trading_analysis.get("error", "Unknown error"),
                "error_type": "trading_analysis_error",
                "success": False,
                "language": language,
                "has_trading_data": False
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
            "language": language,
            "has_trading_data": True
        }
        
        return prompt_data
        
    except Exception as e:
        print(f"❌ Lỗi trong _prepare_trading_data: {e}")
        return {
            "question": question,
            "error": str(e),
            "error_type": "data_processing_error",
            "success": False,
            "language": language,
            "total_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_consecutive_losses": 0,
            "comprehensive_report": f"Không thể phân tích dữ liệu trading do lỗi: {str(e)}",
            "trading_data": {},
            "analysis_result": {},
            "data_summary": {},
            "has_trading_data": False
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
        
        # Kiểm tra xem có trading data không
        if not prepared_data.get("has_trading_data", False):
            # Trường hợp không có trading data - đưa ra lời khuyên chung
            if prepared_data.get("error_type") == "missing_data":
                return f"""
🔍 **PHÂN TÍCH CÂU HỎI**: {prepared_data['question']}

⚠️ **TRẠNG THÁI**: Không có dữ liệu trading để phân tích

💡 **LỜI KHUYÊN CHUNG DÀNH CHO TRADER**:

🎯 **Nguyên tắc cơ bản**:
• Luôn có kế hoạch giao dịch rõ ràng trước khi vào lệnh
• Sử dụng stop-loss và take-profit để quản lý rủi ro
• Không bao giờ đầu tư quá 2-5% vốn vào một lệnh
• Ghi chép lại mọi giao dịch để học hỏi

🧠 **Tâm lý giao dịch**:
• Kiểm soát cảm xúc - không để FOMO hoặc sợ hãi chi phối
• Chấp nhận thua lỗ là một phần của trading
• Kiên nhẫn chờ cơ hội tốt thay vì giao dịch liên tục
• Tập trung vào quá trình thay vì kết quả ngắn hạn

📊 **Quản lý vốn**:
• Xác định rõ mức rủi ro chấp nhận được
• Đa dạng hóa danh mục đầu tư
• Không sử dụng đòn bẩy quá cao
• Luôn giữ một phần vốn dự phòng

🚀 **Để được tư vấn cụ thể và cá nhân hóa**:
• **Đăng nhập vào hệ thống LUMIR-AI** với thông tin cá nhân
• **Cung cấp dữ liệu trading** (file Excel) để phân tích chi tiết
• **Kết nối với numerology analysis** để hiểu tính cách trading phù hợp
• **Nhận Behavioral Report** để phát hiện patterns và cải thiện

📈 **Các bước tiếp theo**:
1. Tạo tài khoản và đăng nhập vào LUMIR-AI
2. Cung cấp thông tin cá nhân (tên, ngày sinh)
3. Upload file Excel chứa lịch sử giao dịch
4. Nhận phân tích chi tiết và tư vấn cá nhân hóa

Bạn có muốn tôi hướng dẫn cách bắt đầu với LUMIR-AI không?
"""
            elif prepared_data.get("error_type") == "file_error":
                return f"""
❌ **LỖI ĐỌC FILE**: Không thể đọc file Excel

🔍 **Câu hỏi**: {prepared_data['question']}

⚠️ **Vấn đề**: File Excel có thể bị hỏng hoặc không đúng format

🔧 **Nguyên nhân có thể**:
• File bị hỏng trong quá trình upload
• File không phải định dạng Excel (.xlsx, .xls)
• File có encoding không tương thích
• File quá lớn hoặc quá nhỏ

💡 **Giải pháp**:
1. **Kiểm tra file**: Đảm bảo file là .xlsx hoặc .xls
2. **Thử lại**: Upload lại file Excel
3. **Format file**: Đảm bảo file có các cột cần thiết:
   - symbol (cặp tiền)
   - side (hướng giao dịch: BUY/SELL)
   - close_time (thời gian đóng lệnh)
   - net_profit (lợi nhuận ròng)
4. **Liên hệ hỗ trợ**: Nếu vẫn gặp vấn đề

📋 **Format Excel chuẩn**:
| symbol | side | close_time | net_profit | ... |
|--------|------|------------|------------|-----|
| EURUSD | BUY  | 2024-01-01 | 100.50     | ... |

Bạn có thể thử upload lại file không?
"""
            else:
                # Trường hợp có lỗi khác
                return f"""
❌ **LỖI PHÂN TÍCH**: {prepared_data.get('error', 'Unknown error')}

🔍 **Câu hỏi**: {prepared_data['question']}

⚠️ **Vấn đề**: {prepared_data.get('error_type', 'unknown')}

💡 **Giải pháp**: 
• Kiểm tra lại file Excel có đúng format không
• Đảm bảo file không bị hỏng
• Thử upload lại file

Nếu vẫn gặp vấn đề, vui lòng liên hệ hỗ trợ kỹ thuật.
"""
        
        # Trường hợp có trading data - tạo báo cáo chi tiết
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
            "comprehensive_report": prepared_data.get("comprehensive_report", 'Không có dữ liệu để phân tích'),
            "language": prepared_data.get("language", 'vi')
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


