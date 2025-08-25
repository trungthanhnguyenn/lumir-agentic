# 🚀 QUICK START GUIDE - LUMIR-AI SYSTEM

## ⚡ **KHỞI ĐỘNG NHANH**

### **1. Cài đặt Dependencies**
```bash
cd study/rag-agentic-parlant
pip install -r requirements.txt
```

### **2. Cấu hình API Keys**
Tạo file `.env` trong thư mục gốc:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=your_base_url_here
MODEL_NAME=your_model_name_here
```

### **3. Test Hệ Thống**
```bash
python test_infer.py
```

## 🎯 **CÁC LOẠI TEST**

### **Test Câu Hỏi Đơn Lẻ**
```bash
python test_infer.py
# Nhập câu hỏi và thông tin khi được hỏi
```

**Ví dụ câu hỏi:**
- "Tôi thường bị FOMO khi thấy giá tăng"
- "Tình hình trading hiện tại của tôi thế nào?"
- "Tôi phù hợp với kiểu trading nào?"

### **Test Multi-Turn Chat**
```bash
python test_infer.py
# Nhập 'chat' khi được hỏi
# Sau đó nhập các câu hỏi liên tiếp
```

### **Kiểm Tra System Status**
```bash
python test_infer.py
# Nhập 'status' khi được hỏi
```

## 🔧 **CẤU TRÚC DỰ ÁN**

```
rag-agentic-parlant/
├── agents/                 # Các agent chính
│   ├── question_decomposition_agent.py  # 🔍 Phân tích câu hỏi
│   ├── numerology_agent.py              # 🔮 Phân tích thần số học
│   ├── trading_agent.py                 # 📈 Phân tích trading
│   ├── lumir_synthesis_agent.py         # 🤖 Tổng hợp và trả lời
│   ├── memory_agent.py                  # 🧠 Quản lý memory
│   └── multi_agent_orchestrator.py      # 🎯 Điều phối hệ thống
├── tools/                  # Công cụ hỗ trợ
│   ├── trading_tool.py                  # 📊 Xử lý dữ liệu Excel
│   ├── numerology_tool.py               # 🔢 Tính toán thần số học
│   └── data_validator_tool.py          # ✅ Validation dữ liệu
├── prompts/                # Prompt templates
├── .memory_cache/          # Cache files
├── config.py               # Cấu hình
├── test_infer.py          # File test chính
├── README.md              # Tài liệu chính
├── ARCHITECTURE.md        # Kiến trúc chi tiết
└── QUICK_START.md         # Hướng dẫn này
```

## 🧪 **VÍ DỤ SỬ DỤNG**

### **Test với Input Tùy Chỉnh**
```python
from agents.multi_agent_orchestrator import build_multi_agent_orchestrator

# Khởi tạo orchestrator
orchestrator = build_multi_agent_orchestrator()

# Xử lý câu hỏi
result = orchestrator.process_user_question(
    question="Tôi thường bị FOMO khi thấy giá tăng",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    excel_path=None,
    language="vi",
    username="nguyenvana"
)

print(f"Response: {result['response']}")
print(f"Processing time: {result['processing_time']:.2f}s")
```

### **Test Multi-Turn Conversation**
```python
# Turn 1
result1 = orchestrator.process_user_question(
    question="Tôi phù hợp với kiểu trading nào?",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    username="nguyenvana"
)

# Turn 2 (hệ thống sẽ nhớ context từ turn 1)
result2 = orchestrator.process_user_question(
    question="Vậy tôi nên làm gì cụ thể?",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    username="nguyenvana"
)

# Kiểm tra conversation history
history = orchestrator.get_conversation_history()
print(f"Conversation turns: {len(history)}")
```

## 📊 **CÁC LOẠI RESPONSE**

| Loại Response | Mô tả | Ví dụ |
|---------------|-------|-------|
| **General Chat** | Câu hỏi đơn giản, không liên quan trading | "Xin chào", "Thời tiết hôm nay" |
| **Trading Advice** | Kết hợp numerology + trading data | "Dựa vào tính cách và dữ liệu..." |
| **Information Gathering** | Cần thêm thông tin | "Bạn có thể cung cấp thêm..." |
| **Abnormal Behavior** | Phát hiện hành vi lệch chuẩn | "Tôi thấy bạn có dấu hiệu FOMO..." |

## 🔍 **PHÁT HIỆN HÀNH VI LỆCH CHUẨN**

Hệ thống tự động phát hiện:

- **FOMO** (Fear Of Missing Out): Sợ bỏ lỡ cơ hội
- **Revenge Trading**: Giao dịch trả thù khi thua lỗ
- **Overtrading**: Giao dịch quá nhiều
- **Emotional Trading**: Giao dịch theo cảm xúc

## 🧠 **MEMORY SYSTEM**

### **Tính năng:**
- Tự động cache kiến thức quan trọng
- Trả lời nhanh từ cache (0.1s vs 5s)
- Quản lý theo user UUID
- Tự động cleanup và refresh

### **Cache Structure:**
```json
{
  "key": "trading_style",
  "summary": "User prefers swing trading",
  "confidence": 0.95,
  "timestamp": "2025-08-24T11:17:22",
  "source_turns": [1, 5]
}
```

## 🌐 **HỖ TRỢ ĐA NGÔN NGỮ**

- **Tiếng Việt** (`vi`): Mặc định
- **Tiếng Anh** (`en`): Hỗ trợ đầy đủ

```python
result = orchestrator.process_user_question(
    question="I often feel FOMO when I see prices rising",
    language="en",
    username="john_doe"
)
```

## ⚠️ **LỖI THƯỜNG GẶP & CÁCH KHẮC PHỤC**

### **1. Import Error**
```bash
# Đảm bảo chạy từ thư mục gốc
cd study/rag-agentic-parlant
python test_infer.py
```

### **2. API Key Error**
```bash
# Kiểm tra file .env
cat .env
# Đảm bảo có OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
```

### **3. File Not Found**
```bash
# Kiểm tra đường dẫn file Excel
ls -la *.xlsx
# Hoặc sử dụng đường dẫn tuyệt đối
```

### **4. Memory Error**
```bash
# Clear conversation history
# Trong test_infer.py, nhập 'status' và xem memory usage
```

## 📈 **PERFORMANCE METRICS**

| Scenario | Response Time | Notes |
|----------|---------------|-------|
| **Cache Hit** | 0.1-0.3s | Trả lời từ memory |
| **General Chat** | 1.5-2.5s | Chỉ cần question decomposition |
| **Trading Analysis** | 4.0-7.0s | Full pipeline với Excel data |
| **Complex Question** | 5.0-8.0s | Cả numerology + trading |

## 🚀 **NEXT STEPS**

### **Sau khi test thành công:**

1. **Đọc README.md**: Hiểu tổng quan hệ thống
2. **Xem ARCHITECTURE.md**: Hiểu kiến trúc chi tiết
3. **Customize prompts**: Điều chỉnh behavior của các agent
4. **Add new agents**: Mở rộng hệ thống
5. **Integration**: Tích hợp vào ứng dụng khác

### **Development Tips:**

- Sử dụng `verbose=True` để debug
- Monitor memory usage qua `get_memory_status()`
- Test với nhiều loại input khác nhau
- Validate response quality và relevance

---

**🎯 Mục tiêu: Hiểu và sử dụng LUMIR-AI System trong 5 phút!**

**📚 Để biết thêm chi tiết, xem README.md và ARCHITECTURE.md**
