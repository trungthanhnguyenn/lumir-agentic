# 🚀 HƯỚNG DẪN SỬ DỤNG LUMIR-AI API ENDPOINTS

## 📋 **TỔNG QUAN**

LUMIR-AI API Endpoints cung cấp 7 endpoint riêng biệt, mỗi endpoint tương ứng với một agent cụ thể. Bạn có thể sử dụng từng endpoint độc lập hoặc kết hợp chúng để tạo pipeline hoàn chỉnh như `test_infer`.

## 🏗️ **CẤU TRÚC ENDPOINTS**

### **1. 🧠 Memory Check Endpoint**
- **Chức năng**: Kiểm tra memory cache để trả lời nhanh
- **Sử dụng**: Khi muốn kiểm tra xem có thể trả lời từ cache không

### **2. 🔍 Question Decomposition Endpoint**
- **Chức năng**: Phân tích câu hỏi và quyết định routing
- **Sử dụng**: Khi muốn biết câu hỏi thuộc loại nào và cần gọi agent nào

### **3. 🔮 Numerology Endpoint**
- **Chức năng**: Phân tích tính cách và tâm lý dựa trên thần số học
- **Sử dụng**: Khi cần insights về tính cách, tâm lý giao dịch

### **4. 📈 Trading Endpoint**
- **Chức năng**: Phân tích dữ liệu giao dịch từ Excel
- **Sử dụng**: Khi cần phân tích hiệu suất trading, patterns, metrics

### **5. 🤖 LUMIR-AI Synthesis Endpoint**
- **Chức năng**: Tổng hợp context từ các agent và tạo final response
- **Sử dụng**: Khi muốn có câu trả lời hoàn chỉnh từ LUMIR-AI

### **6. 🚀 Complete Pipeline Endpoint**
- **Chức năng**: Pipeline hoàn chỉnh mô phỏng logic `test_infer`
- **Sử dụng**: Khi muốn sử dụng toàn bộ hệ thống như một lần gọi

### **7. 🧠 Memory Management Endpoint**
- **Chức năng**: Quản lý memory cache (xem status, xóa, lấy summary)
- **Sử dụng**: Khi cần quản lý memory của user

## 💻 **CÁCH SỬ DỤNG CƠ BẢN**

### **Khởi tạo API**

```python
from api_endpoints import build_lumir_api_endpoints

# Khởi tạo API endpoints
api = build_lumir_api_endpoints()
```

### **Sử dụng từng endpoint riêng lẻ**

#### **1. Memory Check Endpoint**

```python
# Kiểm tra memory cache
memory_result = api.memory_check_endpoint(
    question="Tôi phù hợp với kiểu trading nào?",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    username="nguyenvana",
    language="vi"
)

if memory_result["success"]:
    if memory_result["can_answer_from_cache"]:
        print(f"✅ Cache hit! Confidence: {memory_result['cache_confidence']:.2f}")
        print(f"Response: {memory_result['suggested_response']}")
    else:
        print("❌ Cache miss - cần xử lý thêm")
```

#### **2. Question Decomposition Endpoint**

```python
# Phân tích câu hỏi
decomposition_result = api.question_decomposition_endpoint(
    question="Tôi thường bị FOMO khi thấy giá tăng",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    excel_path="trading_data.xlsx",
    language="vi",
    username="nguyenvana"
)

if decomposition_result["success"]:
    decomp_data = decomposition_result["decomposition_result"]
    print(f"Question Type: {decomp_data['question_type']}")
    print(f"Should Call Agents: {decomp_data['should_call_agents']}")
    print(f"Focus Areas: {decomp_data['focus_areas']}")
```

#### **3. Numerology Endpoint**

```python
# Phân tích numerology
numerology_result = api.numerology_endpoint(
    question="Tôi phù hợp với kiểu trading nào?",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    language="vi"
)

if numerology_result["success"]:
    print(f"Numerology Analysis: {numerology_result['numerology_response']}")
```

#### **4. Trading Endpoint**

```python
# Phân tích trading data
trading_result = api.trading_endpoint(
    question="Tình hình trading hiện tại của tôi thế nào?",
    excel_path="trading_data.xlsx",
    language="vi"
)

if trading_result["success"]:
    print(f"Trading Analysis: {trading_result['trading_response']}")
```

#### **5. LUMIR-AI Synthesis Endpoint**

```python
# Tổng hợp với LUMIR-AI
lumir_result = api.lumir_synthesis_endpoint(
    question="Tôi thường bị FOMO khi thấy giá tăng",
    question_type="trading_related",
    numerology_context="User có tính cách...",
    trading_context="Trading data shows...",
    user_name="Nguyễn Văn A",
    username="nguyenvana",
    language="vi",
    has_trading_data=True,
    focus_areas=["emotion", "trading_psychology"],
    needs_user_info=False,
    suggested_questions=[],
    conversation_history=[]
)

if lumir_result["success"]:
    print(f"LUMIR-AI Response: {lumir_result['lumir_response']}")
```

#### **6. Complete Pipeline Endpoint**

```python
# Sử dụng toàn bộ pipeline (như test_infer)
pipeline_result = api.complete_pipeline_endpoint(
    question="Tôi thường bị FOMO khi thấy giá tăng",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    excel_path="trading_data.xlsx",
    language="vi",
    username="nguyenvana"
)

if pipeline_result["success"]:
    print(f"✅ Pipeline completed in {pipeline_result['processing_time']:.2f}s")
    print(f"Question Type: {pipeline_result['question_type']}")
    print(f"Response: {pipeline_result['response']}")
    
    # Kiểm tra context summary
    context = pipeline_result["context_summary"]
    print(f"Numerology: {'✅' if context['numerology_available'] else '❌'}")
    print(f"Trading: {'✅' if context['trading_available'] else '❌'}")
```

#### **7. Memory Management Endpoint**

```python
# Lấy memory status
status_result = api.memory_management_endpoint(
    action="get_status",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    username="nguyenvana"
)

if status_result["success"]:
    print(f"Memory Status: {status_result['memory_status']}")

# Xóa memory
clear_result = api.memory_management_endpoint(
    action="clear",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    username="nguyenvana"
)

if clear_result["success"]:
    print(f"Memory cleared: {clear_result['message']}")
```

## 🔄 **SỬ DỤNG KẾT HỢP CÁC ENDPOINT**

### **Ví dụ 1: Pipeline tùy chỉnh**

```python
def custom_pipeline(question, user_name, birthday, excel_path, language, username):
    """Pipeline tùy chỉnh với logic riêng"""
    
    # Bước 1: Phân tích câu hỏi
    decomp_result = api.question_decomposition_endpoint(
        question, user_name, birthday, excel_path, language, username
    )
    
    if not decomp_result["success"]:
        return {"error": "Question decomposition failed"}
    
    decomp_data = decomp_result["decomposition_result"]
    
    # Bước 2: Gọi agent theo logic tùy chỉnh
    if decomp_data["question_type"] == "numerology_related":
        # Chỉ gọi numerology agent
        numerology_result = api.numerology_endpoint(
            question, user_name, birthday, language
        )
        
        if numerology_result["success"]:
            # Tổng hợp với LUMIR-AI
            lumir_result = api.lumir_synthesis_endpoint(
                question=question,
                question_type=decomp_data["question_type"],
                numerology_context=numerology_result["numerology_response"],
                trading_context="",
                user_name=user_name,
                username=username,
                language=language,
                has_trading_data=False,
                focus_areas=decomp_data["focus_areas"],
                needs_user_info=decomp_data["needs_user_info"],
                suggested_questions=decomp_data["suggested_questions"]
            )
            
            return lumir_result
    
    # Xử lý các trường hợp khác...
    return {"error": "Unsupported question type"}

# Sử dụng
result = custom_pipeline(
    question="Tôi muốn biết tính cách của mình",
    user_name="Nguyễn Văn A",
    birthday="15/03/1990",
    excel_path=None,
    language="vi",
    username="nguyenvana"
)
```

### **Ví dụ 2: Multi-turn conversation**

```python
def multi_turn_conversation():
    """Hội thoại multi-turn sử dụng các endpoint"""
    
    # Thông tin user
    user_name = "Nguyễn Văn A"
    birthday = "15/03/1990"
    username = "nguyenvana"
    language = "vi"
    
    # Turn 1: Câu hỏi đầu tiên
    print("=== TURN 1 ===")
    question1 = "Tôi phù hợp với kiểu trading nào?"
    
    result1 = api.complete_pipeline_endpoint(
        question1, user_name, birthday, None, language, username
    )
    
    if result1["success"]:
        print(f"LUMIR: {result1['response']}")
    
    # Turn 2: Câu hỏi follow-up
    print("\n=== TURN 2 ===")
    question2 = "Vậy tôi nên làm gì cụ thể?"
    
    result2 = api.complete_pipeline_endpoint(
        question2, user_name, birthday, None, language, username
    )
    
    if result2["success"]:
        print(f"LUMIR: {result2['response']}")
    
    # Kiểm tra memory status
    print("\n=== MEMORY STATUS ===")
    memory_status = api.memory_management_endpoint(
        "get_status", user_name, birthday, username
    )
    
    if memory_status["success"]:
        print(f"Memory entries: {memory_status['memory_status']['total_entries']}")

# Chạy multi-turn conversation
multi_turn_conversation()
```

### **Ví dụ 3: Batch processing**

```python
def batch_process_questions(questions, user_name, birthday, username, language="vi"):
    """Xử lý hàng loạt câu hỏi"""
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Processing Question {i}/{len(questions)} ---")
        print(f"Question: {question}")
        
        try:
            result = api.complete_pipeline_endpoint(
                question, user_name, birthday, None, language, username
            )
            
            results.append({
                "question_number": i,
                "question": question,
                "result": result
            })
            
            if result["success"]:
                print(f"✅ Success - Processing time: {result['processing_time']:.2f}s")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                "question_number": i,
                "question": question,
                "result": {"success": False, "error": str(e)}
            })
    
    return results

# Sử dụng batch processing
questions = [
    "Tôi phù hợp với kiểu trading nào?",
    "Tôi thường bị FOMO khi thấy giá tăng",
    "Làm sao để kiểm soát cảm xúc khi giao dịch?",
    "Tôi nên tập trung vào chiến lược nào?"
]

batch_results = batch_process_questions(
    questions, "Nguyễn Văn A", "15/03/1990", "nguyenvana", "vi"
)

# Tổng kết
success_count = sum(1 for r in batch_results if r["result"]["success"])
print(f"\n📊 Batch Processing Summary: {success_count}/{len(questions)} successful")
```

## 🎯 **SO SÁNH VỚI TEST_INFER**

### **Logic tương đương:**

| **test_infer** | **API Endpoint** | **Mô tả** |
|----------------|------------------|-----------|
| `test_single_question()` | `complete_pipeline_endpoint()` | Pipeline hoàn chỉnh |
| Memory check | `memory_check_endpoint()` | Kiểm tra cache |
| Question decomposition | `question_decomposition_endpoint()` | Phân tích câu hỏi |
| Agent execution | `numerology_endpoint()`, `trading_endpoint()` | Gọi agent riêng lẻ |
| LUMIR synthesis | `lumir_synthesis_endpoint()` | Tổng hợp cuối cùng |
| Memory update | `memory_management_endpoint()` | Quản lý memory |

### **Ưu điểm của API Endpoints:**

1. **Modular**: Mỗi agent là một endpoint riêng biệt
2. **Flexible**: Có thể kết hợp theo ý muốn
3. **Reusable**: Sử dụng lại các endpoint cho nhiều mục đích
4. **Testable**: Dễ dàng test từng component
5. **Scalable**: Có thể deploy riêng biệt

## ⚠️ **LƯU Ý QUAN TRỌNG**

### **1. Error Handling**
- Luôn kiểm tra `result["success"]` trước khi sử dụng kết quả
- Xử lý exception cho mỗi endpoint

### **2. Memory Management**
- Memory cache được quản lý theo user UUID
- Conversation history được lưu trữ trong memory
- Có thể xóa memory khi cần

### **3. Performance**
- `complete_pipeline_endpoint()` tương đương với `test_infer`
- Các endpoint riêng lẻ nhanh hơn khi chỉ cần một chức năng
- Memory cache giúp tăng tốc độ response

### **4. Language Support**
- Tất cả endpoint đều hỗ trợ `language` parameter
- Hỗ trợ tiếng Việt (`vi`) và tiếng Anh (`en`)

## 🚀 **BƯỚC TIẾP THEO**

### **1. Test các endpoint cơ bản**
```bash
cd study/rag-agentic-parlant
python api_endpoints.py
```

### **2. Tích hợp vào ứng dụng**
- Import `build_lumir_api_endpoints()`
- Sử dụng các endpoint theo nhu cầu
- Xử lý error và response

### **3. Mở rộng functionality**
- Thêm endpoint mới
- Customize logic cho từng endpoint
- Tối ưu performance

---

**🎯 Mục tiêu**: Sử dụng các API endpoints để có được tính linh hoạt cao hơn so với `test_infer`, đồng thời vẫn giữ được logic hoàn chỉnh của hệ thống.

**💡 Tip**: Bắt đầu với `complete_pipeline_endpoint()` để hiểu logic, sau đó sử dụng các endpoint riêng lẻ để tùy chỉnh theo nhu cầu cụ thể.
