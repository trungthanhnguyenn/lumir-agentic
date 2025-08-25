# LUMIR-AI FastAPI Implementation

## Overview

This document describes the FastAPI implementation of the LUMIR-AI system, which converts the existing API endpoints into HTTP endpoints with full user-specific cache memory isolation.

## 🏗️ Architecture

### FastAPI Application Structure

```
fastapi_app.py              # Main FastAPI application
├── /api/memory/check      # Memory cache checking
├── /api/question/decompose # Question analysis and routing
├── /api/numerology/analyze # Numerology analysis
├── /api/trading/analyze   # Trading data analysis (with file upload)
├── /api/lumir/synthesize  # Final response synthesis
├── /api/pipeline/complete # Complete pipeline (main endpoint)
├── /api/memory/manage     # Memory management operations
├── /health                # Health check
└── /api/info             # API information
```

### User-Specific Cache Memory System

**Key Features:**
- **Complete Isolation**: Each user has a completely separate cache memory
- **UUID Generation**: Unique user identification based on `user_name + birthday + username`
- **Hash-Based**: Uses MD5 hashing for consistent UUID generation
- **Persistent Storage**: Memory persists across sessions and server restarts

**UUID Generation Logic:**
```python
def _generate_user_uuid(self, user_name: str, birthday: str, username: str) -> str:
    user_string = f"{user_name}_{birthday}_{username}"
    return hashlib.md5(user_string.encode('utf-8')).hexdigest()
```

**Example UUIDs:**
- User A: `nguyenvana_1990-05-15_nguyenvana` → `abc123...`
- User B: `tranthib_1985-12-20_tranthib` → `def456...`
- User C: `nguyenvanc_1992-08-10_nguyenvanc` → `ghi789...`

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Install FastAPI-specific dependencies
pip install -r requirements_fastapi.txt

# Install main LUMIR-AI dependencies
pip install -r requirements.txt
```

### 2. Start the FastAPI Server

```bash
python fastapi_app.py
```

The server will start on `http://localhost:8000`

### 3. Access API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Info**: http://localhost:8000/api/info

## 📡 API Endpoints

### 1. Memory Check Endpoint

**URL:** `POST /api/memory/check`

**Purpose:** Check if a user's question can be answered from cache memory

**Request Body:**
```json
{
    "question": "Tôi muốn biết về numerology của mình",
    "user_name": "Nguyễn Văn A",
    "birthday": "1990-05-15",
    "username": "nguyenvana",
    "language": "vi"
}
```

**Response:**
```json
{
    "endpoint": "memory_check",
    "success": true,
    "question": "Tôi muốn biết về numerology của mình",
    "user_uuid": "abc123...",
    "can_answer_from_cache": true,
    "cache_confidence": 0.85,
    "suggested_response": "Dựa vào ngày sinh...",
    "needs_refresh": false,
    "relevant_entries": [...],
    "timestamp": "2024-01-15T10:30:00"
}
```

### 2. Question Decomposition Endpoint

**URL:** `POST /api/question/decompose`

**Purpose:** Intelligently analyze and route user questions

**Request Body:**
```json
{
    "question": "Tôi muốn biết về con số chủ đạo của mình",
    "user_name": "Nguyễn Văn A",
    "birthday": "1990-05-15",
    "username": "nguyenvana",
    "language": "vi"
}
```

**Response:**
```json
{
    "endpoint": "question_decomposition",
    "success": true,
    "question": "Tôi muốn biết về con số chủ đạo của mình",
    "decomposition_result": {
        "question_type": "numerology",
        "should_call_agents": true,
        "numerology_question": "Tính toán con số chủ đạo...",
        "trading_question": null,
        "focus_areas": ["numerology"],
        "needs_user_info": false,
        "suggested_questions": []
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

### 3. Numerology Analysis Endpoint

**URL:** `POST /api/numerology/analyze`

**Purpose:** Perform numerology calculations for a specific user

**Request Body:**
```json
{
    "question": "Tôi muốn biết về con số chủ đạo và vận mệnh",
    "user_name": "Nguyễn Văn A",
    "birthday": "1990-05-15",
    "language": "vi"
}
```

**Response:**
```json
{
    "endpoint": "numerology",
    "success": true,
    "question": "Tôi muốn biết về con số chủ đạo và vận mệnh",
    "user_name": "Nguyễn Văn A",
    "birthday": "1990-05-15",
    "numerology_response": "Con số chủ đạo của bạn là 5...",
    "timestamp": "2024-01-15T10:30:00"
}
```

### 4. Trading Analysis Endpoint

**URL:** `POST /api/trading/analyze`

**Purpose:** Analyze trading data from Excel file upload

**Request:** Multipart form data
- `question`: Trading question (string)
- `language`: Language preference (string, default: "vi")
- `excel_file`: Excel file upload

**Response:**
```json
{
    "endpoint": "trading",
    "success": true,
    "question": "Phân tích tình hình trading của tôi",
    "excel_path": "/tmp/temp_file.xlsx",
    "trading_response": "Dựa vào dữ liệu trading...",
    "timestamp": "2024-01-15T10:30:00"
}
```

### 5. LUMIR-AI Synthesis Endpoint

**URL:** `POST /api/lumir/synthesize`

**Purpose:** Synthesize final response from all agents

**Request Body:**
```json
{
    "question": "Tôi muốn biết về numerology và trading",
    "question_type": "mixed_analysis",
    "numerology_context": "Con số chủ đạo: 5...",
    "trading_context": "Win rate: 65%...",
    "user_name": "Nguyễn Văn A",
    "username": "nguyenvana",
    "language": "vi",
    "has_trading_data": true,
    "focus_areas": ["numerology", "trading"],
    "needs_user_info": false,
    "suggested_questions": [],
    "conversation_history": []
}
```

**Response:**
```json
{
    "endpoint": "lumir_synthesis",
    "success": true,
    "question": "Tôi muốn biết về numerology và trading",
    "question_type": "mixed_analysis",
    "numerology_available": true,
    "trading_available": true,
    "lumir_response": "Dựa vào phân tích tổng hợp...",
    "timestamp": "2024-01-15T10:30:00"
}
```

### 6. Complete Pipeline Endpoint

**URL:** `POST /api/pipeline/complete`

**Purpose:** Execute the complete LUMIR-AI pipeline (main endpoint)

**Request Body:**
```json
{
    "question": "Tôi muốn biết về con số chủ đạo và vận mệnh",
    "user_name": "Nguyễn Văn A",
    "birthday": "1990-05-15",
    "username": "nguyenvana",
    "language": "vi"
}
```

**Response:**
```json
{
    "endpoint": "complete_pipeline",
    "success": true,
    "question": "Tôi muốn biết về con số chủ đạo và vận mệnh",
    "response": "Dựa vào phân tích tổng hợp...",
    "processing_time": 2.5,
    "question_type": "numerology",
    "decomposition_result": {...},
    "context_summary": {
        "numerology_available": true,
        "trading_available": false,
        "response_type": "partial",
        "needs_user_info": false,
        "suggested_questions": []
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

### 7. Memory Management Endpoint

**URL:** `POST /api/memory/manage`

**Purpose:** Manage user-specific memory cache

**Request Body:**
```json
{
    "action": "get_status",  // "get_status", "clear", "get_summary"
    "user_name": "Nguyễn Văn A",
    "birthday": "1990-05-15",
    "username": "nguyenvana"
}
```

**Response:**
```json
{
    "endpoint": "memory_management",
    "action": "get_status",
    "success": true,
    "user_uuid": "abc123...",
    "memory_status": {
        "total_entries": 5,
        "last_updated": "2024-01-15T10:30:00",
        "cache_size": "2.3MB"
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

## 🔒 User-Specific Cache Memory

### How It Works

1. **User Identification**: Each user is uniquely identified by combining:
   - `user_name`
   - `birthday` 
   - `username`

2. **UUID Generation**: A hash-based UUID is generated for each unique combination

3. **Memory Isolation**: Each user's memory cache is completely separate:
   - User A cannot access User B's cache
   - Different users can have the same question without interference
   - Memory persists across sessions

4. **Cache Operations**:
   - **Store**: Knowledge extracted from conversations
   - **Query**: Search for relevant cached information
   - **Update**: Add new knowledge or refresh existing
   - **Clear**: Remove user's entire cache

### Memory Cache Structure

```json
{
    "user_uuid": "abc123...",
    "entries": [
        {
            "key": "numerology_question_1",
            "summary": "User asked about life path number",
            "context": "Full conversation context",
            "timestamp": "2024-01-15T10:30:00",
            "tags": ["numerology", "life_path"],
            "confidence": 0.9,
            "source_turns": [1, 2]
        }
    ]
}
```

## 🧪 Testing

### Run the Test Suite

```bash
python test_fastapi_endpoints.py
```

This comprehensive test suite will:
- Verify all endpoints are functional
- Confirm user-specific cache memory isolation
- Test multi-turn conversations
- Validate UUID generation uniqueness
- Check memory persistence

### Manual Testing

1. **Start the server**: `python fastapi_app.py`
2. **Open browser**: Navigate to `http://localhost:8000/docs`
3. **Test endpoints**: Use the interactive Swagger UI
4. **Verify responses**: Check that different users get different UUIDs

## 🔧 Configuration

### Environment Variables

The FastAPI app inherits configuration from the main LUMIR-AI system:
- AI model configurations
- API keys
- Memory storage settings

### CORS Settings

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### File Upload Handling

- Excel files are temporarily stored during processing
- Automatic cleanup after analysis
- Support for `.xlsx` format

## 🚀 Production Deployment

### Using Uvicorn

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn

```bash
gunicorn fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements*.txt ./
RUN pip install -r requirements.txt -r requirements_fastapi.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔍 Monitoring & Debugging

### Health Check

```bash
curl http://localhost:8000/health
```

### API Information

```bash
curl http://localhost:8000/api/info
```

### Logs

The FastAPI app includes comprehensive logging:
- Request/response logging
- Error tracking
- Performance metrics
- Memory cache operations

## 🔐 Security Considerations

### User Data Privacy

- Each user's data is completely isolated
- No cross-user data access
- Memory cache is user-specific

### API Security

- Input validation via Pydantic models
- Error handling without information leakage
- CORS configuration for frontend integration

### File Upload Security

- Temporary file storage only
- Automatic cleanup
- File type validation

## 📊 Performance

### Caching Benefits

- **Fast Responses**: Cached questions respond in ~0.1s
- **Reduced LLM Calls**: Significant cost and time savings
- **Scalability**: Memory grows with user base

### Response Times

- **Cache Hit**: ~0.1 seconds
- **Fresh Question**: 2-5 seconds (depending on complexity)
- **File Upload**: Additional 1-2 seconds for Excel processing

## 🔄 Integration Examples

### Frontend Integration

```javascript
// Example: Complete pipeline call
const response = await fetch('/api/pipeline/complete', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        question: "Tôi muốn biết về numerology",
        user_name: "Nguyễn Văn A",
        birthday: "1990-05-15",
        username: "nguyenvana",
        language: "vi"
    })
});

const result = await response.json();
console.log(result.response);
```

### Python Client

```python
import requests

# Memory check
response = requests.post('http://localhost:8000/api/memory/check', json={
    "question": "Tôi muốn biết về numerology",
    "user_name": "Nguyễn Văn A",
    "birthday": "1990-05-15",
    "username": "nguyenvana",
    "language": "vi"
})

print(response.json())
```

## 🎯 Key Benefits

1. **User Isolation**: Complete separation of user data and memory
2. **Scalability**: Each user's memory grows independently
3. **Performance**: Fast cached responses for repeated questions
4. **Flexibility**: Individual endpoints for specific use cases
5. **Standards**: RESTful API design with proper HTTP status codes
6. **Documentation**: Auto-generated interactive API docs
7. **Testing**: Comprehensive test suite for validation

## 🚨 Troubleshooting

### Common Issues

1. **Server not starting**: Check dependencies and port availability
2. **Memory errors**: Verify user identification parameters
3. **File upload failures**: Check file format and size
4. **Cache misses**: Ensure consistent user identification

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export TASKMASTER_LOG_LEVEL=DEBUG
python fastapi_app.py
```

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **LUMIR-AI Architecture**: See `ARCHITECTURE.md`
- **API Usage Guide**: See `API_USAGE_GUIDE.md`
- **Test Examples**: See `test_fastapi_endpoints.py`

---

**Note**: This FastAPI implementation maintains all the functionality of the original LUMIR-AI system while providing HTTP endpoints and ensuring complete user data isolation through the UUID-based cache memory system.
