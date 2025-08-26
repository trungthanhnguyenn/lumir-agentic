#!/usr/bin/env python3
"""
FastAPI Application cho LUMIR-AI System
Chuyển đổi các API endpoints thành HTTP endpoints với user-specific cache memory
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
import os
import tempfile
from pathlib import Path
from datetime import datetime

# Thêm thư mục gốc vào Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from api_endpoints import build_lumir_api_endpoints

# Khởi tạo FastAPI app
app = FastAPI(
    title="LUMIR-AI API",
    description="Smart Multi-Agent Chatbot System for Trading Advice and Emotional Control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo LUMIR API endpoints
lumir_api = build_lumir_api_endpoints()

# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ============================================================================

class MemoryCheckRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của user")
    user_name: str = Field(..., description="Tên user")
    birthday: str = Field(..., description="Ngày sinh")
    username: str = Field(..., description="Username")
    language: str = Field(default="vi", description="Ngôn ngữ (mặc định: vi)")

class QuestionDecompositionRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của user")
    user_name: Optional[str] = Field(None, description="Tên user (optional)")
    birthday: Optional[str] = Field(None, description="Ngày sinh (optional)")
    username: Optional[str] = Field(None, description="Username (optional)")
    language: str = Field(default="vi", description="Ngôn ngữ (mặc định: vi)")

class NumerologyRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi về numerology")
    user_name: str = Field(..., description="Tên user")
    birthday: str = Field(..., description="Ngày sinh")
    language: str = Field(default="vi", description="Ngôn ngữ (mặc định: vi)")

class TradingRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi về trading")
    language: str = Field(default="vi", description="Ngôn ngữ (mặc định: vi)")

class LumirSynthesisRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi gốc")
    question_type: str = Field(default="general_chat", description="Loại câu hỏi")
    numerology_context: str = Field(default="", description="Context từ numerology agent")
    trading_context: str = Field(default="", description="Context từ trading agent")
    user_name: str = Field(default="", description="Tên user")
    username: str = Field(default="", description="Username")
    language: str = Field(default="vi", description="Ngôn ngữ (mặc định: vi)")
    has_trading_data: bool = Field(default=False, description="Có dữ liệu trading không")
    focus_areas: List[str] = Field(default=[], description="Các lĩnh vực tập trung")
    needs_user_info: bool = Field(default=False, description="Có cần thêm thông tin không")
    suggested_questions: List[str] = Field(default=[], description="Câu hỏi gợi ý")
    conversation_history: List[Dict[str, Any]] = Field(default=[], description="Lịch sử hội thoại")

class CompletePipelineRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của user")
    user_name: Optional[str] = Field(None, description="Tên user (optional)")
    birthday: Optional[str] = Field(None, description="Ngày sinh (optional)")
    username: Optional[str] = Field(None, description="Username (optional)")
    language: str = Field(default="vi", description="Ngôn ngữ (mặc định: vi)")

class MemoryManagementRequest(BaseModel):
    action: str = Field(..., description="Hành động ('get_status', 'clear', 'get_summary')")
    user_name: str = Field(..., description="Tên user")
    birthday: str = Field(..., description="Ngày sinh")
    username: str = Field(..., description="Username")

class MemoryUpdateRequest(BaseModel):
    action: str = Field(..., description="Hành động ('add_entry', 'update_entry', 'bulk_update', 'remove_entry')")
    user_name: str = Field(..., description="Tên user")
    birthday: str = Field(..., description="Ngày sinh")
    username: str = Field(..., description="Username")
    entry_key: Optional[str] = Field(None, description="Key của entry (cho update_entry, remove_entry)")
    entry_data: Optional[Dict[str, Any]] = Field(None, description="Dữ liệu entry để thêm/cập nhật")
    entries: Optional[List[Dict[str, Any]]] = Field(None, description="Danh sách entries (cho bulk_update)")
    language: str = Field(default="vi", description="Ngôn ngữ")

class MemoryHistoryRequest(BaseModel):
    user_name: str = Field(..., description="Tên user")
    birthday: str = Field(..., description="Ngày sinh")
    username: str = Field(..., description="Username")
    limit: Optional[int] = Field(None, description="Số lượng turn gần nhất")

# ============================================================================
# ENDPOINT 1: MEMORY CHECK
# ============================================================================

@app.post("/api/memory/check", response_model=Dict[str, Any])
async def memory_check_endpoint(request: MemoryCheckRequest):
    """
    Kiểm tra memory cache cho user cụ thể
    
    Mỗi user có cache memory riêng biệt dựa trên user_name, birthday, và username
    """
    try:
        result = lumir_api.memory_check_endpoint(
            question=request.question,
            user_name=request.user_name,
            birthday=request.birthday,
            username=request.username,
            language=request.language
        )
        
        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        error_response = {
            "endpoint": "memory_check",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# =========================================================================
# ENDPOINT 9: MEMORY HISTORY (NEW)
# =========================================================================

@app.post("/api/memory/history", response_model=Dict[str, Any])
async def memory_history_endpoint(request: MemoryHistoryRequest):
    """
    Lấy conversation history để truyền vào synthesize endpoint
    """
    try:
        result = lumir_api.memory_history_endpoint(
            user_name=request.user_name,
            birthday=request.birthday,
            username=request.username,
            limit=request.limit
        )
        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
    except Exception as e:
        error_response = {
            "endpoint": "memory_history",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# ============================================================================
# ENDPOINT 2: QUESTION DECOMPOSITION
# ============================================================================

@app.post("/api/question/decompose", response_model=Dict[str, Any])
async def question_decomposition_endpoint(
    question: str = Form(...),
    user_name: Optional[str] = Form(None),
    birthday: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    language: str = Form(default="vi"),
    excel_file: Optional[UploadFile] = File(None)
):
    # Lưu file tạm thời và truyền excel_path
    excel_path = None
    if excel_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            content = await excel_file.read()
            temp_file.write(content)
            excel_path = temp_file.name
    
    try:
        result = lumir_api.question_decomposition_endpoint(
            question=question,
            user_name=user_name,
            birthday=birthday,
            excel_path=excel_path,  # Truyền đường dẫn file
            language=language,
            username=username
        )
        return result
    finally:
        # Xóa file tạm
        if excel_path and os.path.exists(excel_path):
            os.unlink(excel_path)

# ============================================================================
# ENDPOINT 3: NUMEROLOGY ANALYSIS
# ============================================================================

@app.post("/api/numerology/analyze", response_model=Dict[str, Any])
async def numerology_endpoint(request: NumerologyRequest):
    """
    Phân tích numerology cho user cụ thể
    
    Mỗi user có cache memory riêng biệt
    """
    try:
        result = lumir_api.numerology_endpoint(
            question=request.question,
            user_name=request.user_name,
            birthday=request.birthday,
            language=request.language
        )
        
        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        error_response = {
            "endpoint": "numerology",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# ============================================================================
# ENDPOINT 4: TRADING ANALYSIS (WITH FILE UPLOAD)
# ============================================================================

@app.post("/api/trading/analyze", response_model=Dict[str, Any])
async def trading_endpoint(
    question: str = Form(...),
    language: str = Form(default="vi"),
    excel_file: UploadFile = File(...)
):
    """
    Phân tích trading data với file Excel upload
    
    File Excel được lưu tạm thời và xử lý, sau đó xóa
    """
    try:
        # Lưu file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            content = await excel_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            result = lumir_api.trading_endpoint(
                question=question,
                excel_path=temp_file_path,
                language=language
            )
            
            if result["success"]:
                return JSONResponse(content=result, status_code=200)
            else:
                return JSONResponse(content=result, status_code=400)
                
        finally:
            # Xóa file tạm thời
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        error_response = {
            "endpoint": "trading",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# ============================================================================
# ENDPOINT 5: LUMIR-AI SYNTHESIS
# ============================================================================

@app.post("/api/lumir/synthesize", response_model=Dict[str, Any])
async def lumir_synthesis_endpoint(request: LumirSynthesisRequest):
    """
    Tổng hợp và tạo final response từ LUMIR-AI
    
    Kết hợp thông tin từ các agent khác để tạo response hoàn chỉnh
    """
    try:
        result = lumir_api.lumir_synthesis_endpoint(
            question=request.question,
            question_type=request.question_type,
            numerology_context=request.numerology_context,
            trading_context=request.trading_context,
            user_name=request.user_name,
            username=request.username,
            language=request.language,
            has_trading_data=request.has_trading_data,
            focus_areas=request.focus_areas,
            needs_user_info=request.needs_user_info,
            suggested_questions=request.suggested_questions,
            conversation_history=request.conversation_history
        )
        
        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        error_response = {
            "endpoint": "lumir_synthesis",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# ============================================================================
# ENDPOINT 6: COMPLETE PIPELINE (Mô phỏng test_infer)
# ============================================================================

@app.post("/api/pipeline/complete", response_model=Dict[str, Any])
async def complete_pipeline_endpoint(request: CompletePipelineRequest):
    """
    Complete pipeline - mô phỏng logic test_infer
    
    Đây là endpoint chính để sử dụng toàn bộ hệ thống LUMIR-AI
    Mỗi user có cache memory riêng biệt dựa trên user_name, birthday, và username
    """
    try:
        result = lumir_api.complete_pipeline_endpoint(
            question=request.question,
            user_name=request.user_name,
            birthday=request.birthday,
            excel_path=None,  # Không có file upload trong endpoint này
            language=request.language,
            username=request.username
        )
        
        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        error_response = {
            "endpoint": "complete_pipeline",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# ============================================================================
# ENDPOINT 7: MEMORY MANAGEMENT
# ============================================================================

@app.post("/api/memory/manage", response_model=Dict[str, Any])
async def memory_management_endpoint(request: MemoryManagementRequest):
    """
    Quản lý memory cache cho user cụ thể
    
    Mỗi user có cache memory riêng biệt dựa trên user_name, birthday, và username
    """
    try:
        result = lumir_api.memory_management_endpoint(
            action=request.action,
            user_name=request.user_name,
            birthday=request.birthday,
            username=request.username
        )
        
        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        error_response = {
            "endpoint": "memory_management",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# ============================================================================
# ENDPOINT 8: MEMORY UPDATE (NEW)
# ============================================================================

@app.post("/api/memory/update", response_model=Dict[str, Any])
async def memory_update_endpoint(request: MemoryUpdateRequest):
    """
    Cập nhật memory cache cho user cụ thể
    
    Hỗ trợ:
    - add_entry: Thêm entry mới
    - update_entry: Cập nhật entry hiện có
    - bulk_update: Cập nhật nhiều entries cùng lúc
    - remove_entry: Xóa entry cụ thể
    """
    try:
        result = lumir_api.memory_update_endpoint(
            action=request.action,
            user_name=request.user_name,
            birthday=request.birthday,
            username=request.username,
            entry_key=request.entry_key,
            entry_data=request.entry_data,
            entries=request.entries,
            language=request.language
        )
        
        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        error_response = {
            "endpoint": "memory_update",
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=error_response, status_code=500)

# ============================================================================
# HEALTH CHECK & INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint với thông tin hệ thống"""
    return {
        "message": "LUMIR-AI API System",
        "version": "1.0.0",
        "description": "Smart Multi-Agent Chatbot System for Trading Advice and Emotional Control",
        "endpoints": {
            "memory_check": "/api/memory/check",
            "question_decomposition": "/api/question/decompose",
            "numerology": "/api/numerology/analyze",
            "trading": "/api/trading/analyze",
            "lumir_synthesis": "/api/lumir/synthesize",
            "complete_pipeline": "/api/pipeline/complete",
            "memory_management": "/api/memory/manage",
            "memory_update": "/api/memory/update",
            "memory_history": "/api/memory/history"
        },
        "features": [
            "User-specific cache memory (UUID-based)",
            "Intelligent question routing",
            "Multi-turn conversation support",
            "Language consistency",
            "Parallel agent execution"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "LUMIR-AI API"
    }

@app.get("/api/info")
async def api_info():
    """Thông tin chi tiết về API"""
    return {
        "api_name": "LUMIR-AI API",
        "version": "1.0.0",
        "description": "Smart Multi-Agent Chatbot System",
        "architecture": {
            "question_decomposition_agent": "Intelligent routing and question analysis",
            "numerology_agent": "Numerology calculations and insights",
            "trading_agent": "Trading data analysis and reporting",
            "lumir_synthesis_agent": "Final response synthesis",
            "memory_agent": "User-specific knowledge caching"
        },
        "user_memory_system": {
            "description": "Each user has separate cache memory",
            "identification": "Based on user_name + birthday + username",
            "storage": "UUID-based hashing for unique identification",
            "isolation": "Complete memory isolation between users"
        },
        "endpoints": {
            "memory_check": {
                "url": "/api/memory/check",
                "method": "POST",
                "description": "Check user-specific memory cache"
            },
            "question_decomposition": {
                "url": "/api/question/decompose", 
                "method": "POST",
                "description": "Intelligent question analysis and routing"
            },
            "numerology": {
                "url": "/api/numerology/analyze",
                "method": "POST", 
                "description": "Numerology analysis for specific user"
            },
            "trading": {
                "url": "/api/trading/analyze",
                "method": "POST",
                "description": "Trading data analysis with Excel upload"
            },
            "lumir_synthesis": {
                "url": "/api/lumir/synthesize",
                "method": "POST",
                "description": "Final response synthesis from all agents"
            },
            "complete_pipeline": {
                "url": "/api/pipeline/complete",
                "method": "POST",
                "description": "Complete LUMIR-AI pipeline (main endpoint)"
            },
            "memory_management": {
                "url": "/api/memory/manage",
                "method": "POST",
                "description": "Manage user-specific memory cache"
            },
            "memory_update": {
                "url": "/api/memory/update",
                "method": "POST",
                "description": "Update user-specific memory cache (add, update, remove entries)"
            },
            "memory_history": {
                "url": "/api/memory/history",
                "method": "POST",
                "description": "Get conversation history for a specific user"
            }
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "endpoint": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": f"Internal server error: {str(exc)}",
            "timestamp": datetime.now().isoformat(),
            "endpoint": str(request.url)
        }
    )

# ============================================================================
# MAIN FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting LUMIR-AI FastAPI Server...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    print("🔍 API Info available at: http://localhost:8000/api/info")
    print("💡 Health check available at: http://localhost:8000/health")
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
