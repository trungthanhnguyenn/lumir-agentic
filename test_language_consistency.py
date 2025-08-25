#!/usr/bin/env python3
"""
Test script để kiểm tra tính nhất quán ngôn ngữ trong hệ thống LUMIR-AI
"""

import sys
import os
from pathlib import Path

# Thêm thư mục gốc vào Python path
sys.path.insert(0, str(Path(__file__).parent))

from agents.multi_agent_orchestrator import build_multi_agent_orchestrator


def test_language_consistency():
    """Test tính nhất quán ngôn ngữ"""
    
    print("🌐 LUMIR-AI LANGUAGE CONSISTENCY TEST")
    print("=" * 80)
    print("Kiểm tra xem tất cả agents có trả lời đúng ngôn ngữ không")
    print("=" * 80)
    
    # Khởi tạo orchestrator
    orchestrator = build_multi_agent_orchestrator()
    
    # Test cases với các ngôn ngữ khác nhau
    test_cases = [
        {
            "language": "vi",
            "question": "Xin chào, bạn là ai?",
            "user_name": "Nguyễn Văn A",
            "birthday": "15/03/1990",
            "username": "nguyenvana"
        },
        {
            "language": "en",
            "question": "Hello, who are you?",
            "user_name": "John Doe",
            "birthday": "15/03/1990",
            "username": "johndoe"
        },
        {
            "language": "vi",
            "question": "Tình hình trading hiện tại của tôi thế nào?",
            "user_name": "Nguyễn Văn A",
            "birthday": "15/03/1990",
            "username": "nguyenvana",
            "excel_path": "trading_data/sample_trading_data.xlsx"
        },
        {
            "language": "en",
            "question": "How is my current trading situation?",
            "user_name": "John Doe",
            "birthday": "15/03/1990",
            "username": "johndoe",
            "excel_path": "trading_data/sample_trading_data.xlsx"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}: {test_case['language'].upper()}")
        print(f"❓ Question: {test_case['question']}")
        print(f"👤 User: {test_case['user_name']}")
        print(f"🌐 Language: {test_case['language']}")
        print("-" * 60)
        
        try:
            # Xử lý câu hỏi
            result = orchestrator.process_user_question(
                question=test_case['question'],
                user_name=test_case['user_name'],
                birthday=test_case['birthday'],
                excel_path=test_case.get('excel_path'),
                language=test_case['language'],
                username=test_case['username']
            )
            
            if result['success']:
                print(f"✅ Response ({test_case['language']}):")
                print(f"📝 {result['response'][:200]}...")
                
                # Kiểm tra ngôn ngữ trong response
                response = result['response'].lower()
                if test_case['language'] == 'vi':
                    # Kiểm tra có từ tiếng Việt không
                    vietnamese_words = ['xin', 'chào', 'bạn', 'là', 'ai', 'tôi', 'của', 'thế', 'nào']
                    has_vietnamese = any(word in response for word in vietnamese_words)
                    print(f"🇻🇳 Vietnamese detected: {has_vietnamese}")
                elif test_case['language'] == 'en':
                    # Kiểm tra có từ tiếng Anh không
                    english_words = ['hello', 'who', 'are', 'you', 'how', 'is', 'my', 'current', 'situation']
                    has_english = any(word in response for word in english_words)
                    print(f"🇺🇸 English detected: {has_english}")
                    
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("=" * 80)
    
    print("\n🎯 LANGUAGE CONSISTENCY TEST COMPLETED!")
    print("Kiểm tra xem tất cả responses có đúng ngôn ngữ không")


if __name__ == "__main__":
    test_language_consistency()
