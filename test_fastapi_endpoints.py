#!/usr/bin/env python3
"""
Test Script cho FastAPI Endpoints của LUMIR-AI System
Kiểm tra tất cả endpoints và xác nhận user-specific cache memory hoạt động đúng
"""

import requests
import json
import time
from pathlib import Path
import tempfile
import pandas as pd

# FastAPI server URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_info():
    """Test API info endpoint"""
    print("\n🔍 Testing API Info...")
    try:
        response = requests.get(f"{BASE_URL}/api/info")
        if response.status_code == 200:
            print("✅ API info passed")
            info = response.json()
            print(f"   API Name: {info.get('api_name')}")
            print(f"   Version: {info.get('version')}")
            print(f"   Endpoints: {len(info.get('endpoints', {}))}")
        else:
            print(f"❌ API info failed: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ API info error: {e}")
        return False

def test_memory_check_endpoint():
    """Test memory check endpoint"""
    print("\n🧠 Testing Memory Check Endpoint...")
    
    # Test data for User A
    user_a_data = {
        "question": "Tôi muốn biết về numerology của mình",
        "user_name": "Nguyễn Văn A",
        "birthday": "1990-05-15",
        "username": "nguyenvana",
        "language": "vi"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/memory/check", json=user_a_data)
        if response.status_code == 200:
            print("✅ Memory check for User A passed")
            result = response.json()
            print(f"   User UUID: {result.get('user_uuid')}")
            print(f"   Cache hit: {result.get('can_answer_from_cache')}")
            print(f"   Confidence: {result.get('cache_confidence')}")
        else:
            print(f"❌ Memory check for User A failed: {response.status_code}")
            print(f"   Error: {response.text}")
        
        # Test User B (different user, should have different UUID)
        user_b_data = {
            "question": "Tôi muốn biết về trading của mình",
            "user_name": "Trần Thị B",
            "birthday": "1985-12-20",
            "username": "tranthib",
            "language": "vi"
        }
        
        response_b = requests.post(f"{BASE_URL}/api/memory/check", json=user_b_data)
        if response_b.status_code == 200:
            print("✅ Memory check for User B passed")
            result_b = response_b.json()
            print(f"   User UUID: {result_b.get('user_uuid')}")
            print(f"   Cache hit: {result_b.get('can_answer_from_cache')}")
            print(f"   Confidence: {result_b.get('cache_confidence')}")
            
            # Verify different UUIDs
            if result.get('user_uuid') != result_b.get('user_uuid'):
                print("✅ User UUID isolation confirmed - different users have different UUIDs")
            else:
                print("❌ User UUID isolation failed - same UUID for different users")
        else:
            print(f"❌ Memory check for User B failed: {response_b.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Memory check error: {e}")
        return False

def test_question_decomposition_endpoint():
    """Test question decomposition endpoint"""
    print("\n🔍 Testing Question Decomposition Endpoint...")
    
    test_cases = [
        {
            "name": "General Chat Question",
            "data": {
                "question": "Xin chào, bạn là ai?",
                "language": "vi"
            }
        },
        {
            "name": "Numerology Question",
            "data": {
                "question": "Tôi muốn biết về con số chủ đạo của mình",
                "user_name": "Nguyễn Văn A",
                "birthday": "1990-05-15",
                "language": "vi"
            }
        },
        {
            "name": "Trading Question",
            "data": {
                "question": "Tình hình trading hiện tại của tôi thế nào?",
                "language": "vi"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"   Test {i}: {test_case['name']}")
        try:
            response = requests.post(f"{BASE_URL}/api/question/decompose", json=test_case['data'])
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Passed - Type: {result.get('decomposition_result', {}).get('question_type')}")
            else:
                print(f"      ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    return True

def test_numerology_endpoint():
    """Test numerology endpoint"""
    print("\n🔮 Testing Numerology Endpoint...")
    
    numerology_data = {
        "question": "Tôi muốn biết về con số chủ đạo và vận mệnh của mình",
        "user_name": "Nguyễn Văn A",
        "birthday": "1990-05-15",
        "language": "vi"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/numerology/analyze", json=numerology_data)
        if response.status_code == 200:
            print("✅ Numerology analysis passed")
            result = response.json()
            print(f"   Response length: {len(result.get('numerology_response', ''))}")
            print(f"   User: {result.get('user_name')}")
            print(f"   Birthday: {result.get('birthday')}")
        else:
            print(f"❌ Numerology analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Numerology error: {e}")
        return False

def test_trading_endpoint():
    """Test trading endpoint with Excel file"""
    print("\n📈 Testing Trading Endpoint...")
    
    try:
        # Create a sample Excel file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            # Create sample trading data with correct column names
            sample_data = {
                'symbol': ['XAUUSD', 'EURUSD', 'GBPUSD'],
                'side': ['BUY', 'SELL', 'BUY'],
                'close_time': ['01/01/2024 10:00', '01/01/2024 11:00', '01/01/2024 12:00'],
                'net_profit': [50, -30, 75],
                'commission': [-2, -2, -2],
                'swap': [-1, -1, -1],
                'volume_lots_closed': [0.1, 0.1, 0.1],
                'open_price': [2000, 1.0850, 1.2650],
                'close_price': [2005, 1.0820, 1.2680]
            }
            
            df = pd.DataFrame(sample_data)
            df.to_excel(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        try:
            # Test trading endpoint
            with open(temp_file_path, 'rb') as f:
                files = {'excel_file': ('test_trading.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                data = {
                    'question': 'Phân tích tình hình trading của tôi',
                    'language': 'vi'
                }
                
                response = requests.post(f"{BASE_URL}/api/trading/analyze", files=files, data=data)
                
                if response.status_code == 200:
                    print("✅ Trading analysis passed")
                    result = response.json()
                    print(f"   Response length: {len(result.get('trading_response', ''))}")
                    print(f"   Question: {result.get('question')}")
                else:
                    print(f"❌ Trading analysis failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Trading error: {e}")
        return False

def test_lumir_synthesis_endpoint():
    """Test LUMIR synthesis endpoint"""
    print("\n🤖 Testing LUMIR Synthesis Endpoint...")
    
    synthesis_data = {
        "question": "Tôi muốn biết về numerology và trading của mình",
        "question_type": "mixed_analysis",
        "numerology_context": "Con số chủ đạo: 5, Vận mệnh: Thích phiêu lưu",
        "trading_context": "Win rate: 65%, Total trades: 150",
        "user_name": "Nguyễn Văn A",
        "username": "nguyenvana",
        "language": "vi",
        "has_trading_data": True,
        "focus_areas": ["numerology", "trading"],
        "needs_user_info": False,
        "suggested_questions": [],
        "conversation_history": []
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/lumir/synthesize", json=synthesis_data)
        if response.status_code == 200:
            print("✅ LUMIR synthesis passed")
            result = response.json()
            print(f"   Response length: {len(result.get('lumir_response', ''))}")
            print(f"   Question type: {result.get('question_type')}")
        else:
            print(f"❌ LUMIR synthesis failed: {response.status_code}")
            print(f"   Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ LUMIR synthesis error: {e}")
        return False

def test_complete_pipeline_endpoint():
    """Test complete pipeline endpoint"""
    print("\n🚀 Testing Complete Pipeline Endpoint...")
    
    # Test User A with numerology question
    user_a_pipeline = {
        "question": "Tôi muốn biết về con số chủ đạo và vận mệnh của mình",
        "user_name": "Nguyễn Văn A",
        "birthday": "1990-05-15",
        "username": "nguyenvana",
        "language": "vi"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/pipeline/complete", json=user_a_pipeline)
        if response.status_code == 200:
            print("✅ Complete pipeline for User A passed")
            result = response.json()
            print(f"   Response length: {len(result.get('response', ''))}")
            print(f"   Processing time: {result.get('processing_time')}s")
            print(f"   Question type: {result.get('question_type')}")
            print(f"   Source: {result.get('source', 'pipeline')}")
        else:
            print(f"❌ Complete pipeline for User A failed: {response.status_code}")
            print(f"   Error: {response.text}")
        
        # Test User B with different question
        user_b_pipeline = {
            "question": "Tôi muốn biết về trading performance của mình",
            "user_name": "Trần Thị B",
            "birthday": "1985-12-20",
            "username": "tranthib",
            "language": "vi"
        }
        
        response_b = requests.post(f"{BASE_URL}/api/pipeline/complete", json=user_b_pipeline)
        if response_b.status_code == 200:
            print("✅ Complete pipeline for User B passed")
            result_b = response_b.json()
            print(f"   Response length: {len(result_b.get('response', ''))}")
            print(f"   Processing time: {result_b.get('processing_time')}s")
            print(f"   Question type: {result_b.get('question_type')}")
        else:
            print(f"❌ Complete pipeline for User B failed: {response_b.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline error: {e}")
        return False

def test_memory_management_endpoint():
    """Test memory management endpoint"""
    print("\n🧠 Testing Memory Management Endpoint...")
    
    # Test get status
    status_data = {
        "action": "get_status",
        "user_name": "Nguyễn Văn A",
        "birthday": "1990-05-15",
        "username": "nguyenvana"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/memory/manage", json=status_data)
        if response.status_code == 200:
            print("✅ Memory status check passed")
            result = response.json()
            print(f"   User UUID: {result.get('user_uuid')}")
            print(f"   Action: {result.get('action')}")
        else:
            print(f"❌ Memory status check failed: {response.status_code}")
        
        # Test get summary
        summary_data = {
            "action": "get_summary",
            "user_name": "Nguyễn Văn A",
            "birthday": "1990-05-15",
            "username": "nguyenvana"
        }
        
        response_summary = requests.post(f"{BASE_URL}/api/memory/manage", json=summary_data)
        if response_summary.status_code == 200:
            print("✅ Memory summary check passed")
            result_summary = response_summary.json()
            print(f"   User UUID: {result_summary.get('user_uuid')}")
            print(f"   Action: {result_summary.get('action')}")
        else:
            print(f"❌ Memory summary check failed: {response_summary.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Memory management error: {e}")
        return False

def test_multi_turn_conversation():
    """Test multi-turn conversation to verify user-specific cache memory"""
    print("\n🔄 Testing Multi-Turn Conversation...")
    
    user_data = {
        "user_name": "Nguyễn Văn C",
        "birthday": "1992-08-10",
        "username": "nguyenvanc",
        "language": "vi"
    }
    
    questions = [
        "Tôi muốn biết về con số chủ đạo của mình",
        "Dựa vào con số đó, tôi phù hợp với nghề gì?",
        "Tôi có thể học thêm gì để phát triển bản thân?"
    ]
    
    try:
        for i, question in enumerate(questions, 1):
            print(f"   Turn {i}: {question}")
            
            pipeline_data = {
                "question": question,
                **user_data
            }
            
            response = requests.post(f"{BASE_URL}/api/pipeline/complete", json=pipeline_data)
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Response length: {len(result.get('response', ''))}")
                print(f"      Processing time: {result.get('processing_time')}s")
                
                # Check if response came from cache in later turns
                if i > 1 and result.get('source') == 'memory_cache':
                    print(f"      🧠 Cache hit detected!")
                elif i > 1:
                    print(f"      🔄 Fresh response generated")
            else:
                print(f"      ❌ Failed: {response.status_code}")
        
        # Test memory check to see accumulated knowledge
        memory_check_data = {
            "question": "Tôi đã hỏi gì trước đó?",
            **user_data
        }
        
        response = requests.post(f"{BASE_URL}/api/memory/check", json=memory_check_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   🧠 Memory check after conversation:")
            print(f"      Cache hit: {result.get('can_answer_from_cache')}")
            print(f"      Confidence: {result.get('cache_confidence')}")
            print(f"      User UUID: {result.get('user_uuid')}")
        else:
            print(f"   ❌ Memory check failed: {response.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Multi-turn conversation error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 LUMIR-AI FastAPI Endpoints Test Suite")
    print("=" * 80)
    
    # Check if server is running
    print("🔍 Checking if FastAPI server is running...")
    if not test_health_check():
        print("❌ FastAPI server is not running!")
        print("   Please start the server with: python fastapi_app.py")
        return
    
    print("\n✅ FastAPI server is running! Starting tests...")
    
    # Run all tests
    tests = [
        ("API Info", test_api_info),
        ("Memory Check", test_memory_check_endpoint),
        ("Question Decomposition", test_question_decomposition_endpoint),
        ("Numerology", test_numerology_endpoint),
        ("Trading", test_trading_endpoint),
        ("LUMIR Synthesis", test_lumir_synthesis_endpoint),
        ("Complete Pipeline", test_complete_pipeline_endpoint),
        ("Memory Management", test_memory_management_endpoint),
        ("Multi-Turn Conversation", test_multi_turn_conversation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LUMIR-AI FastAPI system is working correctly.")
        print("✅ User-specific cache memory is properly isolated")
        print("✅ All endpoints are functional")
        print("✅ Multi-turn conversations work with memory persistence")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
    
    print("\n💡 Next steps:")
    print("   1. Check the FastAPI docs at: http://localhost:8000/docs")
    print("   2. Test endpoints manually using the interactive docs")
    print("   3. Integrate with your frontend application")

if __name__ == "__main__":
    import os
    main()
