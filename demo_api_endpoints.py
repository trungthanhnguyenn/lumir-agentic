#!/usr/bin/env python3
"""
Demo sử dụng LUMIR-AI API Endpoints
Mô phỏng logic của test_infer với các endpoint riêng biệt
"""

from api_endpoints import build_lumir_api_endpoints


def demo_memory_check():
    """Demo Memory Check Endpoint"""
    print("\n🧠 === DEMO MEMORY CHECK ENDPOINT ===")
    
    api = build_lumir_api_endpoints()
    
    # Test memory check
    result = api.memory_check_endpoint(
        question="Tôi phù hợp với kiểu trading nào?",
        user_name="Nguyễn Văn A",
        birthday="15/03/1990",
        username="nguyenvana",
        language="vi"
    )
    
    print(f"Result: {result}")
    
    if result["success"]:
        if result["can_answer_from_cache"]:
            print(f"✅ Cache hit! Confidence: {result['cache_confidence']:.2f}")
            print(f"Response: {result['suggested_response']}")
        else:
            print("❌ Cache miss - cần xử lý thêm")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


def demo_question_decomposition():
    """Demo Question Decomposition Endpoint"""
    print("\n🔍 === DEMO QUESTION DECOMPOSITION ENDPOINT ===")
    
    api = build_lumir_api_endpoints()
    
    # Test question decomposition
    result = api.question_decomposition_endpoint(
        question="Tôi thường bị FOMO khi thấy giá tăng",
        user_name="Nguyễn Văn A",
        birthday="15/03/1990",
        excel_path=None,  # Không có Excel file
        language="vi",
        username="nguyenvana"
    )
    
    print(f"Result: {result}")
    
    if result["success"]:
        decomp_data = result["decomposition_result"]
        print(f"✅ Question Type: {decomp_data['question_type']}")
        print(f"✅ Should Call Agents: {decomp_data['should_call_agents']}")
        print(f"✅ Focus Areas: {decomp_data['focus_areas']}")
        print(f"✅ Needs User Info: {decomp_data['needs_user_info']}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


def demo_numerology():
    """Demo Numerology Endpoint"""
    print("\n🔮 === DEMO NUMEROLOGY ENDPOINT ===")
    
    api = build_lumir_api_endpoints()
    
    # Test numerology analysis
    result = api.numerology_endpoint(
        question="Tôi phù hợp với kiểu trading nào?",
        user_name="Nguyễn Văn A",
        birthday="15/03/1990",
        language="vi"
    )
    
    print(f"Result: {result}")
    
    if result["success"]:
        print(f"✅ Numerology Analysis completed!")
        print(f"Response length: {len(result['numerology_response'])} characters")
        print(f"Response preview: {result['numerology_response'][:200]}...")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


def demo_complete_pipeline():
    """Demo Complete Pipeline Endpoint (như test_infer)"""
    print("\n🚀 === DEMO COMPLETE PIPELINE ENDPOINT ===")
    
    api = build_lumir_api_endpoints()
    
    # Test complete pipeline (mô phỏng test_infer)
    result = api.complete_pipeline_endpoint(
        question="Tôi thường bị FOMO khi thấy giá tăng",
        user_name="Nguyễn Văn A",
        birthday="15/03/1990",
        excel_path=None,  # Không có Excel file
        language="vi",
        username="nguyenvana"
    )
    
    print(f"Result: {result}")
    
    if result["success"]:
        print(f"✅ Pipeline completed in {result['processing_time']:.2f}s")
        print(f"✅ Question Type: {result['question_type']}")
        print(f"✅ Response length: {len(result['response'])} characters")
        
        # Kiểm tra context summary (nếu có)
        if "context_summary" in result:
            context = result["context_summary"]
            print(f"✅ Numerology: {'✅' if context['numerology_available'] else '❌'}")
            print(f"✅ Trading: {'✅' if context['trading_available'] else '❌'}")
            print(f"✅ Response Type: {context['response_type']}")
        else:
            print(f"✅ Source: {result.get('source', 'unknown')}")
            print(f"✅ Cache Confidence: {result.get('cache_confidence', 'N/A')}")
        
        print(f"\n💬 RESPONSE:")
        print("-" * 80)
        print(result['response'])
        
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


def demo_memory_management():
    """Demo Memory Management Endpoint"""
    print("\n🧠 === DEMO MEMORY MANAGEMENT ENDPOINT ===")
    
    api = build_lumir_api_endpoints()
    
    # Test memory status
    status_result = api.memory_management_endpoint(
        action="get_status",
        user_name="Nguyễn Văn A",
        birthday="15/03/1990",
        username="nguyenvana"
    )
    
    print(f"Memory Status Result: {status_result}")
    
    if status_result["success"]:
        memory_status = status_result["memory_status"]
        print(f"✅ Memory Status: {memory_status}")
        
        if memory_status.get("status") == "active":
            print(f"✅ Total Entries: {memory_status.get('total_entries', 0)}")
            print(f"✅ Average Confidence: {memory_status.get('average_confidence', 0):.2f}")
    else:
        print(f"❌ Error: {status_result.get('error', 'Unknown error')}")


def demo_multi_turn_conversation():
    """Demo Multi-turn Conversation sử dụng các endpoint"""
    print("\n🔄 === DEMO MULTI-TURN CONVERSATION ===")
    
    api = build_lumir_api_endpoints()
    
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
        print(f"✅ Turn 1 completed in {result1['processing_time']:.2f}s")
        print(f"LUMIR: {result1['response'][:200]}...")
    else:
        print(f"❌ Turn 1 failed: {result1.get('error', 'Unknown error')}")
        return
    
    # Turn 2: Câu hỏi follow-up
    print("\n=== TURN 2 ===")
    question2 = "Vậy tôi nên làm gì cụ thể?"
    
    result2 = api.complete_pipeline_endpoint(
        question2, user_name, birthday, None, language, username
    )
    
    if result2["success"]:
        print(f"✅ Turn 2 completed in {result2['processing_time']:.2f}s")
        print(f"LUMIR: {result2['response'][:200]}...")
    else:
        print(f"❌ Turn 2 failed: {result2.get('error', 'Unknown error')}")
        return
    
    # Kiểm tra memory status
    print("\n=== MEMORY STATUS ===")
    memory_status = api.memory_management_endpoint(
        "get_status", user_name, birthday, username
    )
    
    if memory_status["success"]:
        memory_info = memory_status["memory_status"]
        print(f"✅ Memory entries: {memory_info.get('total_entries', 0)}")
        print(f"✅ Average confidence: {memory_info.get('average_confidence', 0):.2f}")
    else:
        print(f"❌ Memory status failed: {memory_status.get('error', 'Unknown error')}")


def main():
    """Main demo function"""
    print("🚀 LUMIR-AI API ENDPOINTS DEMO")
    print("=" * 80)
    
    try:
        # Demo các endpoint cơ bản
        demo_memory_check()
        demo_question_decomposition()
        demo_numerology()
        demo_memory_management()
        
        # Demo pipeline hoàn chỉnh
        demo_complete_pipeline()
        
        # Demo multi-turn conversation
        demo_multi_turn_conversation()
        
        print("\n🎉 Tất cả demo hoàn thành thành công!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
