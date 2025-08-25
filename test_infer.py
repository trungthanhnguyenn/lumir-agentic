#!/usr/bin/env python3
"""
File test đơn giản để tự test hệ thống LUMIR-AI
Cho phép user nhập input tùy ý và xem kết quả
"""

import sys
import os
from pathlib import Path

# Thêm thư mục gốc vào Python path
sys.path.insert(0, str(Path(__file__).parent))

from agents.multi_agent_orchestrator import build_multi_agent_orchestrator


def auto_detect_test_type():
    """Tự động nhận diện loại test dựa trên input của user"""
    
    print("🎉 LUMIR-AI INTELLIGENT SYSTEM TESTING")
    print("=" * 80)
    print("🚀 Hệ thống chatbot đa agent thông minh")
    print("🔍 Question Decomposition + Intelligent Routing + Synthesis")
    print("🧠 Memory multi-turn chat support")
    print("🌐 Multi-language support")
    print("=" * 80)
    
    print("\n💡 BẠN MUỐN TEST GÌ?")
    print("• Nhập câu hỏi bình thường → Test câu hỏi đơn lẻ")
    print("• Nhập 'chat' → Bắt đầu hội thoại multi-turn")
    print("• Nhập 'status' → Kiểm tra trạng thái hệ thống")
    print("• Nhập 'quit' → Thoát")
    print("=" * 80)
    
    while True:
        try:
            user_input = input("\n🎯 Nhập lệnh hoặc câu hỏi: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'thoát']:
                print("👋 Cảm ơn bạn đã sử dụng LUMIR-AI Testing System!")
                break
                
            elif user_input.lower() == 'chat':
                test_conversation_flow()
                
            elif user_input.lower() == 'status':
                test_system_status()
                
            else:
                # Tự động nhận diện là câu hỏi đơn lẻ
                test_single_question(user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 Thoát chương trình...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def test_single_question(question=None):
    """Test một câu hỏi đơn lẻ"""
    
    print("\n🧪 TEST HỆ THỐNG LUMIR-AI - CÂU HỎI ĐƠN LẺ")
    print("=" * 80)
    
    # Nếu không có question, nhập từ user
    if not question:
        question = input("❓ Nhập câu hỏi của bạn: ").strip()
        if not question:
            print("❌ Câu hỏi không được để trống!")
            return
    
    print(f"❓ Câu hỏi: {question}")
    
    # Nhập thông tin test
    user_name = input("👤 Tên user (Enter để bỏ qua): ").strip() or None
    birthday = input("🎂 Ngày sinh dd/mm/yyyy (Enter để bỏ qua): ").strip() or None
    excel_path = input("📁 Đường dẫn file Excel (Enter để bỏ qua): ").strip() or None
    language = input("🌐 Ngôn ngữ (vi/en, Enter để dùng vi): ").strip() or "vi"
    username = input("🏷️ Username (Enter để bỏ qua): ").strip() or None
    
    print("\n🔄 Bắt đầu test...")
    print("=" * 80)
    
    try:
        # Khởi tạo orchestrator
        orchestrator = build_multi_agent_orchestrator()
        
        # Xử lý câu hỏi
        result = orchestrator.process_user_question(
            question=question,
            user_name=user_name,
            birthday=birthday,
            excel_path=excel_path,
            language=language,
            username=username
        )
        
        # Hiển thị kết quả
        print("\n📊 KẾT QUẢ TEST:")
        print("=" * 80)
        
        if result['success']:
            print(f"✅ Thành công!")
            print(f"⏱️ Thời gian xử lý: {result['processing_time']:.2f}s")
            print(f"🎯 Loại câu hỏi: {result['question_type']}")
            print(f"📊 Loại response: {result['context_summary']['response_type']}")
            print(f"🔮 Numerology: {'✅' if result['context_summary']['numerology_available'] else '❌'}")
            print(f"📈 Trading: {'✅' if result['context_summary']['trading_available'] else '❌'}")
            
            if result['context_summary'].get('needs_user_info'):
                print(f"🔍 Cần thêm thông tin: ✅")
                print(f"💬 Câu hỏi gợi ý: {result['context_summary'].get('suggested_questions', [])}")
            
            print(f"\n💬 RESPONSE:")
            print("-" * 80)
            print(result['response'])
            
        else:
            print(f"❌ Thất bại: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


def test_conversation_flow():
    """Test luồng hội thoại multi-turn"""
    
    print("\n🧪 TEST HỆ THỐNG LUMIR-AI - LUỒNG HỘI THOẠI")
    print("=" * 80)
    
    # Nhập thông tin cơ bản
    user_name = input("👤 Tên user: ").strip()
    if not user_name:
        print("❌ Tên user không được để trống!")
        return
    
    birthday = input("🎂 Ngày sinh dd/mm/yyyy: ").strip()
    if not birthday:
        print("❌ Ngày sinh không được để trống!")
        return
    
    excel_path = input("📁 Đường dẫn file Excel (Enter để bỏ qua): ").strip() or None
    language = input("🌐 Ngôn ngữ (vi/en, Enter để dùng vi): ").strip() or "vi"
    username = input("🏷️ Username: ").strip()
    if not username:
        print("❌ Username không được để trống!")
        return
    
    print(f"\n👤 User: {user_name} ({username})")
    print(f"🎂 Birthday: {birthday}")
    print(f"📁 Excel: {excel_path or 'Không có'}")
    print(f"🌐 Language: {language}")
    print("=" * 80)
    
    try:
        # Khởi tạo orchestrator
        orchestrator = build_multi_agent_orchestrator()
        
        # Bắt đầu hội thoại
        print("💬 Bắt đầu hội thoại (nhập 'quit' để thoát):")
        print("-" * 80)
        
        turn_count = 0
        while True:
            turn_count += 1
            question = input(f"\n🔄 Turn {turn_count} - Bạn: ").strip()
            
            if question.lower() in ['quit', 'exit', 'thoát']:
                print("👋 Kết thúc hội thoại!")
                break
            
            if not question:
                continue
            
            print("🤖 LUMIR-AI: Đang xử lý...")
            
            try:
                result = orchestrator.process_user_question(
                    question=question,
                    user_name=user_name,
                    birthday=birthday,
                    excel_path=excel_path,
                    language=language,
                    username=username
                )
                
                if result['success']:
                    print(f"\n🤖 LUMIR-AI (xử lý trong {result['processing_time']:.2f}s):")
                    print(f"🎯 Loại: {result['question_type']} - {result['context_summary']['response_type']}")
                    print("-" * 60)
                    print(result['response'])
                    
                    # Hiển thị conversation history
                    history = orchestrator.get_conversation_history()
                    print(f"\n📝 Conversation history: {len(history)} turns")
                    
                else:
                    print(f"❌ Lỗi: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 80)
        
        # Tổng kết cuộc hội thoại
        final_history = orchestrator.get_conversation_history()
        print(f"\n📊 TỔNG KẾT CUỘC HỘI THOẠI")
        print("=" * 80)
        print(f"🔄 Tổng số turn: {len(final_history)}")
        print(f"👤 User: {user_name} ({username})")
        print(f"🎂 Birthday: {birthday}")
        print(f"📁 Excel: {excel_path or 'Không có'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_system_status():
    """Test trạng thái hệ thống"""
    
    print("\n🧪 TEST TRẠNG THÁI HỆ THỐNG")
    print("=" * 80)
    
    try:
        orchestrator = build_multi_agent_orchestrator()
        status = orchestrator.get_system_status()
        
        print("📊 TRẠNG THÁI HỆ THỐNG:")
        print(f"• Status: {status['status']}")
        print(f"• Question Decomposer: {status['agents']['question_decomposer']}")
        print(f"• Numerology Agent: {status['agents']['numerology_agent']}")
        print(f"• Trading Agent: {status['agents']['trading_agent']}")
        print(f"• LUMIR Agent: {status['agents']['lumir_agent']}")
        print(f"• Conversation History: {status['conversation_history_length']} turns")
        print(f"• Last Activity: {status['last_activity']}")
        
        # Test conversation history
        history = orchestrator.get_conversation_history()
        if history:
            print(f"\n📝 LỊCH SỬ HỘI THOẠI GẦN ĐÂY:")
            for i, turn in enumerate(history[-3:], 1):
                print(f"Turn {i}: {turn['user_question'][:50]}...")
        else:
            print("\n📝 Chưa có lịch sử hội thoại")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Chạy test tự động nhận diện"""
    auto_detect_test_type()


if __name__ == "__main__":
    main()
