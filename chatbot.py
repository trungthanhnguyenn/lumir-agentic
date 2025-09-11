from typing import List, Dict, Any

from module.rag_orchestrator import RAGOrchestrator, RAGQuery
from module.database.qdrant_manager import SearchResult
from config import get_openai_llm
import sys


class Reranker:
    """Cross-encoder reranker. Falls back to simple cosine if model missing."""

    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder  # lazy import
            # lightweight cross-encoder for MS MARCO
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def rerank(self, query: str, candidates: List[SearchResult], top_n: int = 10) -> List[SearchResult]:
        if not candidates:
            return []
        if self.available:
            pairs = [(query, c.payload.get("content", "")) for c in candidates]
            try:
                scores = self.model.predict(pairs)
                scored = list(zip(candidates, scores))
                scored.sort(key=lambda x: float(x[1]), reverse=True)
                return [s[0] for s in scored[:top_n]]
            except Exception:
                pass
        # fallback: use qdrant score sorting
        sorted_by_score = sorted(candidates, key=lambda x: float(x.score), reverse=True)
        return sorted_by_score[:top_n]


class LUMIRChatbot:
    def __init__(self, orchestrator: RAGOrchestrator):
        self.orchestrator = orchestrator
        self.reranker = Reranker()
        self.llm = get_openai_llm()

    def _retrieve(self, question: str, k: int = 10) -> List[SearchResult]:
        # search both collections, no threshold to maximize recall
        q = RAGQuery(text=question, limit=k, score_threshold=0.0)
        # force both collections
        collections = ["faq", "knowledge_base"]
        all_results: List[SearchResult] = []
        query_emb = self.orchestrator.embedding_manager.get_embedding(question)
        for col in collections:
            res = self.orchestrator.qdrant_manager.search(
                collection_name=col,
                query_vector=query_emb,
                limit=k,
                score_threshold=None,
                with_payload=True,
                with_vectors=False,
            )
            all_results.extend(res)
        # keep best k across unions
        all_results.sort(key=lambda x: float(x.score), reverse=True)
        return all_results[:k]

    def _is_on_topic(self, question: str, contexts: List[SearchResult]) -> bool:
        # Very light heuristic: require at least one context mentioning LUMIR or having reasonable score
        if not contexts:
            return False
        max_score = max(float(c.score) for c in contexts)
        has_brand = any("lumir" in (c.payload.get("content", "") or "").lower() for c in contexts)
        return has_brand or max_score >= 0.2

    def _has_enough_context(self, contexts: List[SearchResult]) -> bool:
        # Require some minimal aggregate token length
        total_chars = sum(len((c.payload.get("content", "") or "")) for c in contexts)
        return total_chars >= 300

    def _analyze_user_context(self, question: str, user_name: str, user_birthday: str, username: str, trading_data: bool) -> Dict[str, Any]:
        """
        Analyze user context to understand login status and available information
        """
        # Check login status
        is_logged_in = bool(username and username.strip())
        
        # Check personal info availability
        has_personal_info = bool(user_name and user_name.strip() and user_birthday and user_birthday.strip())
        
        # Check trading data availability
        has_trading_info = bool(trading_data)
        
        # Analyze question type to determine what information is needed
        question_lower = question.lower()
        
        # Check if question needs trading analysis
        # needs_trading_data = any(word in question_lower for word in [
        #     "trading", "giao dịch", "lệnh", "profit", "lỗ", "win rate", "hiệu suất", 
        #     "thị trường", "cổ phiếu", "forex", "crypto", "đầu tư", "đầu tư"
        # ])
        
        # # Check if question needs personal analysis (numerology, psychology)
        # needs_personal_info = any(word in question_lower for word in [
        #     "tính cách", "tâm lý", "cảm xúc", "thần số học", "numerology", 
        #     "bản thân", "tôi", "mình", "cá nhân", "phù hợp", "nên làm gì",
        #     "kiểm soát", "quản lý", "cải thiện", "thay đổi"
        # ])
        
        # Determine if current info is sufficient for the question
        info_sufficient = True
        missing_info = []
        
        if not has_trading_info:
            info_sufficient = False
            missing_info.append("dữ liệu giao dịch")
            
        if not has_personal_info:
            info_sufficient = False
            missing_info.append("thông tin cá nhân (tên và ngày sinh)")
        
        return {
            "is_logged_in": is_logged_in,
            "has_personal_info": has_personal_info,
            "has_trading_info": has_trading_info,
            # "needs_trading_data": needs_trading_data,
            # "needs_personal_info": needs_personal_info,
            "info_sufficient": info_sufficient,
            "missing_info": missing_info,
            "question_type": self._classify_question_type(question_lower)
        }
    
    def _classify_question_type(self, question_lower: str) -> str:
        """Classify question type to determine appropriate response strategy"""
        if any(word in question_lower for word in ["trading", "giao dịch", "lệnh", "profit", "lỗ"]):
            return "trading_related"
        elif any(word in question_lower for word in ["tính cách", "tâm lý", "cảm xúc", "thần số học"]):
            return "personal_analysis"
        elif any(word in question_lower for word in ["hệ thống", "tính năng", "hướng dẫn", "làm thế nào"]):
            return "system_info"
        else:
            return "general"
    
    def _generate_smart_suggestions(self, user_context: Dict[str, Any], question: str, language: str = "vi") -> str:
        """
        Generate intelligent suggestions based on user context and question
        """
        suggestions = []
        
        if not user_context["is_logged_in"]:
            # User not logged in
            if user_context["question_type"] == "trading_related":
                if language == "en":
                    suggestions.append("**For detailed trading consultation:** Log in and provide your trading data")
                else:  # Vietnamese
                    suggestions.append("**Để được tư vấn trading chi tiết:** Đăng nhập và cung cấp dữ liệu giao dịch của bạn")
            elif user_context["question_type"] == "personal_analysis":
                if language == "en":
                    suggestions.append("**For personality analysis:** Log in and provide your name and birth date")
                else:  # Vietnamese
                    suggestions.append("**Để được phân tích tính cách:** Đăng nhập và cung cấp tên cùng ngày sinh")
            else:
                if language == "en":
                    suggestions.append("**For full experience:** Log in to the LUMIR system")
                    suggestions.append("**Free account:** Create an account to use advanced features")
                else:  # Vietnamese
                    suggestions.append("**Để trải nghiệm đầy đủ:** Đăng nhập vào hệ thống LUMIR")
                    suggestions.append("**Tài khoản miễn phí:** Tạo tài khoản để sử dụng các tính năng nâng cao")
        else:
            # User is logged in but may be missing some info
            if user_context["question_type"] == "trading_related" and not user_context["has_trading_info"]:
                if language == "en":
                    suggestions.append("**Trading data needed:** Upload Excel file or connect MT4/MT5 account for detailed analysis")
                else:  # Vietnamese
                    suggestions.append("**Cần dữ liệu giao dịch:** Upload file Excel hoặc kết nối tài khoản MT4/MT5 để được phân tích chi tiết")
            elif user_context["question_type"] == "personal_analysis" and not user_context["has_personal_info"]:
                if language == "en":
                    suggestions.append("**Personal info needed:** Update name and birth date in profile for personality analysis")
                else:  # Vietnamese
                    suggestions.append("**Cần thông tin cá nhân:** Cập nhật tên và ngày sinh trong hồ sơ để được phân tích tính cách")
            
            if user_context["info_sufficient"]:
                if language == "en":
                    suggestions.append("**Complete information:** You can use all LUMIR features")
                else:  # Vietnamese
                    suggestions.append("**Thông tin đầy đủ:** Bạn có thể sử dụng đầy đủ các tính năng của LUMIR")
        
        # Add general helpful suggestions
        if user_context["question_type"] == "system_info":
            if language == "en":
                suggestions.append("**Documentation:** See detailed documentation in Help section")
            else:  # Vietnamese
                suggestions.append("**Tài liệu hướng dẫn:** Xem thêm tài liệu chi tiết trong phần Help")
        
        if not suggestions:
            if language == "en":
                suggestions = [
                    "**Explore more:** The LUMIR system has many interesting features for you to explore",
                    "**Support:** If you need additional support, please contact our customer care team"
                ]
            else:  # Vietnamese
                suggestions = [
                    "**Khám phá thêm:** Hệ thống LUMIR có nhiều tính năng thú vị để bạn khám phá",
                    "**Hỗ trợ:** Nếu cần hỗ trợ thêm, hãy liên hệ đội ngũ chăm sóc khách hàng"
                ]
        
        return "\n".join(suggestions)

    def _build_prompt(self, question: str, contexts: List[SearchResult], user_context: Dict[str, Any], user_name: str, language: str) -> str:
        """
        Build prompt for LLM to generate response
        """
        # Extract relevant context
        context_summary = self._extract_relevant_context(question, contexts, language)
        ctx_text = "\n\n".join((c.payload.get("content", "") or "") for c in contexts)
        
        # Analyze context to create smart suggestions
        context_analysis = self._analyze_context_for_suggestions(contexts, language)
        
        # Multilingual user status
        user_display_name = user_name.strip() if user_name and user_name.strip() else "Unknown User"
        
        if language == "en":
            user_status = f"""
**User Status:**
- Name: {user_display_name}
- Logged in: {'Yes' if user_context['is_logged_in'] else 'No'}
- Has personal info: {'Yes' if user_context['has_personal_info'] else 'No'}
- Has trading data: {'Yes' if user_context['has_trading_info'] else 'No'}
- Question type: {user_context['question_type']}
- Info sufficient: {'Sufficient' if user_context['info_sufficient'] else 'Missing: ' + ', '.join(user_context['missing_info'])}
"""
        else:  # Default Vietnamese
            user_status = f"""
**Thông tin người dùng:**
- Tên: {user_display_name}
- Đã đăng nhập: {'Có' if user_context['is_logged_in'] else 'Chưa'}
- Có thông tin cá nhân: {'Có' if user_context['has_personal_info'] else 'Chưa'}
- Có dữ liệu giao dịch: {'Có' if user_context['has_trading_info'] else 'Chưa'}
- Loại câu hỏi: {user_context['question_type']}
- Thông tin đủ: {'Đủ' if user_context['info_sufficient'] else 'Thiếu: ' + ', '.join(user_context['missing_info'])}
"""
        
        # Multilingual system prompt
        if language == "en":
            system = (
                "You are LUMIR assistant of the LUMIR system - An effective trading support platform, a product of BEQ-Holdings. "
                "Your goal is to answer customer questions and introduce LUMIR's outstanding features in a professional and friendly manner. "
                "Always answer to the point, based on the provided context.\n\n"
                "Rules:\n"
                "- Use the name **LUMIR** when referring to yourself\n"
                "- **PERSONALIZATION**: If user name is provided in the user information, address them naturally by name. If not provided, use friendly general greeting.\n"
                "- If the question is outside LUMIR/LUMIR-AI knowledge: politely decline and explain what you can do. Then suggest logging in to chat with LUMIR-AI\n"
                "- If the context is not enough to answer accurately: ask for additional information and specify what is missing. Use open-ended questions to encourage customers to provide more data.\n"
                "- Avoid making things up.\n"
                "- After answering, **if the question content** relates to a LUMIR feature, skillfully introduce that feature and encourage customers to experience it.\n"
                f"- Response language: {language}\n"
                f"- Based on context, additional suggestions: {context_analysis}\n"
                f"- User information: {user_status}\n"
                "- **IMPORTANT**: Based on user status to provide appropriate suggestions. Don't suggest logging in if already logged in.\n\n"
                "**Handle missing information cases:**\n"
                "- If user hasn't provided enough information to answer the question:\n"
                "  1. Use available context to explain what LUMIR can help with\n"
                "  2. Explain why more information is needed\n"
                "  3. Give specific guidance on what user needs to do next\n"
                "  4. Encourage logging in and providing necessary information\n"
                "- Be natural, don't show raw context data, create understandable responses based on that context"
            )
        else:  # Default Vietnamese
            system = (
                "Bạn là trợ lý của hệ thống LUMIR - Nền tảng hỗ trợ trader giao dịch hiệu quả, sản phẩm của BEQ-Holdings. "
                "Mục tiêu của bạn là giải đáp thắc mắc của khách hàng, đồng thời giới thiệu các tính năng nổi bật của LUMIR một cách chuyên nghiệp, gần gũi. "
                "Hãy luôn trả lời đúng trọng tâm, dựa trên ngữ cảnh cung cấp.\n\n"
                "Quy tắc:\n"
                "- Sử dụng cách xưng hô bằng tên **LUMIR**\n"
                "- **CÁ NHÂN HÓA**: Nếu có tên người dùng trong thông tin, hãy xưng hô tự nhiên bằng tên. Nếu chưa có thì dùng cách chào thân thiện chung.\n"
                "- Nếu câu hỏi nằm ngoài kiến thức LUMIR/LUMIR-AI: từ chối lịch sự và giải thích về những gì bạn có thể làm được. Sau đó gợi ý đăng nhập để trò chuyện với LUMIR-AI\n"
                "- Nếu ngữ cảnh chưa đủ để trả lời chính xác: yêu cầu bổ sung thông tin và nêu rõ còn thiếu gì. Hãy sử dụng những câu hỏi gợi mở để khách hàng cung cấp thêm dữ liệu.\n"
                "- Tránh bịa đặt.\n"
                "- Sau khi trả lời, **nếu nội dung câu hỏi** liên quan đến một tính năng của LUMIR, hãy khéo léo giới thiệu tính năng đó và khuyến khích khách hàng trải nghiệm.\n"
                f"- Ngôn ngữ yêu cầu: {language}\n"
                f"- Dựa trên context, gợi ý thêm: {context_analysis}\n"
                f"- Thông tin người dùng: {user_status}\n"
                "- **QUAN TRỌNG**: Dựa vào trạng thái người dùng để đưa ra gợi ý phù hợp. Không gợi ý đăng nhập nếu đã đăng nhập rồi.\n\n"
                "**Xử lý trường hợp thiếu thông tin:**\n"
                "- Nếu user chưa cung cấp đủ thông tin để trả lời câu hỏi, hãy:\n"
                "  1. Sử dụng context có sẵn để giải thích LUMIR có thể giúp gì\n"
                "  2. Giải thích tại sao cần thêm thông tin\n"
                "  3. Hướng dẫn cụ thể user cần làm gì tiếp theo\n"
                "  4. Khuyến khích đăng nhập và cung cấp thông tin cần thiết\n"
                "- Hãy tự nhiên, không hiển thị raw context data, mà tạo response dễ hiểu dựa trên context đó"
            )
        # Multilingual user section
        if language == "en":
            user = (
                f"Question: {question}\n\n"
                f"Related context:\n{ctx_text}\n\n"
                "Please answer according to the rules above and based on the requested language."
            )
        else:  # Vietnamese
            user = (
                f"Câu hỏi: {question}\n\n"
                f"Ngữ cảnh liên quan:\n{ctx_text}\n\n"
                "Hãy trả lời theo các quy tắc trên và dựa vào ngôn ngữ yêu cầu."
            )
        return f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>"
    
    def _analyze_context_for_suggestions(self, contexts: List[SearchResult], language: str = "vi") -> str:
        """
        Analyze context to create smart suggestions for user
        """
        if not contexts:
            return "Không có context để phân tích" if language == "vi" else "No context to analyze"
        
        # Collect information from context
        topics = set()
        features = set()
        trading_related = False
        numerology_related = False
        
        for ctx in contexts:
            content = ctx.payload.get("content", "").lower()
            
            # Detect topics
            if any(word in content for word in ["trading", "giao dịch", "thị trường", "lệnh"]):
                trading_related = True
                topics.add("trading")
            
            if any(word in content for word in ["thần số học", "numerology", "tính cách", "tâm lý"]):
                numerology_related = True
                topics.add("numerology")
            
            if any(word in content for word in ["hướng dẫn", "tutorial", "bắt đầu", "đầu tiên"]):
                topics.add("hướng dẫn")
            
            if any(word in content for word in ["tính năng", "feature", "chức năng"]):
                topics.add("tính năng")
        
        # Create suggestions based on context
        suggestions = []
        
        if trading_related:
            if language == "en":
                suggestions.append("• You can share trading data for detailed analysis")
                suggestions.append("• The system can analyze performance and provide specific recommendations")
            else:  # Vietnamese
                suggestions.append("• Bạn có thể chia sẻ dữ liệu giao dịch để được phân tích chi tiết")
                suggestions.append("• Hệ thống có thể phân tích hiệu suất và đưa ra khuyến nghị cụ thể")
        
        if numerology_related:
            if language == "en":
                suggestions.append("• You can provide name and birth date for personality analysis")
                suggestions.append("• The system will provide advice suitable for your personality")
            else:  # Vietnamese
                suggestions.append("• Bạn có thể cung cấp tên và ngày sinh để được phân tích tính cách")
                suggestions.append("• Hệ thống sẽ đưa ra lời khuyên phù hợp với tính cách của bạn")
        
        if "hướng dẫn" in topics:
            if language == "en":
                suggestions.append("• The system has detailed guides for beginners")
                suggestions.append("• You can refer to documentation and tutorial videos")
            else:  # Vietnamese
                suggestions.append("• Hệ thống có hướng dẫn chi tiết cho người mới bắt đầu")
                suggestions.append("• Bạn có thể tham khảo các tài liệu và video hướng dẫn")
        
        if "tính năng" in topics:
            if language == "en":
                suggestions.append("• The system has many advanced features to explore")
                suggestions.append("• You can try demo features")
            else:  # Vietnamese
                suggestions.append("• Hệ thống có nhiều tính năng nâng cao để khám phá")
                suggestions.append("• Bạn có thể trải nghiệm các tính năng demo")
        
        # General suggestions
        if not suggestions:
            if language == "en":
                suggestions = [
                    "• The system can analyze personal data to provide advice",
                    "• You can join the trading community to learn from experience"
                ]
            else:  # Vietnamese
                suggestions = [
                    "• Hệ thống có thể phân tích dữ liệu cá nhân để đưa ra lời khuyên",
                    "• Bạn có thể tham gia cộng đồng trader để học hỏi kinh nghiệm"
                ]
        
        return "\n".join(suggestions)

    def answer(self, question: str, user_name: str, user_birthday: str, username: str, trading_data: bool, language: str = "vi") -> Dict[str, Any]:
        # 1) retrieve
        retrieved = self._retrieve(question, k=10)

        # 2) guard: on-topic
        if not self._is_on_topic(question, retrieved):
            if language == "en":
                message = (
                    "Sorry, this question is outside my knowledge. I am only designed to answer "
                    "information about LUMIR and LUMIR-AI systems.\n\n"
                    "**I can help you with:**\n"
                    "• Information about LUMIR system and features\n"
                    "• Usage guides and getting started\n"
                    "• Trading and numerology questions (when personal data is available)\n"
                    "• Trading psychology consultation\n\n"
                    "**For personalized consultation:**\n"
                    "• Provide name and birth date for personality analysis\n"
                    "• Upload trading data for performance analysis\n"
                    "• Log in to experience full features"
                )
            else:  # Vietnamese
                message = (
                    "Xin lỗi, câu hỏi nằm ngoài kiến thức của tôi. Tôi chỉ được thiết kế để trả lời "
                    "các thông tin về hệ thống LUMIR và LUMIR-AI.\n\n"
                    "**Tôi có thể giúp bạn với:**\n"
                    "• Thông tin về hệ thống LUMIR và các tính năng\n"
                    "• Hướng dẫn sử dụng và bắt đầu\n"
                    "• Câu hỏi về trading và thần số học (khi có dữ liệu cá nhân)\n"
                    "• Tư vấn về tâm lý giao dịch\n\n"
                    "**Để được tư vấn cá nhân hóa:**\n"
                    "• Cung cấp tên và ngày sinh để phân tích tính cách\n"
                    "• Upload dữ liệu giao dịch để phân tích hiệu suất\n"
                    "• Đăng nhập để trải nghiệm đầy đủ tính năng"
                )
            
            return {
                "success": False,
                "reason": "off_topic",
                "message": message,
                "retrieved": len(retrieved)
            }

        # 3) rerank top-5
        top5 = self.reranker.rerank(question, retrieved, top_n=10)

        # 4) Analyze user context
        user_context = self._analyze_user_context(question, user_name, user_birthday, username, trading_data)
        
        # 5) Check if we need to handle needs_user_info case
        if not user_context["info_sufficient"] and user_context["question_type"] in ["trading_related", "personal_analysis"]:
            # Let LLM handle this naturally with the improved prompt
            # The prompt now includes specific guidance for handling insufficient user information
            pass  # Continue to normal LLM processing
        
        # 6) Check context sufficiency for other cases
        if not self._has_enough_context(top5):
            if language == "en":
                message = (
                    "The current context is not sufficient to answer accurately. Please provide more specific information "
                    "(e.g., feature, documentation page, or chapter in the Handbook).\n\n"
                    "**I can help you with:**\n"
                    "• General information about LUMIR system\n"
                    "• Basic guides\n"
                    "• Explanation of trading and numerology concepts\n\n"
                    "**For more detailed consultation:**\n"
                    "• Provide personal information (name, birth date)\n"
                    "• Upload trading data\n"
                    "• Ask more specific questions about features you're interested in"
                )
            else:  # Vietnamese
                message = (
                    "Ngữ cảnh hiện tại chưa đủ để trả lời chính xác. Vui lòng bổ sung thông tin cụ thể hơn "
                    "(ví dụ: tính năng, trang tài liệu, hoặc chương mục trong Handbook).\n\n"
                    "**Tôi có thể giúp bạn với:**\n"
                    "• Thông tin chung về hệ thống LUMIR\n"
                    "• Hướng dẫn cơ bản\n"
                    "• Giải thích các khái niệm trading và thần số học\n\n"
                    "**Để được tư vấn chi tiết hơn:**\n"
                    "• Cung cấp thông tin cá nhân (tên, ngày sinh)\n"
                    "• Upload dữ liệu giao dịch\n"
                    "• Đặt câu hỏi cụ thể hơn về tính năng bạn quan tâm"
                )
            
            return {
                "success": False,
                "reason": "insufficient_context",
                "message": message,
                "retrieved": len(retrieved),
                "reranked": len(top5)
            }

        # 7) Generate smart suggestions
        smart_suggestions = self._generate_smart_suggestions(user_context, question, language)

        # 8) LLM generate with user context
        prompt = self._build_prompt(question, top5, user_context, user_name, language)
        llm = self.llm
        output = llm.invoke(prompt)
        text = getattr(output, "content", None) or str(output)
        
        # 9) Append smart suggestions if needed
        final_answer = text
        if not user_context["info_sufficient"] and user_context["question_type"] in ["trading_related", "personal_analysis"]:
            final_answer += f"\n\n{smart_suggestions}"
        
        return {
            "success": True,
            "answer": final_answer,
            "retrieved": len(retrieved),
            "reranked": len(top5),
            "user_context": user_context,
            "suggestions": smart_suggestions,
            "response_type": "full_response"
        }

    def _extract_relevant_context(self, question: str, contexts: List[SearchResult], language: str = "vi") -> str:
        """
        Extract relevant context information to provide a meaningful response
        even when user needs to provide more information
        """
        if not contexts:
            if language == "en":
                return "I understand you're interested in improving trading skills and emotion management."
            else:  # Vietnamese
                return "Tôi hiểu bạn đang quan tâm đến việc cải thiện kỹ năng giao dịch và quản lý cảm xúc."
        
        # Find most relevant context
        relevant_contexts = []
        for ctx in contexts[:3]:  # Top 3 most relevant
            content = ctx.payload.get("content", "")
            if content and len(content) > 50:  # Only meaningful content
                relevant_contexts.append(content)
        
        if relevant_contexts:
            context_summary = " ".join(relevant_contexts)
            # Clean up and make it more natural
            context_summary = context_summary.replace("\n", " ").replace("  ", " ")
            if language == "en":
                return f"Based on LUMIR knowledge, {context_summary.lower()}"
            else:  # Vietnamese
                return f"Dựa trên kiến thức về LUMIR, {context_summary.lower()}"
        else:
            if language == "en":
                return "I understand you're interested in improving trading skills and emotion management."
            else:  # Vietnamese
                return "Tôi hiểu bạn đang quan tâm đến việc cải thiện kỹ năng giao dịch và quản lý cảm xúc."


def build_chatbot(orchestrator: RAGOrchestrator) -> LUMIRChatbot:
    return LUMIRChatbot(orchestrator)


def run_chat(question: str, user_name: str = "", user_birthday: str = "", username: str = "", trading_data: bool = False, language: str = "vi") -> Dict[str, Any]:
    """Convenience function to run a one-shot chat with default orchestrator."""
    from module.rag_orchestrator import RAGOrchestratorFactory
    orchestrator = RAGOrchestratorFactory.create_optimal_orchestrator()
    bot = build_chatbot(orchestrator)
    return bot.answer(question, user_name, user_birthday, username, trading_data, language)


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python chatbot.py 'your question here'")
#         sys.exit(1)
#     q = " ".join(sys.argv[1:]).strip()
#     resp = run_chat(q)
#     print("=== Chatbot Result ===")
#     for k, v in resp.items():
#         if k == "answer":
#             print(f"{k}:\n{v}")
#         else:
#             print(f"{k}: {v}")


