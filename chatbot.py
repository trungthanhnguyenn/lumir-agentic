"""
Simple RAG Chatbot for LUMIR using existing orchestrator components.

Pipeline:
1) Embed query with current embedding manager
2) Retrieve top 10 from Qdrant (both collections)
3) Rerank top 10 to top 5 with a cross-encoder
4) Generate answer with LLM (OpenAI-compatible via config.get_openai_llm)

Refusal & safety:
- If query and contexts are off-topic or too weak -> refuse per policy
"""

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

    def _build_prompt(self, question: str, contexts: List[SearchResult]) -> str:
        ctx_text = "\n\n".join((c.payload.get("content", "") or "") for c in contexts)
        
        # Analyze context to create smart suggestions
        context_analysis = self._analyze_context_for_suggestions(contexts)
        
        system = (
            "Bạn là trợ lý của hệ thống LUMIR - Nền tảng hỗ trợ trader giao dịch hiệu quả, sản phẩm của BEQ-Holdings. "
            "Mục tiêu của bạn là giải đáp thắc mắc của khách hàng, đồng thời giới thiệu các tính năng nổi bật của LUMIR một cách chuyên nghiệp, gần gũi. "
            "Hãy luôn trả lời đúng trọng tâm, dựa trên ngữ cảnh cung cấp.\n\n"
            "Quy tắc:\n"
            "- Nếu câu hỏi nằm ngoài kiến thức LUMIR/LUMIR-AI: từ chối lịch sự và giải thích về những gì bạn có thể làm được. Sau đó gợi ý đăng nhập để trò chuyện với LUMIR-AI\n"
            "- Nếu ngữ cảnh chưa đủ để trả lời chính xác: yêu cầu bổ sung thông tin và nêu rõ còn thiếu gì. Hãy sử dụng những câu hỏi gợi mở để khách hàng cung cấp thêm dữ liệu.\n"
            "- Tránh bịa đặt.\n"
            "- Sau khi trả lời, **nếu nội dung câu hỏi** liên quan đến một tính năng của LUMIR, hãy khéo léo giới thiệu tính năng đó và khuyến khích khách hàng trải nghiệm.\n"
            "- Sử dụng tiếng Việt.\n"
            f"- Dựa trên context, gợi ý thêm: {context_analysis}\n"
        )
        user = (
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh liên quan:\n{ctx_text}\n\n"
            "Hãy trả lời theo các quy tắc trên."
        )
        return f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>"
    
    def _analyze_context_for_suggestions(self, contexts: List[SearchResult]) -> str:
        """
        Analyze context to create smart suggestions for user
        """
        if not contexts:
            return "Không có context để phân tích"
        
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
            suggestions.append("• Bạn có thể chia sẻ dữ liệu giao dịch để được phân tích chi tiết")
            suggestions.append("• Hệ thống có thể phân tích hiệu suất và đưa ra khuyến nghị cụ thể")
        
        if numerology_related:
            suggestions.append("• Bạn có thể cung cấp tên và ngày sinh để được phân tích tính cách")
            suggestions.append("• Hệ thống sẽ đưa ra lời khuyên phù hợp với tính cách của bạn")
        
        if "hướng dẫn" in topics:
            suggestions.append("• Hệ thống có hướng dẫn chi tiết cho người mới bắt đầu")
            suggestions.append("• Bạn có thể tham khảo các tài liệu và video hướng dẫn")
        
        if "tính năng" in topics:
            suggestions.append("• Hệ thống có nhiều tính năng nâng cao để khám phá")
            suggestions.append("• Bạn có thể trải nghiệm các tính năng demo")
        
        # General suggestions
        if not suggestions:
            suggestions = [
                "• Bạn có thể đăng nhập để trải nghiệm đầy đủ tính năng",
                "• Hệ thống có thể phân tích dữ liệu cá nhân để đưa ra lời khuyên",
                "• Bạn có thể tham gia cộng đồng trader để học hỏi kinh nghiệm"
            ]
        
        return "\n".join(suggestions)

    def answer(self, question: str) -> Dict[str, Any]:
        # 1) retrieve
        retrieved = self._retrieve(question, k=10)

        # 2) guard: on-topic
        if not self._is_on_topic(question, retrieved):
            return {
                "success": False,
                "reason": "off_topic",
                "message": (
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
                ),
                "retrieved": len(retrieved)
            }

        # 3) rerank top-5
        top5 = self.reranker.rerank(question, retrieved, top_n=10)

        # 4) context sufficiency
        if not self._has_enough_context(top5):
            return {
                "success": False,
                "reason": "insufficient_context",
                "message": (
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
                ),
                "retrieved": len(retrieved),
                "reranked": len(top5)
            }

        # 5) LLM generate
        prompt = self._build_prompt(question, top5)
        llm = self.llm
        output = llm.invoke(prompt)
        text = getattr(output, "content", None) or str(output)
        return {
            "success": True,
            "answer": text,
            "retrieved": len(retrieved),
            "reranked": len(top5)
        }


def build_chatbot(orchestrator: RAGOrchestrator) -> LUMIRChatbot:
    return LUMIRChatbot(orchestrator)


def run_chat(question: str) -> Dict[str, Any]:
    """Convenience function to run a one-shot chat with default orchestrator."""
    from module.rag_orchestrator import RAGOrchestratorFactory
    orchestrator = RAGOrchestratorFactory.create_optimal_orchestrator()
    bot = build_chatbot(orchestrator)
    return bot.answer(question)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py 'your question here'")
        sys.exit(1)
    q = " ".join(sys.argv[1:]).strip()
    resp = run_chat(q)
    print("=== Chatbot Result ===")
    for k, v in resp.items():
        if k == "answer":
            print(f"{k}:\n{v}")
        else:
            print(f"{k}: {v}")


