from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

        context_entries: list[str] = []
        for i, r in enumerate(results, 1):
            source = (
                r.get("metadata", {}).get("doc_id")
                or r.get("metadata", {}).get("source")
                or r.get("id", "unknown")
            )
            context_entries.append(f"[{i}] (Nguồn: {source}):\n{r['content']}")
        context_str = "\n\n".join(context_entries)

        prompt = (
            "Dựa trên các đoạn thông tin ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác.\n"
            "Chỉ sử dụng thông tin có trong ngữ cảnh, không tự suy đoán. Hãy trích dẫn số thứ tự nguồn [1], [2], ... tương ứng với thông tin bạn sử dụng.\n"
            "Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi, hãy trả lời rõ: 'Không tìm thấy thông tin trong tài liệu cung cấp.'\n\n"
            f"Ngữ cảnh:\n{context_str}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời:"
        )
        return self.llm_fn(prompt)
