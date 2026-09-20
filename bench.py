#!/usr/bin/env python3
"""
bench.py - Công cụ Benchmark đo lường chất lượng truy xuất cho Lab 07.
Chiến lược của Thành viên 2: RecursiveChunker
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# Tải file .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


def parse_markdown_file(path: Path) -> tuple[dict[str, str], str]:
    """Tách frontmatter YAML và nội dung phần thân của file .md."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---")
    if len(parts) >= 3:
        raw_fm = re.findall(r"^(\w+):\s*(.+)$", parts[1], re.M)
        metadata = {k: v.strip("\"'") for k, v in raw_fm}
        content = parts[2].strip()
    else:
        metadata = {}
        content = text.strip()
    metadata["doc_id"] = metadata.get("doc_id", path.stem)
    return metadata, content


def load_corpus(data_dir: Path, chunker: Any) -> list[Document]:
    """Đọc mọi file .md trong thư mục và chunk thành danh sách Document."""
    all_chunks: list[Document] = []
    md_files = sorted(data_dir.glob("*.md"))

    for p in md_files:
        metadata, content = parse_markdown_file(p)
        chunks = chunker.chunk(content)
        for i, chunk_text in enumerate(chunks):
            doc_id = f"{metadata['doc_id']}#{i}"
            doc = Document(
                id=doc_id,
                content=chunk_text,
                metadata={**metadata, "chunk_index": i},
            )
            all_chunks.append(doc)
    return all_chunks


def setup_embedder():
    """Khởi tạo embedding backend dựa theo biến môi trường."""
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể khởi tạo GeminiEmbedder ({e}), dùng MockEmbedder thay thế.")
            return _mock_embed
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể khởi tạo OpenAIEmbedder ({e}), dùng MockEmbedder thay thế.")
            return _mock_embed
    elif provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể khởi tạo LocalEmbedder ({e}), dùng MockEmbedder thay thế.")
            return _mock_embed
    return _mock_embed


def mock_llm_for_eval(prompt: str) -> str:
    """Mock LLM trích xuất câu trả lời trực tiếp từ đoạn ngữ cảnh liên quan nhất."""
    # Tìm dòng chứa ngữ cảnh
    if "Ngữ cảnh:" in prompt:
        context_block = prompt.split("Ngữ cảnh:")[1].split("Câu hỏi:")[0].strip()
        lines = [line.strip() for line in context_block.splitlines() if line.strip() and not line.startswith("[")]
        summary = " ".join(lines[:3])
        return f"[Agent] Dựa vào tài liệu: {summary[:250]}..."
    return "[Agent] Đã phân tích theo ngữ cảnh được cung cấp."


# Danh sách 5 câu benchmark thống nhất từ REPORT_NHOM.md
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Người mua có thể gửi yêu cầu Trả hàng/Hoàn tiền cho đơn hàng thông thường trong bao lâu?",
        "gold_answer": "Người mua có thể gửi yêu cầu Trả hàng/Hoàn tiền trong vòng 15 ngày kể từ khi đơn hàng được cập nhật trạng thái 'Giao hàng thành công'",
        "expected_doc": "return-refund-general",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Thời hạn yêu cầu Trả hàng/Hoàn tiền đối với thực phẩm tươi sống và đông lạnh là bao lâu?",
        "gold_answer": "Người mua phải gửi yêu cầu trong vòng 24 giờ kể từ khi đơn hàng được cập nhật trạng thái 'Giao hàng thành công', trừ trường hợp khiếu nại với lý do chưa nhận được hàng.",
        "expected_doc": "return-refund-general",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Với đơn hàng do Người bán tự vận chuyển, thời hạn yêu cầu Trả hàng/Hoàn tiền được tính như thế nào?",
        "gold_answer": "Người mua có thể gửi yêu cầu trong 15 ngày kể từ khi bấm 'Đã nhận được hàng', hoặc 20 ngày kể từ lúc đơn hàng được cập nhật 'Lấy hàng thành công' nếu chưa bấm 'Đã nhận được hàng'.",
        "expected_doc": "return-refund-general",
        "filter": None,
    },
    {
        "id": 4,
        "query": "Người mua cần cung cấp những thông tin hoặc bằng chứng gì khi gửi yêu cầu Trả hàng/Hoàn tiền?",
        "gold_answer": "Người mua cần chọn lý do khiếu nại, mô tả tình trạng sản phẩm và cung cấp hình ảnh hoặc video làm bằng chứng cho yêu cầu Trả hàng/Hoàn tiền.",
        "expected_doc": "return-refund-evidence",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Sau khi nhận thông báo liên quan đến yêu cầu Trả hàng/Hoàn tiền, Người bán phải phản hồi trong bao lâu?",
        "gold_answer": "Người bán cần gửi phản hồi trong vòng 02 ngày lịch kể từ ngày nhận được thông báo của Shopee trong các trường hợp được quy định.",
        "expected_doc": "return-refund-seller",
        "filter": {"audience": "seller"},
    },
]


def main():
    print("=" * 70)
    print(" K4-L3B BENCHMARK RUNNER — THÀNH VIÊN 2 (RecursiveChunker)")
    print("=" * 70)

    # 1. Chọn chiến lược chunker
    chunk_size = 500
    chunker = RecursiveChunker(chunk_size=chunk_size)
    print(f"[*] Chiến lược Chunking : RecursiveChunker (chunk_size={chunk_size})")

    # 2. Đọc và nạp dữ liệu
    data_dir = Path("data/return-refund")
    if not data_dir.exists():
        data_dir = Path("data/ecommerce")
    print(f"[*] Đang nạp dữ liệu từ  : {data_dir}")

    docs = load_corpus(data_dir, chunker)
    print(f"[*] Tổng số chunk đã nạp: {len(docs)} chunks")

    # 3. Khởi tạo Embedder và EmbeddingStore
    embedder = setup_embedder()
    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    print(f"[*] Embedding backend   : {backend_name}")

    store = EmbeddingStore(collection_name="benchmark_collection", embedding_fn=embedder)
    store.add_documents(docs)

    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm_for_eval)

    # 4. Chạy 5 câu benchmark
    print("\n" + "=" * 70)
    print(" KẾT QUẢ CHẠY 5 CÂU HỎI BENCHMARK")
    print("=" * 70)

    total_score = 0

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        gold = item["gold_answer"]
        filter_meta = item["filter"]

        results = store.search_with_filter(query, top_k=3, metadata_filter=filter_meta)

        print(f"\n--- [CÂU HỎI {qid}] ---")
        print(f"Query       : {query}")
        print(f"Filter      : {filter_meta}")
        print(f"Gold Answer : {gold}")
        print("Top-3 Chunks:")

        found_in_top3 = False
        top1_relevant = False

        for rank, res in enumerate(results, 1):
            doc_id = res["metadata"].get("doc_id", "unknown")
            score = res["score"]
            snippet = res["content"][:140].replace("\n", " ")
            print(f"  [{rank}] Score: {score:+.4f} | Nguồn: {doc_id} | Đoạn trích: {snippet}...")

            # Kiểm tra sơ bộ độ liên quan
            if item["expected_doc"] in doc_id:
                found_in_top3 = True
                if rank == 1:
                    top1_relevant = True

        # Điểm đánh giá theo rubric: top-1 liên quan = 2đ, top-2/3 = 1đ, không có = 0đ
        if top1_relevant:
            q_score = 2
        elif found_in_top3:
            q_score = 1
        else:
            q_score = 0
        total_score += q_score

        answer = agent.answer(query, top_k=3)
        print(f"Agent Answer: {answer}")
        print(f"Đánh giá    : {q_score}/2 điểm (Tìm thấy trong Top-3: {found_in_top3})")

    # 5. A/B TEST BẮT BUỘC CHO CÂU HỎI 5 (Audience filter)
    print("\n" + "=" * 70)
    print(" KẾT QUẢ A/B TEST BẮT BUỘC: CÓ FILTER vs KHÔNG FILTER (Câu 5)")
    print("=" * 70)
    q5 = BENCHMARK_QUERIES[4]["query"]

    res_no_filter = store.search_with_filter(q5, top_k=3, metadata_filter=None)
    print("\n[A] KHÔNG LỌC (metadata_filter=None):")
    for rank, r in enumerate(res_no_filter, 1):
        print(f"  Top {rank}: {r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')}) | score={r['score']:.4f}")

    res_with_filter = store.search_with_filter(q5, top_k=3, metadata_filter={"audience": "seller"})
    print("\n[B] CÓ LỌC (metadata_filter={'audience': 'seller'}):")
    for rank, r in enumerate(res_with_filter, 1):
        print(f"  Top {rank}: {r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')}) | score={r['score']:.4f}")

    print("\n" + "=" * 70)
    print(f" TỔNG KẾT BENCHMARK: {total_score}/10 điểm")
    print("=" * 70)


if __name__ == "__main__":
    main()

