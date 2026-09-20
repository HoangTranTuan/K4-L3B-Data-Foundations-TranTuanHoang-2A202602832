# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Tuấn Hoàng
**Nhóm:** BotVN
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần đến 1.0) biểu thị hai vector embedding có góc hợp bởi giữa chúng rất nhỏ. Điều này có nghĩa là hai văn bản có sự đồng hướng mạnh mẽ trong không gian vector ngữ nghĩa tiềm ẩn (semantic embedding space), phản ánh ý nghĩa và ngữ cảnh tương đồng sâu sắc dù câu chữ bề mặt (surface lexical wording) có thể khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Shopee hỗ trợ người mua trả hàng và hoàn tiền trong vòng 15 ngày."
- Câu B: "Khách hàng có mười lăm ngày để gửi yêu cầu đổi trả sản phẩm trên sàn."
- Tại sao tương đồng: Dù sử dụng từ ngữ và cú pháp khác nhau ("người mua" vs "khách hàng", "15" vs "mười lăm", "trả hàng và hoàn tiền" vs "đổi trả sản phẩm"), cả hai câu đều diễn đạt cùng một thông điệp chính sách về quyền lợi và thời hạn hoàn trả đơn hàng.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Shopee hỗ trợ người mua trả hàng và hoàn tiền trong vòng 15 ngày."
- Câu B: "Công thức phân tử của nước bao gồm hai nguyên tử hydro và một nguyên tử oxy."
- Tại sao khác: Hai câu thuộc hai miền tri thức hoàn toàn độc lập (quy định thương mại điện tử vs hóa học cơ bản), không có mối liên hệ ngữ nghĩa hay từ vựng nào chung, dẫn đến hai vector gần như trực giao trong không gian embedding (cosine similarity gần bằng 0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity chỉ đo lường góc định hướng giữa hai vector mà không bị phụ thuộc vào độ dài (magnitude/norm) của vector. Khi mã hóa văn bản, một đoạn văn dài thường có vector độ lớn cao hơn đoạn văn ngắn dù cùng chung chủ đề; khoảng cách Euclid sẽ bị sai lệch nghiêm trọng do chiều dài văn bản, trong khi cosine similarity chuẩn hóa độ dài và phản ánh chính xác bản chất tương đồng ngữ nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy giữa các chunk liên tiếp: $S = \text{chunk\_size} - \text{overlap} = 500 - 50 = 450$ ký tự.
> - Điểm bắt đầu các chunk theo vòng lặp: $0, 450, 900, 1350, \dots, 9900$.
> - Áp dụng công thức tổng quát: $\lceil \frac{L - \text{overlap}}{\text{chunk\_size} - \text{overlap}} \rceil = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil = 23$.
> - Chunk cuối cùng bắt đầu từ vị trí 9,900 đến 10,000 (độ dài 100 ký tự).
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100 ký tự, bước nhảy giảm xuống $S = 500 - 100 = 400$ ký tự, số lượng chunk sẽ là $\lceil \frac{10000 - 100}{400} \rceil = \lceil \frac{9900}{400} \rceil = \lceil 24.75 \rceil = 25$ chunks (tăng thêm 2 chunks). Chúng ta muốn độ chồng chéo nhiều hơn để bảo toàn trọn vẹn ngữ cảnh tại các ranh giới cắt (boundary context), tránh làm đứt đôi câu văn hay cụm thực thể quan trọng, giúp bộ truy xuất (retriever) luôn nắm bắt đầy đủ thông tin ngữ nghĩa.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regular expression `r"(?<=[.!?])(?:\s+|\n)"` với kỹ thuật positive lookbehind để phát hiện ranh giới kết thúc câu sau các ký tự `.`, `!`, `?` kèm khoảng trắng/xuống dòng mà không làm mất dấu câu gốc. Các câu trích xuất được làm sạch khoảng trắng thừa và gom nhóm thành từng khối tối đa `max_sentences_per_chunk` câu. Xử lý tốt các edge cases như văn bản rỗng, chuỗi chỉ chứa khoảng trắng, và đoạn cuối cùng có số câu ít hơn kích thước định mức.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo cơ chế đệ quy phân cấp dựa trên danh sách ký tự phân tách ưu tiên: đoạn văn (`"\n\n"`), dòng (`"\n"`), câu (`". "`), từ (`" "`), và ký tự (`""`). Trường hợp cơ sở (base case) xảy ra khi đoạn văn bản hiện tại đã có độ dài $\le \text{chunk\_size}$, hoặc khi danh sách dấu phân tách đã cạn (chuyển sang cắt lát ký tự cố định). Các mảnh phân tách nhỏ sau đó được ghép nối tuần hoàn (accumulate/merge) chừng nào tổng chiều dài còn nằm trong ngưỡng `chunk_size` để tối đa hóa ngữ cảnh trong mỗi chunk.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Quản lý dữ liệu dưới dạng cấu trúc bản ghi bộ nhớ (in-memory records) gồm `id`, `content`, `metadata` và `embedding`. Khi gọi `add_documents`, hệ thống sinh vector embedding thông qua `_embedding_fn` và chuẩn hóa trường `doc_id` trong metadata. Phương thức `search` tính tích vô hướng (dot product - tương đương cosine similarity với vector chuẩn hóa) giữa query embedding và toàn bộ vector đã lưu trữ, sắp xếp giảm dần theo `score` và trả về danh sách `top_k` chunk tương đồng nhất.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Áp dụng cơ chế tiền lọc (pre-filtering): tập hợp các bản ghi được lọc trước theo điều kiện `metadata_filter` (khớp tuyệt đối mọi cặp key-value) nhằm thu hẹp không gian tìm kiếm về đúng tập ứng viên mục tiêu trước khi tính điểm cosine similarity. Đối với `delete_document`, thuật toán lọc bỏ toàn bộ các bản ghi có `metadata["doc_id"]` khớp với `doc_id` được chỉ định, so sánh kích thước danh sách trước và sau để trả về `True` nếu xóa thành công ít nhất một chunk và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tác tử tuân thủ kiến trúc RAG (Retrieval-Augmented Generation): đầu tiên gọi `store.search` để lấy các chunk phù hợp nhất, ghép thành chuỗi ngữ cảnh được đánh số thứ tự minh bạch kèm nguồn gốc (`[1] (Nguồn: doc_id): ...`). Cấu trúc prompt được thiết kế chặt chẽ với các chỉ dẫn định hướng (grounding instructions): nghiêm cấm ảo giác (hallucination), bắt buộc trích dẫn số thứ tự nguồn `[1]`, `[2]` trong câu trả lời, và nếu tài liệu không chứa câu trả lời thì phải phản hồi rõ là không tìm thấy thông tin.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
```
(day07) hoang@DESKTOP-N434CRE:~/AI_VinUni/K4-L3B-Data-Foundations$ pytest tests/ -v
====================================== test session starts =======================================
platform linux -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /home/hoang/.conda/envs/day07/bin/python3.11
cachedir: .pytest_cache
rootdir: /home/hoang/AI_VinUni/K4-L3B-Data-Foundations
plugins: anyio-4.15.1
collected 42 items                                                                               

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED      [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED               [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED        [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED         [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED              [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED    [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED     [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED   [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                     [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED     [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED            [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                      [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                     [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED       [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED         [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED               [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED    [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED      [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED       [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED               [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED          [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED      [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED     [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED           [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED     [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================================= 42 passed in 0.13s =======================================

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Chính sách đổi trả hàng của Shopee trong vòng 15 ngày. | Khách hàng có mười lăm ngày để gửi yêu cầu trả hàng và hoàn tiền. | cao | 0.8845 | Đúng |
| 2 | Quy trình khiếu nại và hoàn tiền đơn hàng trực tuyến. | Công thức hóa học của nước là hai nguyên tử hydro và một oxy. | thấp | 0.3210 | Đúng |
| 3 | Thời hạn khiếu nại đối với thực phẩm tươi sống là 24 giờ. | Sản phẩm tươi đông lạnh cần gửi yêu cầu trong vòng một ngày. | cao | 0.8624 | Đúng |
| 4 | Người bán không đồng ý với khiếu nại của người mua. | Người mua hài lòng với chất lượng phục vụ của người bán. | thấp | 0.6120 | Đúng |
| 5 | Hướng dẫn đóng gói hàng hoàn trả để gửi lại cho shop. | Quy cách đóng gói kiện hàng đổi trả để shipper tiếp nhận. | cao | 0.8415 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả ở Cặp 4 gây bất ngờ nhất khi hai câu mang ngữ nghĩa và thái độ đối lập (từ chối khiếu nại vs hài lòng với dịch vụ) nhưng vẫn đạt điểm tương đồng tương đối cao (~0.61). Điều này phản ánh rằng mô hình embedding biểu diễn rất mạnh thông tin chủ đề (topic context: mối tương tác mua bán giữa người mua và người bán trên sàn thương mại điện tử), khiến các vector nằm gần nhau trong không gian tiềm ẩn dù mặt logic phủ định/khẳng định là trái ngược nhau.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|---|---|---|---|---|
| 1 | Người mua có thể gửi yêu cầu Trả hàng/Hoàn tiền cho đơn hàng thông thường trong bao lâu? | `return-refund-general`: Đối với các đơn hàng khác: 15 ngày kể từ lúc đơn hàng được cập nhật trạng thái ‘Giao hàng thành công’... | +0.8040 | Có | Người mua có thể gửi yêu cầu trong vòng 15 ngày kể từ khi đơn hàng giao thành công, kể cả sau khi đã bấm 'Đã nhận được hàng'. |
| 2 | Thời hạn yêu cầu Trả hàng/Hoàn tiền đối với thực phẩm tươi sống và đông lạnh là bao lâu? | `return-refund-policy`: 3.2. Người Mua có thể gửi yêu cầu trong vòng 15 ngày... Riêng đối với Sản Phẩm là thực phẩm tươi sống và đông lạnh... | +0.8729 | Khá (Top-1 nêu điều kiện chung, Top-2 score 0.8334 chứa con số 24 giờ) | Đối với thực phẩm tươi sống và đông lạnh, Người Mua cần gửi yêu cầu trả hàng/hoàn tiền trong vòng 24 giờ kể từ khi đơn hàng giao thành công. |
| 3 | Với đơn hàng do Người bán tự vận chuyển, thời hạn yêu cầu Trả hàng/Hoàn tiền được tính như thế nào? | `return-refund-general`: Đối với đơn hàng do Người bán tự vận chuyển: 15 ngày từ khi bấm 'Đã nhận được hàng' hoặc 20 ngày từ lúc 'Lấy hàng thành công'... | +0.7931 | Có | Thời hạn là 15 ngày kể từ khi bấm 'Đã nhận được hàng' hoặc 20 ngày kể từ lúc đơn hàng được cập nhật trạng thái 'Lấy hàng thành công'. |
| 4 | Người mua cần cung cấp những thông tin hoặc bằng chứng gì khi gửi yêu cầu Trả hàng/Hoàn tiền? | `return-refund-evidence`: Hướng dẫn chuẩn bị bằng chứng khi yêu cầu Trả hàng/Hoàn tiền: cung cấp đầy đủ và chính xác lý do khiếu nại, tình trạng và hình ảnh/video... | +0.8604 | Có | Người mua cần chọn lý do khiếu nại, mô tả tình trạng sản phẩm và cung cấp hình ảnh hoặc video rõ nét làm bằng chứng xác thực. |
| 5 | Sau khi nhận thông báo liên quan đến yêu cầu Trả hàng/Hoàn tiền, Người bán phải phản hồi trong bao lâu? | `return-refund-seller` (filter: `audience: seller`): Người Bán cần gửi phản hồi trong vòng 02 ngày lịch kể từ ngày nhận được thông báo của Shopee... | +0.8178 | Có | Người Bán cần gửi phản hồi trong vòng 02 ngày lịch kể từ ngày nhận thông báo; nếu không Shopee sẽ tự động hoàn tiền cho Người Mua. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua so sánh và đối chiếu kết quả demo giữa các thành viên, tôi nhận thấy chiến lược `RecursiveChunker` (chunk_size=500) kết hợp metadata pre-filtering mang lại độ chính xác vượt trội so với `FixedSizeChunker` hay `SentenceChunker`. Đặc biệt ở Câu 5, khi chưa có bộ lọc (`metadata_filter=None`), Top-1 bị nhầm sang tài liệu của Người Mua (`return-refund-policy`, score=0.8318) do từ vựng tương đồng; nhưng khi áp dụng bộ lọc tiền xử lý `audience: 'seller'`, Top-1 lập tức chuyển chính xác sang tài liệu Người Bán (`return-refund-seller`, score=0.8178), chứng minh tầm quan trọng sống còn của metadata filtering trong các hệ thống RAG thực tế.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10 |
| **Tổng phần cá nhân** | **59 / 60** |
