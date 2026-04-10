# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Hoàng Thị Thanh Tuyền
**Nhóm:** C401-X1
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> High cosine similarity là thước đo mức độ tương đồng của 2 vecto dựa trên góc giữa chúng, được tính bằng tích vô hướng trên tích độ dài.

**Ví dụ HIGH similarity:**

- Sentence A: Hôm nay trời nắng to
- Sentence B: Thời tiết hôm nay cực kỳ oi bức
- Tại sao tương đồng: Vì đều nói về thời tiết của ngày hôm nay và đặc điểm giống nhau là đều nắng nóng, chỉ khác nhau về cách diễn đạt.

**Ví dụ LOW similarity:**

- Sentence A: Hôm nay trời nắng to
- Sentence B: Tôi rất thích đọc truyện
- Tại sao khác: Hai câu này nói về 2 chủ đê khác nhau không liên quan: một câu nói về thời tiết hôm nay còn một câu nói về sở thích cá nhân.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Bởi vì cosine similarity đo hướng giữa 2 vector thay vì dựa trên độ dài như khoảng cách Euclide. Xét trong không gian nhiều chiều, đo sai khác về góc tốt hơn là dựa trên độ dài.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> num*chunks = ceil((10000 - 50) / (500 - 50))  
> = ceil(9950 / 450)  
> ≈ ceil(22.11)  
> = 23
> *Đáp án: 23 chunk

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> chunk count tăng lên khoảng 25 chunks vì step nhỏ hơn. Overlap lớn giúp giữ context giữa các chunk đặc biệt là trường hợp các từ quan trọng nằm trúng điểm cắt nhờ đó giúp cải thiện chất lượng retrieval.

---

## 2. Document Selection - Nhóm (10 diem)

### Domain & Lý do chọn

**Domain:** Xanh SM FAQ (user + driver + merchant/restaurant).

**Tại sao chọn domain này?**  
Bộ FAQ này có nhiều nhóm đối tượng và nhiều quy trình nghiệp vụ, có cấu trúc dễ dàng rất phù hợp để kiểm tra retrieval precision, metadata filtering va grounding.

### Data Inventory

| #   | Tên tài liệu                           | Nguồn            | Số ký tự | Metadata đã gán                            |
| --- | -------------------------------------- | ---------------- | -------: | ------------------------------------------ |
| 1   | XanhSM - User FAQs.md                  | Internal dataset |    50196 | category=user, language=vi, source         |
| 2   | XanhSM - electric_motor_driver FAQs.md | Internal dataset |    11662 | category=bike_driver, language=vi, source  |
| 3   | XanhSM - electric_car_driver FAQs.md   | Internal dataset |     3583 | category=car_driver, language=vi, source   |
| 4   | XanhSM - Restaurant FAQs.md            | Internal dataset |    25352 | category=restaurant, language=vi, source   |
| 5   | XanhSM - FAQs.md                       | Internal dataset |    50196 | category=user_general, language=vi, source |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị                                | Tại sao hữu ích cho retrieval?        |
| --------------- | ------ | -------------------------------------------- | ------------------------------------- |
| doc_id          | string | XanhSM - User FAQs                           | Hỗ trợ delete theo tài liệu gốc       |
| category        | string | user / bike_driver / car_driver / restaurant | Hỗ trợ pre-filter theo domain câu hỏi |
| language        | string | vi                                           | Tránh mix kết quả khác ngôn ngữ       |
| source          | string | data/XanhSM - User FAQs.md                   | Truy vết nguồn chunk và đối chiếu     |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 4 tài liệu nhóm:

| Tài liệu                    | Strategy                      | Chunk Count | Avg Length | Preserves Context?          |
| --------------------------- | ----------------------------- | ----------- | ---------- | --------------------------- |
| XanhSM User FAQ (50K chars) | FixedSizeChunker (350 chars)  | 96          | ~520       | Có, nhưng cắt giữa câu      |
| XanhSM User FAQ (50K chars) | SentenceChunker (3 sentences) | 73          | ~687       | Tốt, giữ câu nguyên         |
| XanhSM User FAQ (50K chars) | RecursiveChunker (300 chars)  | 218         | ~231       | Rất tốt, tận dụng structure |

### Strategy Của Tôi

**Loại:** RecursiveChunker (350 chars)

**Mô tả cách hoạt động:**
RecursiveChunker thử separators theo thứ tự `["\n\n", "\n", ". ", " ", ""]`. Nó ưu tiên tách theo paragraph trước, nếu vẫn > 300 chars thì split theo câu, sau đó theo từ, cuối cùng fallback ký tự. Với FAQ markdown có heading (##), nó tách theo ranh giới logical (paragraphs), nên chunk thường bao gồm 1-2 Q&A pairs nguyên vẹn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
Domain FAQ của Xanh SM có cấu trúc rõ ràng: headings, questions, answers. RecursiveChunker khai thác structure này để tạo chunks coherent xung quanh Q&A pairs, thay vì cắt giữa lời giải thích. Với retrieval FAQ, việc giữ nguyên một Q&A pair là critical để trả lời đầy đủ.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu     | Strategy                       | Chunk Count | Avg Length | Retrieval Quality |
| ------------ | ------------------------------ | ----------- | ---------- | ----------------- |
| Tổng 4 files | FixedSize (best baseline)      | 288         | ~486       | 0.8044            |
| Tổng 4 files | **RecursiveChunker (của tôi)** | **654**     | **214**    | **0.8771**        |

### So Sánh Với Thành Viên Khác

| Thành viên                  | Strategy                                      | Retrieval Score (/10) | Điểm mạnh                                               | Điểm yếu                                                                   |
| --------------------------- | --------------------------------------------- | --------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------- | --- |
| Hoàng Thị Thanh Tuyền (tôi) | Recursive (350 chars)                         | 8.77                  | Giữ context, Q&A coherent, score cao nhất               | Số chunk nhiều (654), tốn memory                                           |
| Đỗ Thế Anh                  | Recursive (250 chars)                         | 8.752                 | Trích xuất chính xác, duy trì được thông tin quan trọng | Số chunk nhiều, dẫn đến dư thừa dữ liệu do overlap                         |     |
| Võ Thanh Chung              | RecursiveChunker (250 chars)                  | 8                     | Giữ cấu trúc tự nhiên, chunk đều                        | Có thể cắt ngang câu dài                                                   |
| Nguyễn Hồ Bảo Thiên         | FixedSizeChunker (chunk_size=100, overlap=20) | 8.56                  | Xử lý nhanh                                             | Dễ ngắt câu giữa chừng, gây mất ngữ nghĩa                                  |
| Dương Khoa Điềm             | RecursiveChunker                              | 7.9                   | Giữ được ngữ cảnh cụm Q&A tương đối ổn.                 | Tuỳ biến sai sót separator khiến một số câu dài bị đứt vụn, điểm chưa cao. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
RecursiveChunker tốt nhất với score 0.8771. Trong FAQ domain, structure heading → Q&A là critical, và recursive strategy khai thác điều này optimal. Nó giữ được context nguyên vẹn (Q&A pair không bị chia lẻ), nên retrieval precision cao. Trade-off là chunk count lớn, nhưng với FAQ, precision quan trọng hơn efficiency.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> Dùng regex `(?<=[.!?])\s+|\.\n` để phát hiện sentence boundaries, tách text thành các câu riêng biệt. Sau đó nhóm các câu liên tiếp thành chunks theo `max_sentences_per_chunk`, và strip whitespace để tránh leading/trailing spaces. Edge case: xử lý text rỗng hoặc chunk cuối cùng có thể có ít câu hơn limit.

**`RecursiveChunker.chunk` / `_split`** — approach:

> Algorithm recursively thử từng separator theo thứ tự ưu tiên (từ coarse như `"\n\n"` đến fine như `" "`). Base case: nếu text ngắn hơn `chunk_size` hoặc hết separators, trả về text đó. Recursive case: split theo separator hiện tại, rồi gọi `_split` lại trên từng part với separators còn lại. Cách này đảm bảo chunk coherent và tận dụng cấu trúc tự nhiên của text.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> `add_documents` lặp qua từng doc, tạo record normalize (id, content, embedding, metadata) bằng `_make_record()`, rồi append vào `self._store` (hoặc add vào ChromaDB nếu available). `search` tính embedding của query, rồi gọi `_search_records()` để tính dot product similarity giữa query embedding và tất cả stored embeddings, sort descending, và trả về top-k kết quả kèm score.

**`search_with_filter` + `delete_document`** — approach:

> `search_with_filter` filter trước: lặp qua `self._store` và giữ lại record có metadata match với filter dict, sau đó gọi `_search_records()` trên filtered records. `delete_document` lặp qua store và xóa những record có `metadata['doc_id']` khớp với `doc_id` parameter, rồi return True nếu có xóa được record.

### KnowledgeBaseAgent

**`answer`** — approach:

> `answer` implement RAG pattern: đầu tiên gọi `store.search(question, top_k)` để retrieve top-k chunks liên quan. Sau đó build prompt bằng cách format chunks thành context section (ví dụ: "Context:\n[chunk1]\n[chunk2]\n..."), rồi gắn question vào. Cuối cùng gọi `llm_fn(prompt)` để generate answer từ context và question đó, đảm bảo answer được ground trong retrieved chunks.

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A                                              | Sentence B                                                                   | Dự đoán | Actual Score | Đúng? |
| ---- | ------------------------------------------------------- | ---------------------------------------------------------------------------- | ------- | ------------ | ----- |
| 1    | "Làm sao khi hành khách để quên đồ trên xe?"            | "Cách nhanh nhất để hoàn trả lại đồ để quên là cung cấp thông tin chuyến đi" | high    | 0.880        | Yes   |
| 2    | "Hướng dẫn xuất hóa đơn VAT với các chuyến xe Xanh SM"  | "Để yêu cầu xuất hóa đơn VAT cho chuyến xe, bạn cần tuân theo hướng dẫn"     | high    | 0.876        | Yes   |
| 3    | "Tôi muốn đặt chuyến giao đồ ăn trên ứng dụng"          | "Đặt chuyến xe giao hàng đồ ăn (Xanh Ngon) rất đơn giản"                     | high    | 0.889        | Yes   |
| 4    | "Quán có rating trên Google và muốn đồng bộ về Xanh SM" | "Xanh Ngon định kỳ cập nhật rating Google của các Merchant"                  | high    | 0.878        | Yes   |
| 5    | "Ngoài lương thưởng còn được hưởng chính sách gì?"      | "Bảo vệ linh hoạt, đăng ký/hủy trên ứng dụng, hỗ trợ 24/7"                   | high    | 0.862        | Yes   |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Tất cả 5 cặp đều dự đoán cao và actual cũng cao (0.86-0.89). Embeddings rất mạnh ở semantic keyword matching - chúng capture được ý chính dù câu có diễn đạt khác nhau. Điều này chỉ ra embeddings tốt ở synonym detection và paraphrase, nhưng có thể yếu ở negation logic (ví dụ: "không có" vs "có") hoặc các fine-grained distinctions.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                                                                 | Gold Answer                                                                                                                                          |
| --- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Hướng dẫn yêu cầu xuất hóa đơn VAT và cách kiểm tra hóa đơn với các chuyến xe Xanh SM | Yêu cầu xuất hóa đơn qua 2 cách: Cách 1 qua chuyến xe tương ứng, Cách 2 khi đang trong chuyến đi. Kiểm tra hóa đơn sẽ được gửi qua email đã đăng ký. |
| 2   | Làm sao khi hành khách để quên đồ trên xe?                                            | Cung cấp thông tin chuyến đi cho tài xế, hoặc liên hệ tổng đài hỗ trợ 24/7. Tài xế sẽ hoàn trả đồ hoặc gửi về địa chỉ.                               |
| 3   | Ngoài lương thưởng, tôi còn được hưởng chính sách gì nữa?                             | Tài xế GSM nhận được: bảo vệ linh hoạt, đăng ký/hủy trên ứng dụng, hỗ trợ sự cố 24/7, chương trình khuyến mãi.                                       |
| 4   | Tôi muốn đặt chuyến giao đồ ăn trên ứng dụng                                          | Đặt chuyến Xanh Ngon: nhập địa chỉ đón/giao, chọn menu, thanh toán. Hệ thống sẽ gộp đơn nếu có sẵn.                                                  |
| 5   | Quán có rating trên Google và muốn đồng bộ về Ứng dụng Xanh SM                        | Xanh Ngon định kỳ cập nhật rating Google nếu: tên/địa chỉ trùng khớp + rating ≥ 4.0. Gửi yêu cầu tại "Trung tâm hỗ trợ".                             |

### Kết Quả Của Tôi (Dùng RecursiveChunker Strategy)

| #   | Query                             | Top-1 Retrieved Chunk (tóm tắt)                                                             | Score | Relevant? | Agent Answer (tóm tắt)                                     |
| --- | --------------------------------- | ------------------------------------------------------------------------------------------- | ----- | --------- | ---------------------------------------------------------- |
| 1   | Hướng dẫn xuất hóa đơn VAT        | "Để yêu cầu xuất hóa đơn VAT cho chuyến xe Xanh SM, bạn cần tuân theo các hướng dẫn sau..." | 0.876 | Yes       | Trả lời đầy đủ 2 cách + kiểm tra email                     |
| 2   | Làm sao khi hành khách để quên đồ | "Làm sao khi hành khách để quên đồ trên xe? Cách nhanh nhất để hoàn trả lại đồ..."          | 0.880 | Yes       | Trả lời chính xác: liên hệ tài xế/tổng đài                 |
| 3   | Ngoài lương thưởng còn hưởng gì   | "Ngoài lương thưởng, tôi còn được hưởng chính sách gì nữa? Bên cạnh lương..."               | 0.862 | Yes       | Liệt kê đầy đủ: bảo vệ, ứng dụng, hỗ trợ                   |
| 4   | Đặt chuyến giao đồ ăn trên app    | "Tôi muốn đặt chuyến giao đồ ăn trên ứng dụng. Để đặt chuyến xe giao hàng..."               | 0.889 | Yes       | Hướng dẫn từng bước: nhập địa chỉ → chọn menu → thanh toán |
| 5   | Rating Google đồng bộ Xanh SM     | "Quán có rating trên Google và muốn đồng bộ về Ứng dụng Xanh SM. Hiện tại..."               | 0.878 | Yes       | Giải thích điều kiện: tên/địa chỉ + rating ≥ 4.0           |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
FixedSizeChunker với chunk_size=100 + overlap=20 đạt score 8.56, cao hơn dự tính. Điều này chứng minh hyperparameter tuning quan trọng như lựa chọn strategy type. Tôi đã nghĩ FixedSize sẽ tệ hơn Recursive, nhưng thực tế cho thấy một FixedSize well-tuned có thể gần bằng Recursive tối ưu (8.56 vs 8.77). Lessons: không nên dismiss strategy dựa trên theory - cần test thực nghiệm với parameters phù hợp.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Nhóm khác dùng custom DocumentStructure chunker. Dù score thấp hơn recursive nhưng strategy phù hợp với domain cụ thể của họ. Họ giải thích: domain-specific optimization khó scale sang domain khác. Tôi học: recursive strategy general hơn và robust hơn custom strategy.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
(1) Chuẩn bị metadata richer từ đầu (thêm `section_level`, `faq_type`: question/answer/note) để hỗ trợ filter tốt hơn. (2) Thử chunk_size cho Recursive theo chiến lược từ lớn -> nhỏ để tìm sweet spot. (3) Tạo validation set riêng (1-2 queries/strategy) để tune trước khi chạy benchmark chính thức, tránh overfitting solution.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | 5 / 5            |
| Document selection          | Nhóm    | 9 / 10           |
| Chunking strategy           | Nhóm    | 14 / 15          |
| My approach                 | Cá nhân | 9 / 10           |
| Similarity predictions      | Cá nhân | 5 / 5            |
| Results                     | Cá nhân | 10 / 10          |
| Core implementation (tests) | Cá nhân | 30 / 30          |
| Demo                        | Nhóm    | 5 / 5            |
| **Tổng**                    |         | **86 / 100**     |
