# Reflection — Lab 19

**Tên:** Tong Van Tien
**Cohort:** A20-K1
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Kết quả Precision@10 trên 50 queries của tôi:

- `exact` (n=15): keyword = semantic = hybrid ≈ 96.7–96.7%
- `paraphrase` (n=15): keyword = 33.3%, semantic = 24.0%, hybrid = 32.0%
- `mixed` (n=20): keyword = 97.0%, semantic = 98.5%, **hybrid = 100.0%**

**Phân tích:**

1. **Exact queries** (chứa từ kỹ thuật verbatim): BM25 đã rất mạnh vì match exact term. Hybrid không cải thiện thêm vì signal đã saturated.

2. **Paraphrase queries** (dùng từ Việt không xuất hiện verbatim): Cả BM25 và vector đều yếu vì model `bge-small-en-v1.5` được train chủ yếu trên tiếng Anh — semantic recall trên paraphrase tiếng Việt rất thấp. Đây là bài học "embedding model choice matters". Đổi sang `bge-m3` (multilingual) sẽ cải thiện semantic recall đáng kể.

3. **Mixed queries**: Hybrid thắng vì kết hợp được signal exact từ BM25 + signal semantic từ vector, đạt 100%.

**Khi nào KHÔNG dùng hybrid:**
- Khi latency budget rất chặt (<10ms) — hybrid chạy cả 2 retriever, tốn gấp đôi compute.
- Khi corpus rất nhỏ (<1k docs) — gain từ RRF không bù được overhead.
- Khi query domain rất hẹp và 100% exact (vd: search code) — BM25 thuần đã đủ.
- Khi 100% semantic (vd: image search) — vector thuần đã đủ.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất: model `bge-small-en-v1.5` mặc dù được quảng cáo là default "lite" nhưng semantic recall trên paraphrase tiếng Việt chỉ đạt 24% — thấp hơn BM25 keyword 33%. Điều này cho thấy việc chọn embedding model phù hợp với ngôn ngữ corpus là QUYẾT ĐỊNH KIẾN TRÚC quan trọng, không thể delegate cho AI mà không specify rõ language/corpus size/latency budget.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_