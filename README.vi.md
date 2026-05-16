# Tóm Tắt Cuộc Thi XAI

Repository này chứa 2 bộ dữ liệu và bản tóm tắt các yêu cầu trọng tâm của cuộc thi XAI.

## Mục Tiêu Đánh Giá

Hệ thống sẽ được chấm theo 3 tiêu chí:

- **P1: Correctness of Answers** - độ chính xác của đáp án cuối cùng.
- **P2: Quality of Explanation** - chất lượng giải thích bằng ngôn ngữ tự nhiên, rõ ràng và mạch lạc.
- **P3: Depth of Reasoning** - độ sâu suy luận, thể hiện qua bằng chứng lập luận có cấu trúc (ví dụ: premises, FOL, suy diễn từng bước).

## Yêu Cầu Bắt Buộc

1. **Mỗi câu trả lời phải có phần giải thích**.
   - Giải thích cần ngắn gọn, dễ hiểu và có thể kiểm chứng.

2. **Chỉ được dùng LLM mã nguồn mở**.
   - Mọi thành phần LLM trong hệ thống (trả lời, suy luận, NL-to-logic, v.v.) đều phải là open-source.
   - Kích thước tối đa: **8B tham số**.

3. **Phải công khai toàn bộ dữ liệu ngoài**.
   - Tất cả dataset bên ngoài dùng để fine-tune LLM hoặc symbolic engine phải được khai báo đầy đủ.

## Điều Bị Cấm

- Dùng LLM thương mại/đóng nguồn (ví dụ: GPT, Claude, Gemini).
- Che giấu hoặc không khai báo dữ liệu ngoài dùng để huấn luyện/fine-tune.

Vi phạm có thể dẫn đến **loại trực tiếp**.

## Hướng Tiếp Cận Được Khuyến Khích

- Tích hợp symbolic reasoning (ví dụ: Z3 hoặc engine tự xây) để kiểm chứng kết quả và tăng tính giải thích.
- Symbolic reasoning được khuyến khích nhưng **không bắt buộc**.

## Tổng Quan Dữ Liệu

### 1) Logic-Based Educational Queries

- **File**: `Logic_Based_Educational_Queries.json`
- **Quy mô**: 464 records, 913 câu hỏi.
- **Chủ đề**: quy định học thuật ở đại học (điểm số, đăng ký môn, học bổng, điều kiện học vụ, v.v.).
- **Loại câu hỏi**: Trắc nghiệm, Yes/No/Uncertain, và tự luận.
- **Thông tin đi kèm**:
  - Premises ngôn ngữ tự nhiên (`premises-NL`)
  - Premises dạng FOL
  - Câu hỏi
  - Đáp án chuẩn (ground-truth)
  - Giải thích do con người viết
- **Input khi chấm**: câu hỏi + premises ngôn ngữ tự nhiên.
- Team có thể xử lý premises linh hoạt (làm context prompt, chuyển FOL, symbolic solving, v.v.).

### 2) Physics Problems

- **File**: `Physics_Problems_Text_Only.csv`
- **Quy mô**: 5,520 bài toán dạng text.
- **Chủ đề**: mạch điện và tĩnh điện (điện trở, điện áp, dòng điện, công suất, điện dung, điện trường, năng lượng).
- **Đặc điểm**: bài toán số, cần tính nhiều bước.
- **Nhãn trong dataset**: CoT từng bước + đáp án số cuối cùng kèm đơn vị.
- **Input khi chấm**: **chỉ có câu hỏi**.
- Tài liệu nguồn tạo dataset sẽ được công bố ở kick-off workshop.

## Định Dạng Bài Test

Bộ test chính thức sẽ gộp 2 loại dữ liệu:

- Type 1: question + premises-NL
- Type 2: question only

Dạng câu hỏi có thể gồm:

- Trắc nghiệm
- Yes/No/Uncertain
- Tự luận suy luận
- Tính toán số

Tỷ lệ phân bố chủ đề sẽ được công bố tại kick-off workshop.

## Quy Trình Đánh Giá

- **Vòng 1 & 2 (Selection)**:
  - Chấm tự động theo ground-truth
  - Ban tổ chức review chất lượng giải thích
- **Chung kết (Final Round)**:
  - Chạy trực tiếp trên câu hỏi chưa từng thấy
  - Challenge Chairs đánh giá real-time về đáp án, giải thích, và độ sâu suy luận
- **Điểm cuối cùng**:
  - Tổ hợp trọng số của P1, P2, P3
  - Trọng số cụ thể sẽ công bố cùng bộ dữ liệu chính thức

## Yêu Cầu Nộp Bài

Mỗi team cần nộp:

1. Một **API endpoint**
2. Một bản mô tả giải pháp **1 trang**, gồm:
   - cách tiếp cận
   - mô hình sử dụng
   - dữ liệu dùng để huấn luyện

### Định Dạng Output API

Trường bắt buộc:

- `answer`
- `explanation`

Trường tùy chọn nhưng được khuyến khích (tăng điểm reasoning depth):

- `fol`
- `cot`
- `premises`
- `confidence`

Ví dụ:

```json
{
  "answer": "B",
  "explanation": "Điện áp trên R2 được tính bằng ...",
  "fol": "∀x (Resistor(x) → HasVoltage(x, V))",
  "cot": [
    "Bước 1: Xác định cấu trúc mạch ...",
    "Bước 2: Áp dụng định luật Kirchhoff về điện áp ...",
    "Bước 3: Giải biến chưa biết ..."
  ],
  "premises": [
    "Định luật Ohm: V = IR",
    "KVL: tổng điện áp trong một vòng kín bằng 0"
  ],
  "confidence": 0.92
}
```

> Lưu ý: Format nộp cuối cùng có thể được điều chỉnh tại kick-off workshop.

## Checklist Tuân Thủ Nhanh

- [ ] Mỗi output đều có `answer` và `explanation`.
- [ ] Tất cả thành phần LLM đều open-source và <= 8B tham số.
- [ ] Không dùng mô hình đóng nguồn ở bất kỳ khâu nào.
- [ ] Mọi dữ liệu ngoài dùng để train/fine-tune đều được khai báo.
- [ ] Có bổ sung bằng chứng suy luận (FOL/steps/premises/confidence) khi có thể.
- [ ] Output API đúng schema yêu cầu.

