# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603640
- **Họ và tên:** Bùi Đăng Quang
- **SEED:** 3537

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| `day01_lab.ipynb` — notebook đã chạy, còn output | ☑ Có (đã đính kèm) |
| `lab1_report.json` — do ô “Báo cáo cuối” sinh ra | ☑ Có (đã đính kèm) |
| `hand_calc.jpg` — ảnh tính tay có các bước | ☑ Có (đã đính kèm) |

## 2. Kết quả chính

Bảng dưới đây chép từ kết quả thực nghiệm trong notebook. Lưu ý LLM chỉ được đánh giá trên SUB 40; vì vậy dùng cột F1 (SUB 40) để so sánh chất lượng giữa tất cả phương pháp.

| Phương pháp | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|
| BoW + LogReg | 0.865 | 0.857 | 0.153 | 0.000 |
| BoW + MLP | 0.937 | 0.947 | 0.405 | 0.000 |
| MiniLM + kNN (k=5) | 0.902 | 0.923 | 60.594 | 0.000 |
| MiniLM + LogReg | 0.939 | 0.947 | 53.338 | 0.000 |
| MiniLM + MLP head | 0.933 | 0.947 | 34.754 | 0.000 |
| LLM nhỏ zero-shot | — | 0.952 | 2734.671 | 0.000 |
| LLM lớn zero-shot | — | 0.927 | 2522.948 | 0.000 |
| LLM nhỏ few-shot | — | 0.927 | 2668.609 | 0.000 |

Các chỉ số bổ sung đáng chú ý:
- BoW + LogReg: train 8.204 giây; throughput theo lô 14,589.599 tin/giây.
- BoW + MLP: train 12.949 giây; throughput theo lô 24,914.263 tin/giây.
- MiniLM + MLP head: train 76.156 giây; throughput theo lô 48.788 tin/giây.
- LLM nhỏ zero-shot: throughput đo được 1.031 tin/giây.

Giá USD/1k tin trong notebook đều hiển thị 0.000 theo thông tin chi phí mà provider trả về trong lần chạy này. Không nên suy ra rằng triển khai LLM ở quy mô sản phẩm thực tế chắc chắn miễn phí; cần đối chiếu bảng giá và mức sử dụng thực tế của nhà cung cấp.

## 3. Dự đoán và thực tế

Chọn ba dự đoán có độ lệch rõ nhất so với kết quả:

1. **So sánh LLM nhỏ và LLM lớn**
   - Dự đoán: model lớn (`gpt-oss-120b`) có F1 cao hơn model nhỏ khoảng 20–30%, và đắt hơn 1.5–2 lần.
   - Thực tế: LLM nhỏ zero-shot đạt F1 SUB 0.952, cao hơn LLM lớn zero-shot 0.927. Chi phí cả hai đều được notebook ghi nhận là 0.000 USD/1k tin trong lần đo.
   - Giải thích: kích thước model lớn hơn không đảm bảo kết quả tốt hơn trên mọi tác vụ. Kết quả còn phụ thuộc cách model tuân thủ yêu cầu, dữ liệu đánh giá nhỏ và độ dao động của tập SUB 40. Chi phí bằng 0 ở đây phản ánh dữ liệu đo từ provider trong lần chạy, không phải cam kết giá triển khai.

2. **Dự đoán về đường Pareto**
   - Dự đoán: “không có cách nào” nằm trên đường Pareto.
   - Thực tế: notebook xác định ba phương án trên đường Pareto là BoW + LogReg, BoW + MLP và LLM nhỏ zero-shot.
   - Giải thích: đường Pareto không yêu cầu một phương pháp phải tốt nhất ở cả hai tiêu chí. BoW + LogReg rất nhanh; BoW + MLP cân bằng tốc độ và F1; LLM nhỏ có F1 SUB cao nhất nhưng độ trễ lớn.

3. **Dự đoán về few-shot**
   - Dự đoán: thêm ví dụ few-shot sẽ làm F1 tăng.
   - Thực tế: F1 SUB giảm từ 0.952 (LLM nhỏ zero-shot) xuống 0.927 (LLM nhỏ few-shot).
   - Giải thích: các ví dụ few-shot được chọn không nhất thiết đại diện tốt cho tập cần phân loại. Việc thêm ví dụ cũng làm prompt dài hơn, tăng độ trễ; few-shot không mặc nhiên cải thiện kết quả.

Một dự đoán khác cũng đáng chú ý: accuracy trên 50 case khó được dự đoán khoảng 70–80%, nhưng thực tế dao động rất rộng theo model, từ 0.50–0.52 với BoW đến 0.88–0.92 với LLM.

## 4. Phân tích 50 case khó

Hai mô hình BoW tụt mạnh nhất:
- BoW + LogReg: F1 từ 0.857 trên SUB thường xuống 0.333 trên 50 case khó; accuracy 0.52.
- BoW + MLP: F1 từ 0.947 xuống 0.324; accuracy 0.50.

Nguyên nhân chính là BoW dựa vào sự xuất hiện của token, không nắm chắc ngữ cảnh và ý định của cả câu. Vì vậy mô hình dễ nhầm tin nhắn bình thường có từ như “FREE”, “WON”, “URGENT” thành spam; đồng thời có thể bỏ sót spam viết tự nhiên hoặc cố tình biến dạng từ khóa.

Embedding cải thiện khả năng tổng quát hóa nhưng vẫn chưa ổn định bằng LLM trên bộ case viết tay:
- MiniLM + kNN: F1 0.746; accuracy 0.70.
- MiniLM + LogReg: F1 0.640; accuracy 0.64.
- MiniLM + MLP head: F1 0.692; accuracy 0.68.

LLM đạt kết quả tốt nhất:
- LLM nhỏ zero-shot: F1 0.903; accuracy 0.88.
- LLM nhỏ + prompt phòng thủ: F1 0.935; accuracy 0.92.

Hai case cụ thể mà LLM có prompt phòng thủ vẫn phân loại sai:
- **Case #2 — “FREE pizza in the office kitchen right now, come quick before it's gone”**: nhãn thật là ham nhưng bị đoán thành spam. Các từ “FREE”, “right now” và cách thúc giục giống quảng cáo khiến model quá nhạy với dấu hiệu từ khóa.
- **Case #8 — “Cash prize for the quiz night goes to the winning team, sign up with Sarah by Friday”**: nhãn thật là ham nhưng bị đoán thành spam. Cụm “Cash prize” và “winning team” giống nội dung trao thưởng, trong khi ngữ cảnh thực tế là hoạt động quiz nội bộ.

Prompt phòng thủ giúp tăng accuracy tổng thể từ 0.88 lên 0.92 trên 50 case, nhưng vẫn cần tránh xem bộ 50 case viết tay là đại diện cho phân phối SMS thực tế. Đây là bộ kiểm tra tình huống biên, hữu ích để phát hiện điểm yếu hơn là để ước lượng hiệu năng triển khai.

## 5. Thử thách cá nhân

**Đề số 4 (SEED 3537):** Chạy `knn_predict` với `k = 1, 5, 25`, báo F1 và giải thích vì sao k quá nhỏ hoặc quá lớn đều có hại.

Cách làm: giữ nguyên embedding MiniLM, tập train (`E_tr`, `y_tr`) và tập test (`E_te`, `y_te`); chỉ thay đổi số láng giềng `k`. Với mỗi giá trị k, lấy nhãn của k vector gần nhất và dự đoán spam khi số láng giềng spam chiếm quá một nửa. Đã bổ sung đoạn code thực hiện cả ba lượt vào ô thử thách cá nhân trong notebook.

Kết quả mốc đã có từ thí nghiệm k=5 ở phần so sánh mô hình: F1 test = **0.902**, F1 SUB 40 = **0.923**. Hai giá trị k=1 và k=25 cần chạy ô thử thách vừa bổ sung trong Colab để lấy số đo thực tế; không điền số ước đoán thay cho kết quả chạy.

Giải thích:
- **k quá nhỏ (k=1):** dự đoán phụ thuộc vào đúng một láng giềng gần nhất. Nếu láng giềng đó là ngoại lệ, nhiễu hoặc một tin nhắn có nội dung rất giống nhưng khác nhãn, kết quả dễ bị đảo.
- **k quá lớn (k=25):** nhiều láng giềng ít liên quan hơn được đưa vào bỏ phiếu, có thể làm loãng tín hiệu của nhóm gần nhất và thiên về lớp chiếm đa số trong dữ liệu (ham).
- **k=5:** là mức trung gian trong thí nghiệm gốc; cần so sánh số F1 thực tế của cả ba giá trị để kết luận mức nào phù hợp nhất với tập dữ liệu này.

## 6. Phương án cho sản phẩm thật

Yêu cầu là 10 triệu SMS/ngày, tương đương trung bình khoảng 116 tin/giây, với p95 dưới 50 ms/tin.

Theo số liệu notebook:
- **Đạt SLA độ trễ:** BoW + LogReg (p95 0.153 ms), BoW + MLP (0.405 ms) và MiniLM + MLP head (34.754 ms).
- **Không đạt SLA:** MiniLM + kNN (60.594 ms), MiniLM + LogReg (53.338 ms) và cả ba cấu hình LLM (p95 khoảng 2,523–2,735 ms).
- **Lựa chọn phù hợp nhất trong số liệu hiện có:** BoW + MLP, vì F1 test 0.937, F1 SUB 0.947, p95 0.405 ms và throughput theo lô khoảng 24,914 tin/giây. Đây là phương án vừa đạt SLA vừa có năng lực xử lý trung bình cao hơn nhiều so với yêu cầu 116 tin/giây.
- **Đánh đổi:** BoW + MLP vẫn yếu trên stress test (accuracy 0.50), nên không thể chỉ dựa vào F1 test thường để khẳng định an toàn. Cần bổ sung kiểm thử dữ liệu thực tế, theo dõi drift và có cơ chế chuyển các tin không chắc chắn sang tầng phân tích mạnh hơn.

Đề xuất router:
1. Dùng BoW + MLP làm tầng phân loại nhanh cho toàn bộ SMS.
2. Lấy xác suất dự đoán thay vì chỉ lấy nhãn. Nếu xác suất gần ngưỡng quyết định (ví dụ khoảng 0.3–0.7), chuyển sang MiniLM hoặc LLM để đánh giá bổ sung; các tin có độ tin cậy cao được xử lý ngay.
3. Với tin bị đánh dấu spam, lưu lý do/đặc trưng và cơ chế khiếu nại hoặc rà soát để hạn chế chặn nhầm OTP, giao dịch và tin nhắn cá nhân.

Router có thể giảm số lần gọi tầng chậm/đắt, đồng thời tận dụng tốc độ của mô hình BoW và khả năng hiểu ngữ cảnh của mô hình mạnh hơn. Ngưỡng 0.3–0.7 chỉ là đề xuất ban đầu, cần hiệu chỉnh bằng tập validation và chi phí của hai loại lỗi (chặn nhầm ham và bỏ lọt spam).

Về chi phí LLM: notebook hiện ghi 0.000 USD/1k tin cho lần chạy qua Groq, nên không đủ dữ liệu để tính chi phí tháng đáng tin cậy cho triển khai thương mại. Nếu lấy đúng con số đo được thì phép nhân cho 300 triệu tin/tháng vẫn ra 0 USD, nhưng đây không phải ước tính giá sản phẩm; cần thay bằng đơn giá thực tế của API và số token trung bình trước khi quyết định.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Hoàn thiện báo cáo | Hỗ trợ đọc notebook, lấy đúng số liệu thực nghiệm và điền báo cáo Markdown. | Tổng hợp bảng F1, độ trễ, stress test; soạn phần nhận xét và phương án triển khai. | Đối chiếu từng số với output notebook; giữ nguyên số liệu và ghi rõ phần thử thách k=1, k=25 chưa chạy. |
| Phân tích mô hình | Giải thích sự khác nhau giữa BoW, Embedding, MLP và LLM trong bài toán spam/ham. | Giải thích ưu nhược điểm dựa trên kết quả thực nghiệm và các case khó. | Đối chiếu F1 và accuracy 50 case trong notebook, không xem kết quả stress test là đại diện dữ liệu thực tế. |
| Phân tích triển khai | Chọn mô hình đáp ứng p95 dưới 50 ms và tải 10 triệu SMS/ngày. | So sánh p95, throughput, F1 và đề xuất router nhiều tầng. | Kiểm tra cả độ trễ lẫn throughput; lưu ý số đo Colab không thay thế benchmark trên hạ tầng production. |

**Điều đã học:** Em hiểu rằng F1 cao trên tập test cùng phân phối chưa chắc đồng nghĩa mô hình hiểu đúng ngữ cảnh; stress test có thể làm lộ rõ điểm yếu của mô hình dựa trên từ khóa. Em cũng hiểu việc chọn mô hình triển khai phải cân bằng chất lượng, độ trễ và năng lực xử lý thực tế, chứ không chỉ chọn model có F1 cao nhất.
