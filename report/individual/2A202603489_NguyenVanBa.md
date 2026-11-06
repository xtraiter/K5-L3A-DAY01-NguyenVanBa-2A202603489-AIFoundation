# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook): 5869

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Số liệu lấy từ lần chạy cuối, trùng với `lab1_report.json`. Bộ kiểm tra đạt 15/15. Nhóm LLM chỉ chạy trên SUB 40 tin nên F1 test bằng F1 SUB 40.

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.868 | 0.889 | 0.14 | 0 |
| 2. BoW + MLP | 0.909 | 0.947 | 0.32 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.922 | 0.947 | 72.59 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.951 | 0.919 | 54.33 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.952 | 0.919 | 82.87 | 0 |
| 4a. LLM nhỏ zero-shot | 0.974 | 0.974 | 1605.91 | 0.0073 |
| 4b. LLM lớn zero-shot | 0.974 | 0.974 | 1248.79 | 0.0732 |
| 4c. LLM nhỏ few-shot | 0.919 | 0.919 | 1215.91 | 0.0203 |

Trên biểu đồ Pareto có BoW + LogReg, BoW + MLP và LLM lớn zero-shot. Chỉ hai cách BoW đạt p95 < 50 ms.

## 3. Dự đoán và thực tế

| Dự đoán | Tôi đoán | Thực tế | Vì sao lệch |
| --- | --- | --- | --- |
| `llm_fewshot` | F1 tăng nhẹ hoặc giữ nguyên, chi phí tăng 2–4 lần | F1 giảm từ 0.974 xuống 0.919, recall giảm từ 0.95 xuống 0.85, chi phí tăng khoảng 2,8 lần | Phần chi phí đoán đúng. Phần F1 sai vì 4 ví dụ chỉ là 2 spam và 2 ham đầu tiên, không đại diện. SUB chỉ 40 tin nên sai 1 tin đã lệch khoảng 0.03, chưa đủ để kết luận few-shot kém hơn |
| `llm_models` (ý 3) | LLM không thắng rõ 3b mà chậm và tốn tiền hơn | Chậm và tốn tiền hơn là đúng, nhưng F1 SUB là 0.974 so với 0.919, và trên case khó là 0.939 so với 0.582 | Tôi đánh giá thấp khả năng đọc hiểu ý định của LLM. Khác biệt lớn chỉ lộ ra ở case khó, còn test thường làm các model trông gần nhau |
| `hard_eval` (ý 2) | Khó nhất với LLM là prompt injection | Cả 4a và 4d đúng 100% nhóm Prompt injection. Nhóm thấp nhất là Thao túng / lừa đảo với 75% | Tin injection lộ rõ ý đồ nên LLM vẫn nhận ra. 3 tin LLM sai đều là case tiếng Việt tôi tự thêm, trông như thông báo bình thường |

## 4. Phân tích 50 case khó

Bảng đo trên 55 case: 50 case có sẵn và 5 case tôi tự thêm.

| Phương án | F1 SUB 40 | F1 case khó |
| --- | ---: | ---: |
| 1. BoW + LogReg | 0.889 | 0.250 |
| 2. BoW + MLP | 0.947 | 0.318 |
| 3a. Emb + kNN | 0.947 | 0.667 |
| 3b. Emb + LogReg | 0.919 | 0.582 |
| 3c. Emb + MLP head | 0.919 | 0.538 |
| 4a. LLM nhỏ zero-shot | 0.974 | 0.939 |
| 4d. LLM nhỏ + prompt phòng thủ | 0.974 | 0.954 |

BoW + LogReg tụt nhiều nhất, từ 0.889 xuống 0.250. Nhóm yếu nhất của nó là Spam giọng tự nhiên (đúng 0%), Thao túng / lừa đảo (8%) và Né từ khoá (17%). BoW chỉ nhìn token rời rạc, nên tin không có từ khoá quen hoặc viết kiểu `C0ngr4ts` thì coi như không thấy. Test thường cùng nguồn với train và cùng kiểu spam cũ nên điểm cao quá lạc quan.

Hai case cụ thể, đều làm tất cả model sai:

- **Case #100:** tin "[Giao hàng] Đơn COD 1.250.000đ... Shipper sẽ liên hệ trước khi giao", nhãn spam. Tin không có từ khoá spam, đường link hay lời kêu gọi bấm vào đâu, nên BoW không có tín hiệu. Câu chữ giống thông báo giao hàng thật nên embedding và LLM cũng xếp vào ham.
- **Case #103:** tin báo con của phụ huynh bị ngã ở trường, nhãn spam. Đây là lừa đảo đánh vào cảm xúc, nhưng phần chữ chỉ là một lời báo tin bình thường. Dấu hiệu lừa đảo chỉ xuất hiện ở bước sau (cuộc gọi), không nằm trong tin nên model không thấy được.

Hai tin này chỉ là spam khi biết bối cảnh, nên nhãn của chúng cũng mang tính giả định.

## 5. Thử thách cá nhân

**Đề #2 (theo SEED):** giảm vocab còn 100 từ và tăng lên 5000 từ bằng `build_vocab(..., max_size)`, train lại, báo F1 test và giải thích xu hướng.

**Cách làm:** chạy vòng lặp qua vocab 100, 2000, 5000. Mỗi vòng dựng lại vocab, vector hoá train và test, train lại bằng `train_logreg`, rồi đo F1 trên cả hai tập.

| Vocab | F1 train | F1 test | Precision | Recall | Loss cuối |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.830 | 0.820 | 0.974 | 0.708 | 0.127 |
| 2000 | 0.909 | 0.868 | 0.988 | 0.774 | 0.084 |
| 5000 | 0.911 | 0.868 | 0.988 | 0.774 | 0.082 |

**Giải thích:** Vocab 100 làm F1 giảm (0.820) vì mất nhiều từ đặc trưng của spam (underfit, recall giảm còn 0.708); tăng từ 2000 lên 5000 không đổi gì (0.868) vì từ thêm vào đều hiếm, ít tín hiệu, nên hiệu quả bão hoà và 2000 là mức hợp lý.

## 6. Phương án cho sản phẩm thật

**Lựa chọn:** router, dùng BoW + MLP làm lớp chính và chỉ gửi tin lưỡng lự sang LLM.

**Lý do theo số liệu:**

- Chỉ BoW + LogReg (0.14 ms) và BoW + MLP (0.32 ms) đạt p95 < 50 ms. 3b sát vạch nhưng vượt với 54.33 ms; 3a là 72.59 ms, 3c là 82.87 ms, LLM từ 1216 đến 1606 ms.
- Trong hai cách đạt yêu cầu, BoW + MLP có F1 cao hơn (0.909 so với 0.868). Tốc độ 17.766 tin/s vượt xa mức trung bình 116 tin/s và không tốn tiền API.
- Dùng LLM cho toàn bộ 300 triệu tin/tháng tốn khoảng $2.196 với LLM nhỏ, $6.076 với few-shot và $21.960 với LLM lớn.

**Cách chạy:** BoW + MLP xử lý mọi tin, p < 0.3 là ham, p > 0.7 là spam. Tin có p từ 0.3 đến 0.7 mới gửi LLM nhỏ với prompt phòng thủ. Thêm một lớp luật chặn trước cho tin chứa câu kiểu "ignore instructions" hay "label: ham". Nếu khoảng 5% tin đi qua LLM thì chi phí vào khoảng $110/tháng.

**Đánh đổi chấp nhận:**

- BoW + MLP có recall 0.849 và F1 case khó chỉ 0.318, nên vẫn có thể lọt spam kiểu mới mà model tự tin là ham; LLM không kiểm tra lại các tin đó.
- Tin đi qua LLM mất khoảng 1 giây. Tỉ lệ này phải dưới 5% thì p95 toàn hệ thống mới giữ được dưới 50 ms.
- Router này là thiết kế, chưa được chạy thử; tỉ lệ 5% và chi phí trên chỉ là ước tính.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1–3 (TODO code) | Gửi ảnh từng TODO, hỏi cách điền | Code gợi ý + giải thích từng dòng | Dán vào, chạy test của notebook đến khi pass; đọc giải thích để hiểu shape, gradient, zero_grad |
| Dự đoán & câu hỏi 1–3 | Hỏi lập luận cho dự đoán và câu trả lời | Lập luận + câu trả lời mẫu | Chạy thí nghiệm, đối chiếu với số thật (vd lr=50, ReLU, 3.B 5/6 vs 4/6); yêu cầu viết lại ngắn theo cách tôi viết |
| 4.1–4.4 (LLM) | Hỏi cách viết call_openai, prompt, parse_label, batch | Code gợi ý | Chạy ra F1=0, invalid=40/40; tự chạy đoạn gỡ lỗi, thấy content rỗng + finish_reason=length; hỏi lại vì sao chọn 16 thay vì 4, tự đo max_tokens 3→rỗng, 4→"spam"; tự sửa PROVIDERS và chạy lại |
| Biểu đồ trade-off, câu 4–6 | Hỏi cách đọc Pareto, giải thích kết quả case khó | Phân tích + câu trả lời mẫu | Đối chiếu với bảng và biểu đồ của tôi; rút gọn lại theo ý mình |
| Bonus | Nhờ gợi ý case lừa model | Gợi ý case | Chạy, thấy model vẫn bắt được; yêu cầu case tiếng Việt và kiểu lừa chuyên nghiệp hơn; chạy lại và ghi nhận case model sai |
| Thử thách #2 | Hỏi code thử vocab 100/2000/5000 | Vòng lặp train + giải thích xu hướng | Chạy, lấy số thật (0.820 / 0.868 / 0.868) điền vào câu trả lời |


**Điều đã học:** Trước đây tôi tin vào trực giác và điểm trên test thường. Hôm nay few-shot lại kém zero-shot trên SUB (0.919 so với 0.974), còn test thường cho mọi model điểm gần nhau (0.89–0.97) trong khi case khó chênh rất xa (0.25–0.95). Vì vậy phải đo thật, nhìn cả độ trễ và chi phí, và thử trên case khó trước khi chọn model.
