# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603937
- **Họ và tên:** Bùi Đặng Nhật Nguyên
- **SEED:** 8476

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (trích từ Cell 78 của [`day01_lab.ipynb`](day01_lab.ipynb) sau khi chạy `Runtime → Run all`; trong ngoặc đơn ở các cột thời gian là số đo ở lần chạy tương tác ngay trước đó được lưu trong [`lab1_report.json`](lab1_report.json) — toàn bộ các cột chất lượng `Precision`, `Recall`, `F1 (test)`, `F1 (SUB 40)` của cả 2 lần chạy đều khớp nhau 100%):

| Phương án | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 1.000 | 0.760 | 0.863 | 0.889 | 1.494 (1.420) | 14,132.1 (52,379.6) | 0.032 (0.030) | 0.054 (0.048) | 0.0 |
| 2. BoW + MLP (deep learning) | 1.000 | 0.894 | 0.944 | 0.947 | 2.756 (2.641) | 69,868.5 (71,361.7) | 0.120 (0.118) | 0.146 (0.167) | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.851 | 0.933 | 0.890 | 0.974 | 49.397 (50.060) | 41.8 (43.4) | 28.831 (29.798) | 37.378 (39.006) | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.961 | 0.942 | 0.951 | 0.947 | 49.589 (50.275) | 53.9 (52.3) | 22.171 (23.154) | 30.831 (33.512) | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.961 | 0.942 | 0.951 | 0.947 | 50.136 (50.790) | 52.8 (51.6) | 26.668 (21.743) | 39.014 (29.261) | 0.0 |
| 4a. LLM nhỏ zero-shot | 0.833 | 1.000 | 0.909 | 0.909 | 0.000 | 0.776 (0.850) | 590.181 (663.683) | 3,235.970 (2,688.444) | 0.0 |
| 4b. LLM lớn zero-shot | 0.870 | 1.000 | 0.930 | 0.930 | 0.000 | 0.948 (0.935) | 572.883 (569.147) | 2,717.832 (2,776.343) | 0.0 |
| 4c. LLM nhỏ few-shot | 0.909 | 1.000 | 0.952 | 0.952 | 0.000 | 0.665 (0.846) | 855.989 (626.812) | 2,897.642 (3,256.205) | 0.0 |

Ghi chú:
- Hàng LLM chạy trên Groq (`openai/gpt-oss-20b` là model nhỏ, `openai/gpt-oss-120b` là model lớn) và chỉ đo trên SUB (40 tin), nên so chất lượng bằng cột `F1 (SUB 40)`.
- Cột `USD / 1k tin` của hàng LLM hiển thị 0.0 vì bảng giá `PRICE_PER_1M` trong notebook không có giá của hai model `gpt-oss` (model không có trong bảng thì được tính 0). Đây **không** có nghĩa là LLM miễn phí. Chi phí ước tính cho sản phẩm thật nằm ở mục 6.
- SUB chỉ có 40 tin, nên chênh lệch dưới khoảng 0.05 F1 coi như hòa.

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất (đều lệch theo hướng thực tế tốt hơn dự đoán):

1. **`dl_curve` (Thí nghiệm 2.A):**
   - *Đoán:* F1 của MLP "xấp xỉ bằng hoặc chỉ nhỉnh hơn một chút" so với LogReg.
   - *Thực tế:* MLP đạt F1 `0.944`, LogReg `0.863` (tăng `+0.081`). Precision vẫn `1.000` nhưng recall tăng từ `0.760` lên `0.894`.
   - *Vì sao lệch:* Mình đã đánh giá thấp vai trò của hidden layer trên đầu vào BoW. MLP bắt được tổ hợp phi tuyến giữa các từ nên bỏ sót ít spam hơn, trong khi LogReg chỉ cộng trọng số từng từ.

2. **`dl_relu` (Thí nghiệm 2.B):**
   - *Đoán:* Bỏ ReLU thì mạng tương đương LogReg tuyến tính, F1 tụt từ `0.944` về khoảng `0.86`.
   - *Thực tế:* Không có ReLU, F1 test vẫn `0.944` (không tụt).
   - *Vì sao lệch:* Hai lớp Linear liên tiếp vẫn chỉ biểu diễn được một hàm tuyến tính, nên về mặt biểu diễn nó cùng lớp hàm với LogReg. Vậy chênh lệch `0.944` so với `0.863` không đến từ tính phi tuyến của ReLU. Mình nghi ngờ nguyên nhân nằm ở cách huấn luyện: MLP dùng Adam, mini-batch, 128K tham số và dừng theo val loss, còn LogReg dùng gradient descent thường (`lr=0.5`, 300 epoch). Đây là giả thuyết, mình **chưa kiểm chứng** bằng thí nghiệm riêng.

3. **`llm_fewshot` (Thí nghiệm 4.B):**
   - *Đoán:* F1 của model nhỏ giữ nguyên hoặc tăng nhẹ (`~0.90–0.93`).
   - *Thực tế:* F1 trên SUB tăng từ `0.909` (ở lần chạy lẻ đầu là `0.900`) lên `0.952`, vượt cả model lớn zero-shot (`0.930`).
   - *Vì sao lệch:* Chỉ 4 ví dụ mẫu đã giúp model nhỏ bớt bắt nhầm tin ham thành spam: precision tăng từ `0.833` lên `0.909`, recall giữ `1.000`. Lưu ý SUB chỉ 40 tin nên mức chênh này chỉ nên coi là gợi ý, không phải kết luận chắc chắn.

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** `1. BoW + LogReg`. F1 từ `0.889` (SUB, test thường) xuống `0.333` trên 50 case khó, accuracy chỉ còn `52%`. Model tốt nhất là `4d. LLM nhỏ + prompt phòng thủ` (trong output Cell 87 của notebook đạt accuracy `90%`, F1 `0.918` — sai 5/50 case; ở lần chạy trước `Run all` đạt accuracy `92%`, F1 `0.935`).
- **Nhóm case BoW sai nhiều nhất** (% đoán đúng theo heatmap Cell 87): *Spam giọng tự nhiên* `10%`, *Thao túng / lừa đảo* `12%`, *Né từ khoá* `17%`. BoW đúng `100%` ở nhóm *"Tưởng spam nhưng không phải"* và *"Nói về spam"*, tức là nó gần như luôn đoán `ham` khi không thấy từ khoá quen thuộc.
- **Hai case cụ thể khiến model sai và cơ chế (trích từ output Cell 88):**
  1. *Case #2* (`ham_looks_spam`, thật = `ham`): `"FREE pizza in the office kitchen right now, come quick before it's gone"`. Model `4d` đoán sai thành `spam` (cùng với Case #8 `"Cash prize for the quiz night..."` và Case #9 `"Txt me ur address and I'll send the free tickets..."`). Cơ chế: có `"FREE"`, `"right now"`, `"Cash prize"`, `"free tickets"` và giọng giục giã, đúng những dấu hiệu bề mặt của spam, nên LLM nghiêng về bắt nhầm khi không có ngữ cảnh hội thoại trước đó cho biết đây là tin đồng nghiệp/bạn bè.
  2. *Case #47* (`talks_about_spam`, thật = `ham`): `"lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for that?"`. Model `4d` và các mô hình Embedding (`3a` đúng 25%, `3b/3c` đúng 50% ở nhóm này) đều dễ đoán sai thành `spam`. Cơ chế: tin chứa gần như toàn bộ từ khoá và ngữ nghĩa của một tin trúng thưởng (`won £1000`, `call 0906`), nhưng thực chất là người dùng đang kể lại để chê cười — đòi hỏi mô hình phải phân biệt được *nhắc đến spam* và *hành vi gửi spam*.
- Ngoài ra, các case như #13 (`natural_spam`: `"Hope you're well! Quick reminder that your photos from the event are ready..."`) và #31 (`obfuscation`: `c.a.s.h r.e.w.a.r.d`) thuộc đúng hai nhóm mà BoW chỉ đoán đúng `10%` và `17%`: văn bản không chứa token spam quen thuộc trong vocab 2.000 từ, hoặc từ khoá bị tách bằng dấu chấm nên không khớp token nào.
- **Prompt injection:** Ở nhóm này, trong Cell 87 `BoW` đúng `67%`, `4a` (prompt thường) đúng `67%` (ở lần chạy trước là `83%`), còn `4d` (prompt phòng thủ dùng thẻ `<sms>`) đạt tuyệt đối **`100%`**. Tổng cộng trên cả 50 case, `4d` đạt accuracy `90%` (`F1 = 0.918`) so với `84%` (`F1 = 0.871`) của `4a`. Có `0` case mà tất cả model cùng sai.

## 5. Thử thách cá nhân

- **Đề theo SEED 8476 (`SEED % 6 = 4`, Đề #5):** *Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.*
- **Cách làm:** Viết 4 tin trong Cell 96, vector hoá bằng `vectorize` rồi dự đoán bằng LogReg BoW (Phần 1), và mã hoá bằng MiniLM rồi dự đoán bằng `3b. Encoder + LogReg`.
- **Kết quả thật (output Cell 96):**

| # | Tin nhắn | Thật | BoW | Enc+LogReg |
|---|---|---|---|---|
| 1 | `Your bank account has been suspended due to suspicious activity, verify immediately at tinyurl.com/bank99` | spam | ham ✗ | ham ✗ |
| 2 | `Exclusive job offer: work 2 hours daily from home and earn 500 dollars, reply with your resume` | spam | spam ✓ | spam ✓ |
| 3 | `Hey bro, are you free this afternoon? Let's grab some coffee together` | ham | ham ✓ | ham ✓ |
| 4 | `We played so well and managed to win the championship match last night!` | ham | ham ✓ | ham ✓ |

- **Giải thích từng tin:**
  - *Tin 1 (cả hai sai):* Tin lừa đảo kiểu ngân hàng bị khoá tài khoản không có từ khoá spam cổ điển, nên BoW không có tín hiệu. Embedding cũng không cứu được. Giả thuyết của mình: bộ SMS Spam Collection chủ yếu là spam kiểu trúng thưởng / nhạc chuông / số dịch vụ của Anh, ít tin phishing ngân hàng, nên LogReg trên embedding chưa học được vùng này. Mình chưa kiểm chứng bằng cách thử thêm tin phishing khác.
  - *Tin 2 (cả hai đúng):* Có `"earn"`, `"reply"`, `"offer"`, các tín hiệu mà BoW đã học được từ tập train, nên BoW không cần đến embedding.
  - *Tin 3 và 4 (cả hai đúng):* Dù chứa `"free"` và `"win"`, cả hai model vẫn đoán đúng là `ham`, vì ngữ cảnh còn lại là hội thoại thân mật (`"hey bro"`, `"coffee"`, `"championship match"`).
- **Kết luận:** Với 4 tin này, hai model cho kết quả **giống hệt nhau** (đều `3/4`), nên phần dự đoán ban đầu của mình ("BoW sẽ bị 'free'/'win' đánh lừa, Embedding thì không") **không đúng** trong thí nghiệm này. Với chỉ 4 tin, không thể kết luận embedding hơn BoW hay ngược lại.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu:** 10 triệu SMS/ngày (khoảng 116 tin/giây trung bình), `p95 < 50 ms/tin`.
- **Phương án chọn:** `3b. Encoder + LogReg` nếu ưu tiên chất lượng, hoặc `2. BoW + MLP` nếu ưu tiên tốc độ và tài nguyên.
  - `2. BoW + MLP`: `F1 test = 0.944`, `p95 = 0.146–0.167 ms`, khoảng `70,000–71,000 tin/s` trên CPU. Nhanh hơn ngưỡng SLA hơn 300 lần.
  - `3b. Encoder + LogReg`: `F1 test = 0.951` (cao nhất trong nhóm không dùng LLM), `p95 = 30.8–33.5 ms`, đạt SLA nhưng chỉ còn dư khoảng 16–19 ms.
- **Vì sao không dùng thuần LLM:** `p95` của `4a/4b/4c` là `2,688` đến `3,256 ms`, chậm hơn SLA 50 ms khoảng 50 đến 65 lần, và mỗi lần gọi tuần tự chỉ đạt chưa đến 1 tin/giây. Về chi phí, theo phép tính tay (`gpt-6-luna`, 100 token input + 1 token output): 1 triệu tin tốn `10.50 USD`, nên 300 triệu tin/tháng khoảng **`3,150 USD/tháng`**. Đây là ước tính theo giá model `gpt-6-luna` trong đề; mình dùng Groq nên chi phí thật sẽ khác.
- **Đánh đổi chấp nhận:** Với `3b` phải tốn khoảng 50 giây để encode tập train một lần (`train_s` ~49.6–50.3s), `p95` cao hơn BoW hàng trăm lần, và chất lượng phụ thuộc vào phân phối tập train (xem tin phishing ở mục 5 và nhóm case khó ở mục 4).
- **Thiết kế lai (router):** Dùng `BoW + MLP` cho toàn bộ lưu lượng; tin có xác suất spam $p < 0.1$ thì chốt `ham`, $p > 0.9$ thì chốt `spam`. Chỉ các tin lưỡng lự ($0.1 \le p \le 0.9$) mới chuyển sang LLM (kèm prompt phòng thủ). Cách này giữ `p95` thấp và chi phí gần bằng 0 cho phần lớn tin, còn LLM chỉ xử lý số ít tin khó. Tỉ lệ tin rơi vào vùng lưỡng lự mình **chưa đo** trong lab này.

## 7. AI log và điều đã học

### AI log

Mình dùng AI (trợ lý lập trình trong Antigravity) làm mentor suốt buổi lab. Ghi chú trung thực theo từng phần:

| Phần | Mình hỏi AI gì | AI trả lời / viết gì | Mình sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 0 | Giải thích data leakage, cách chia train/val/test | Giải thích khái niệm, hướng dẫn chạy từng ô | Tự điền `STUDENT_ID`, tự trả lời Câu hỏi 0.1 |
| Phần 1 | Gợi ý cho `vectorize`, `train_logreg` (không đưa code đầy đủ) | Giải thích ý tưởng, gợi ý | Tự viết code. Gặp 2 lỗi `return X` thụt lề sai và `b = b = lr * grad_b`, AI chỉ ra nguyên nhân nhờ loss tăng bất thường và recall = 1.0, mình tự sửa và chạy lại |
| Phần 2 | Gợi ý `build_mlp`, `train_mlp` | Giải thích 5 dòng huấn luyện PyTorch | Tự viết code, tự trả lời Câu hỏi 2.A và Giải thích 2 |
| Phần 3 | Gợi ý `cosine_similarity`, `top_k_similar`, `knn_predict` | Giải thích chuẩn hoá L2 và nhân ma trận, đưa mã giả cho kNN | Tự viết code. Gặp `NameError: scores` ở `top_k_similar`, mình tự thêm bước tính `scores` |
| Phần 4 | Hướng dẫn `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` (vùng được dùng AI) | **AI đưa code mẫu đầy đủ cho cả 4 hàm** | Mình dán code vào notebook, chạy và đọc hiểu. Gặp lỗi thiếu 3 dấu ngoặc kép trong prompt, sửa lại. Chạy các ô test mock cho cả hai hàm |
| Phần 4 | Lỗi `RateLimitError 429` khi dùng Gemini free tier | Giải thích giới hạn 15 request/phút | Quyết định đổi sang Groq, chạy lại 40 tin thành công |
| Phần 5 | Cách đọc biểu đồ Pareto, heatmap 50 case khó | Giải thích đường Pareto và cơ chế prompt phòng thủ thẻ `<sms>` | Đối chiếu với biểu đồ và bảng của chính mình. Nhiều câu trả lời Câu hỏi 3, 4, 5, 6 và mục phản ánh cuối bài do AI soạn khung, mình chỉnh sửa lại |
| Phần 5 | Tính tay | AI đưa đáp án và công thức | Mình viết lại các bước ra giấy và chụp ảnh |
| Báo cáo này | Soạn nội dung các mục 2 đến 7 và rà soát tính nhất quán | AI soạn bản đầu và kiểm tra chéo với output của `day01_lab.ipynb` và `lab1_report.json` | Đã đối chiếu và đồng bộ 100% số liệu giữa notebook, file JSON và báo cáo cá nhân trước khi nộp |

### Điều đã học

1. Điểm trên tập test thường có thể rất lạc quan: `BoW + LogReg` đạt F1 `0.889` trên SUB nhưng tụt còn `0.333` trên 50 case khó, vì từ vựng học từ train không đại diện cho spam né từ khoá hay giọng tự nhiên.
2. Trong sản phẩm thật, độ trễ và chi phí quyết định phương án: LLM có chất lượng tốt nhưng `p95` vượt SLA hàng chục lần, trong khi `BoW + MLP` và `Encoder + LogReg` đạt SLA với F1 từ `0.944` đến `0.951`.
3. Dự đoán trước rồi đối chiếu giúp mình nhận ra chỗ hiểu sai: mình tưởng bỏ ReLU sẽ làm mô hình tụt về mức LogReg, nhưng kết quả không đổi (`0.944`), nên chênh lệch giữa MLP và LogReg không chỉ đến từ tính phi tuyến.
