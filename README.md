# Hệ thống Trợ lý Hỏi đáp & Kiểm duyệt Phác đồ Điều trị Y khoa (Medical RAG Assistant)

Dự án này là hệ thống chatbot RAG (Retrieval-Augmented Generation) chuyên dụng cho y khoa, tích hợp **Hệ chuyên gia (Expert System)** làm màng lọc an toàn và cơ chế duyệt dữ liệu lâm sàng từ bác sĩ **(Human-in-the-Loop)** nhằm khép kín vòng lặp tự học **(Data Flywheel)**.

---

## 🏗️ 1. Cấu trúc Dự án (RAG Project Structure)

Mã nguồn được phân chia theo kiến trúc Modular rõ ràng để phục vụ việc bảo trì và nâng cấp:

```text
rag_project/
├── .env                       # Cấu hình khóa API Gemini (lưu cục bộ)
├── .gitignore                 # Bỏ qua các tệp virtualenv, dữ liệu ChromaDB khi push
├── config.yaml                # Tham số mô hình LLM, cấu hình chia nhỏ văn bản (chunking)
├── requirements.txt           # Thư viện phụ thuộc (google-genai, chromadb, colorama...)
├── main.py                    # Bộ điều phối trung tâm (Orchestrator) chạy ứng dụng CLI
│
├── config/
│   └── expert_rules.json       # Định nghĩa các từ khóa khẩn cấp và ngưỡng tin cậy
│
├── data/
│   ├── dataset.json            # Cơ sở tri thức phác đồ điều trị gốc (JSON)
│   └── chroma_db/              # Thư mục lưu trữ cơ sở dữ liệu vector ChromaDB
│
├── src/
│   ├── ingestion/             # Đọc dữ liệu thô và làm sạch văn bản y khoa
│   ├── chunking/              # Cắt nhỏ tài liệu thành các đoạn văn (chunks)
│   ├── embeddings/            # Chuyển đổi văn bản thành vector (Gemini Embeddings/Local)
│   ├── vectordb/              # Tương tác với ChromaDB (thêm vector, truy vấn Cosine)
│   ├── retrieval/             # Thực hiện tìm kiếm và xếp hạng kết quả tương đồng
│   ├── llm/                   # Quản lý phiên kết nối và gọi API Gemini
│   ├── prompts/               # Tiêu bản prompt chèn ngữ cảnh (RAG Prompt Template)
│   └── utils/                 # Các hàm phụ trợ ghi log, cấu hình hệ thống
│
└── tests/
    └── test_app.py            # Kịch bản kiểm thử tự động của hệ thống
```

---

## 🔬 2. Các tính năng cốt lõi và Cơ chế hoạt động

Hệ thống được thiết kế để giải quyết bài toán cốt lõi: **LLM ảo tưởng thông tin y tế gây nguy hiểm cho người bệnh**.

### A. Lớp kiểm duyệt khẩn cấp (Emergency Alarm)
Hệ thống sử dụng Hệ chuyên gia để lọc trước câu hỏi của người dùng. Nếu phát hiện các triệu chứng nguy kịch đe dọa tính mạng (được định nghĩa trong `expert_rules.json` như: *"đau ngực dữ dội"*, *"co giật"*, *"ngừng tuần hoàn"*), chatbot sẽ lập tức đưa ra cảnh báo đi cấp cứu 115 mà **không đi qua mô hình LLM**, đảm bảo an toàn tuyệt đối và tốc độ phản hồi tối đa.

### B. Định tuyến theo độ tin cậy (Expert System Router)
Với các câu hỏi thông thường, bộ truy xuất sẽ so khớp ngữ nghĩa trên cơ sở dữ liệu vector ChromaDB và tính toán điểm tương đồng (Similarity Score):
*   **Điểm số $\ge$ Ngưỡng cao (mặc định 0.80)**: Tìm thấy phác đồ chuẩn xác. Hệ thống ghép nội dung phác đồ vào prompt và gọi Gemini API trả lời tự nhiên (RAG Direct).
*   **Điểm số nằm trong khoảng [0.50 - 0.79]**: Tìm thấy bệnh lý tương đối khớp nhưng cấu trúc câu của người dùng bị thay đổi. Chatbot hỏi xác nhận lại thông tin bệnh lý trước khi phản hồi (RAG Confirmation).
*   **Điểm số < Ngưỡng thấp (0.50)**: Không có dữ liệu tin cậy. Kích hoạt luồng con người phê duyệt (Human-in-the-Loop) để tránh việc LLM bịa ra câu trả lời.

### C. Trình giải thích logic (Explanation Facility)
Đáp ứng tiêu chí **AI minh bạch (Explainable AI)**, sau mỗi câu trả lời, chatbot hiển thị rõ đường đi suy diễn: luật nào đã được kích hoạt, điểm tương đồng đo được là bao nhiêu và cơ chế định tuyến tương ứng.

### D. Vòng lặp tự học (Data Flywheel)
Khi luồng **Human-in-the-Loop** được kích hoạt, Bác sĩ Trưởng khoa có thể nhập trực tiếp phác đồ điều trị mới. Hệ thống sẽ tự động thực hiện đồng thời:
1.  Ghi đè bản ghi mới vào tệp tin gốc `data/dataset.json`.
2.  Cắt nhỏ dữ liệu mới thành các đoạn văn, chuyển đổi thành vector nhúng và nạp trực tiếp vào CSDL Vector ChromaDB hiện hành mà không cần khởi động lại ứng dụng.
Từ các lượt hỏi sau, chatbot đã học thuộc và tự động trả lời được ngay.

---

## 🚀 3. Hướng dẫn cài đặt và Khởi chạy

### Bước 1: Cài đặt môi trường Python
```bash
pip install -r requirements.txt
```

### Bước 2: Thiết lập biến môi trường
Tạo file `.env` tại thư mục gốc của dự án và điền API Key Gemini của bạn:
```text
GEMINI_API_KEY=AIzaSyYourGeminiAPIKeyHere
```

### Bước 3: Khởi chạy Chatbot CLI
```bash
python main.py
```

---

## 📊 4. Nhật ký chạy thử thực tế (Console Demo Logs)

Dưới đây là log màn hình thực tế ghi lại các ca thử nghiệm các tính năng của chatbot để người xem dễ hình dung luồng chạy mà không cần tải code về chạy thử:

### Ca 1: Kích hoạt Luật khẩn cấp (Emergency Alarm)
```text
Bác sĩ: Bệnh nhân đột nhiên bị co giật và khó thở dữ dội, phải làm sao?

[EXPLANATION FACILITY - ĐƯỜNG SUY DIỄN CỦA HỆ CHUYÊN GIA]
  - Cơ chế kích hoạt: RULE_EMERGENCY_ALARM
  - Lý do: Kích hoạt LUẬT KHẨN CẤP: Dấu hiệu đe dọa tính mạng lâm sàng. Chuyển thẳng cảnh báo khẩn cấp.
--------------------------------------------------

Trợ lý ảo: [CẢNH BÁO Y TẾ KHẨN CẤP]
Phát hiện triệu chứng nguy kịch đe dọa tính mạng. Vui lòng lập tức đưa bệnh nhân đến cơ sở y tế gần nhất hoặc gọi số điện thoại cấp cứu 115 ngay lập tức!
================================================================================
```

### Ca 2: Khớp phác đồ trực tiếp (RAG Direct)
```text
Bác sĩ: Biến chứng nguy hiểm của bệnh Basedow là gì?

[EXPLANATION FACILITY - ĐƯỜNG SUY DIỄN CỦA HỆ CHUYÊN GIA]
  - Cơ chế kích hoạt: RAG_DIRECT
  - Lý do: LUẬT ĐỘ TIN CẬY CAO: Điểm số (0.87) >= Ngưỡng cao (0.80). Đủ an toàn để trả lời bằng RAG.
--------------------------------------------------

Trợ lý ảo: Bệnh Basedow (cường giáp tự miễn) có thể dẫn đến các biến chứng nguy hiểm sau:
1. Mắt lồi ác tính: Nhãn cầu bị đẩy nhanh ra ngoài hốc mắt gây mù hoặc loét giác mạc.
2. Tim mạch: Gây loạn nhịp tim (cuồng nhĩ, rung nhĩ) và suy tim do tim phải hoạt động quá tải.
3. Cơn bão giáp (nhiễm độc giáp cấp): Biến chứng cấp tính nguy kịch nhất với tỷ lệ tử vong rất cao.
4. Loãng xương: Do hormone tuyến giáp tăng cao cản trở hấp thu canxi vào xương.
================================================================================
```

### Ca 3: Khớp mờ cần hỏi lại (RAG Confirmation)
```text
Bác sĩ: rụng nhiều tóc có nhiều mụn mủ trên đầu

[EXPLANATION FACILITY - ĐƯỜNG SUY DIỄN CỦA HỆ CHUYÊN GIA]
  - Cơ chế kích hoạt: RAG_CONFIRMATION
  - Lý do: LUẬT ĐỘ TIN CẬY VỪA PHẢI: Điểm số (0.68) nằm trong khoảng nghi vấn. Cần xác nhận thông tin bệnh lý.
--------------------------------------------------

Trợ lý ảo: Hệ thống: Có phải bạn đang muốn tìm hiểu về phác đồ điều trị của bệnh lý: Rụng tóc thành từng mảng lớn không?
Xác nhận (y/n): y

Trợ lý ảo: Biểu hiện rụng tóc kèm đóng vảy da đầu và mụn mủ nhiều khả năng là bệnh nấm da đầu nặng. Bạn cần đưa bệnh nhân đến khám chuyên khoa Da liễu để soi nấm và có phác đồ kháng nấm chính xác.
================================================================================
```

### Ca 4: Tự học tri thức mới (Human-in-the-Loop & Flywheel)
```text
Bác sĩ: Lab nghiên cứu của chúng ta nằm ở phòng bao nhiêu?

[EXPLANATION FACILITY - ĐƯỜNG SUY DIỄN CỦA HỆ CHUYÊN GIA]
  - Cơ chế kích hoạt: HUMAN_IN_THE_LOOP
  - Lý do: LUẬT ĐỘ TIN CẬY THẤP: Điểm số (0.12) < Ngưỡng an toàn (0.50). Kích hoạt Human-in-the-loop.
--------------------------------------------------

[HUMAN-IN-THE-LOOP] Hệ thống không tìm thấy phác đồ chính thức cho câu hỏi này.
Bạn có muốn đóng vai Bác sĩ Trưởng khoa cập nhật phác đồ điều trị này không? (y/n): y
Nhập tên bệnh lý chính thức: Vị trí phòng Lab nghiên cứu
Mô tả các triệu chứng chính: Địa chỉ, phòng làm việc
Mô tả phác đồ điều trị chuẩn: Phòng nghiên cứu Lab của chúng ta đặt tại phòng 502, Tòa nhà A, Khu công nghệ cao.
Mô tả các dấu hiệu nguy hiểm cần cảnh báo: Không có

Trợ lý ảo: Hệ thống đã lưu nhận phác đồ bệnh 'Vị trí phòng Lab nghiên cứu' (ID: 101). Bánh đà dữ liệu đã tự cập nhật!
================================================================================

Bác sĩ: phòng lab ở đâu thế nhỉ?

[EXPLANATION FACILITY - ĐƯỜNG SUY DIỄN CỦA HỆ CHUYÊN GIA]
  - Cơ chế kích hoạt: RAG_DIRECT
  - Lý do: LUẬT ĐỘ TIN CẬY CAO: Điểm số (0.83) >= Ngưỡng cao (0.80). Đủ an toàn để trả lời bằng RAG.
--------------------------------------------------

Trợ lý ảo: Phòng nghiên cứu Lab của chúng ta đặt tại phòng 502, Tòa nhà A, Khu công nghệ cao.
================================================================================
```

---

## 🔮 5. Định hướng Cải tiến và Nâng cấp tiếp theo

Hệ thống đang hoạt động ổn định ở quy mô Lab thử nghiệm. Để ứng dụng thực tế trên lâm sàng lớn hơn, các cải tiến kỹ thuật đề xuất bao gồm:
1.  **Nâng cấp mô hình tìm kiếm ngữ nghĩa**: Thay thế thuật toán TF-IDF bằng việc sử dụng mô hình Embedding tiếng Việt chuyên dụng như **PhoBERT** nhằm hiểu chính xác các từ đồng nghĩa y khoa (ví dụ: *"nhức đầu"* và *"đau đầu"*).
2.  **Phân quyền người duyệt dữ liệu**: Bổ sung cơ chế bảo mật (như JWT token hoặc phân quyền phân vai RBAC). Chỉ những tài khoản bác sĩ có chứng chỉ hành nghề mới được phê duyệt dữ liệu mới vào file `dataset.json` nhằm bảo vệ hệ thống khỏi thông tin rác.
3.  **Xây dựng UI Web**: Phát triển ứng dụng Web trực quan bằng **Streamlit** (đối với ứng dụng nghiên cứu) hoặc **FastAPI + React** (đối với ứng dụng thực tế) thay thế giao diện Console dòng lệnh hiện tại.
