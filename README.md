# Hệ thống Trợ lý Hỏi đáp & Kiểm duyệt Phác đồ Điều trị Y khoa (Medical RAG Assistant)

Dự án này xây dựng một hệ thống chatbot thông minh hỗ trợ các bác sĩ lâm sàng hỏi đáp nhanh về phác đồ điều trị y khoa. 

Hệ thống kết hợp ba trụ cột lý thuyết cốt lõi trong phòng nghiên cứu AI:
1.  **RAG (Retrieval-Augmented Generation)**: Truy xuất các tài liệu phác đồ liên quan từ cơ sở dữ liệu vector để làm ngữ cảnh đầu vào cho mô hình LLM (Gemini), giúp câu trả lời tuyệt đối chính xác và loại bỏ ảo tưởng lâm sàng.
2.  **Hệ chuyên gia (Expert System)**: Đóng vai trò lớp kiểm duyệt an toàn (Safety Guardrail) thông qua các luật logic để định tuyến câu hỏi (nhận diện trường hợp khẩn cấp cấp cứu, phân luồng theo độ tin cậy của kết quả so khớp).
3.  **Vòng lặp tự học & Can thiệp con người (Human-in-the-Loop & Data Flywheel)**: Cho phép Bác sĩ Trưởng khoa trực tiếp duyệt và bổ sung phác đồ điều trị mới cho các câu hỏi lạ, hệ thống sẽ tự động cập nhật cơ sở dữ liệu và nạp trực tiếp vector vào CSDL mà không cần khởi động lại.

## 🌟 Tính năng nổi bật (Key Features)

*   **Hệ thống Hybrid (RAG + Expert System):** Đóng vai trò lớp kiểm duyệt an toàn y khoa. Tự động chặn và định tuyến các câu hỏi khẩn cấp chuyển thẳng đến hướng dẫn cấp cứu 115 mà không đi qua AI.
*   **Định tuyến theo độ tin cậy (Confidence Routing):** Phân luồng câu hỏi dựa trên điểm Cosine Similarity (tự động RAG trực tiếp, hỏi xác nhận bệnh lý, hoặc kích hoạt Human-in-the-loop).
*   **Vòng lặp tự học khép kín (Data Flywheel):** Bác sĩ có thể duyệt và bổ sung trực tiếp phác đồ điều trị mới vào database dạng JSON và ChromaDB trực tiếp trong thời gian chạy (Runtime) mà không cần khởi động lại.
*   **Dự phòng ngoại tuyến (Offline Fallback Embedder):** Khi không có internet hoặc API Key, hệ thống tự động kích hoạt thuật toán băm từ vựng (Word-Level Hashing với djb2) để tính tương đồng ngữ nghĩa cục bộ.
*   **Chatbot QA TF-IDF phụ trợ:** Tích hợp thêm chatbot QA có lịch sử chat dài hơi (`src/app.py`) hoạt động hoàn toàn offline với cơ chế tính TF-IDF từ đầu.
*   **Bộ kiểm thử tự động (Unit Test Suite):** Mã nguồn đi kèm các kịch bản unittest giúp kiểm chứng tính chính xác của module Loader, Chunker, và Embedder.

---

## 🏗️ 1. Cấu trúc Dự án (Project Structure)

Mã nguồn dự án được tổ chức phân lớp rõ ràng theo đúng chuẩn kiến trúc modular:

```text
rag_project/
├── .env                       # Lưu khóa bí mật API Gemini (không đẩy lên git)
├── .gitignore                 # Bỏ qua tệp tin rác, virtualenv và database khi git push
├── config.yaml                # Cấu hình tham số chunking, mô hình LLM và đường dẫn log
├── requirements.txt           # Danh sách các thư viện Python cần cài đặt
├── main.py                    # Bộ điều phối trung tâm (Orchestrator) chạy ứng dụng CLI
│
├── config/
│   └── expert_rules.json      # Tập luật của hệ chuyên gia (từ khóa cấp cứu, các ngưỡng điểm số)
│
├── data/
│   ├── dataset.json           # Cơ sở dữ liệu phác đồ lâm sàng gốc (JSON)
│   └── chroma_db/             # Thư mục lưu trữ tệp cơ sở dữ liệu vector của ChromaDB
│
├── logs/                      # Thư mục tự động ghi log lỗi vận hành
│
├── reports/
│   ├── report_0.md            # Báo cáo học tập dự án chi tiết (vòng đời dữ liệu, thuật toán...)
│   └── images/                # Nơi lưu trữ hình ảnh chạy thử (screenshots) làm bằng chứng demo
│       └── demo_1.png -> demo_10.png (đầy đủ 10 ảnh chạy thử thực tế)
│
├── src/                       # Thư mục chứa mã nguồn chính của các module
│   ├── __init__.py
│   ├── app.py                 # Chatbot QA phụ trợ có lịch sử chat (TF-IDF)
│   ├── download_dataset.py    # Script tải dữ liệu y tế từ Hugging Face
│   ├── ingestion/             # Load dữ liệu lâm sàng thô và chuẩn hóa tiếng Việt
│   ├── chunking/              # Chia nhỏ tài liệu thành các khối thông tin kèm metadata
│   ├── embeddings/            # Chuyển đổi văn bản thành vector (Gemini API hoặc Hashing Offline)
│   ├── vectordb/              # Tương tác với CSDL Vector ChromaDB (HNSW Cosine Space)
│   ├── retrieval/             # Tìm kiếm, xếp hạng và gộp ngữ cảnh tương đồng nhất
│   ├── llm/                   # Quản lý kết nối và phiên làm việc với Google Gemini API
│   ├── prompts/               # Tiêu bản câu lệnh chèn ngữ cảnh y khoa
│   └── utils/                 # Các hàm bổ trợ đọc cấu hình, ghi log lỗi
│
└── tests/                     # Thư mục chứa kịch bản kiểm thử tự động
    └── test_app.py            # Kịch bản Unit Test kiểm thử Loader, Chunker và Embedder
```

---

## 🔬 2. Cơ chế Hoạt động của các Module

*   **Module Ingestion (`src/ingestion/loader.py`)**: Đọc file dữ liệu phác đồ gốc lâm sàng từ định dạng JSON. Thực hiện làm sạch văn bản bằng biểu thức chính quy (Regex) để loại bỏ các ký tự đặc biệt, đưa về chữ thường và giữ lại tiếng Việt sạch.
*   **Module Chunking (`src/chunking/chunker.py`)**: Chia tài liệu của mỗi bệnh lý thành 3 phần rõ ràng: *Triệu chứng*, *Phác đồ điều trị*, và *Dấu hiệu cảnh báo lâm sàng*. Gán nhãn metadata đầy đủ cho từng phần để truy xuất chính xác thông tin cần tìm.
*   **Module Embeddings (`src/embeddings/embedder.py`)**: Mã hóa các chunk văn bản thành vector nhúng 768 chiều. Tích hợp cơ chế **Offline Fallback** (sử dụng Hashing Vectorizer kết hợp chuẩn hóa L2) giúp hệ thống tự tạo vector cục bộ khi mất mạng mà không bị dừng chương trình.
*   **Module VectorDB (`src/vectordb/vector_store.py`)**: Sử dụng thư viện ChromaDB tạo Persistent Client lưu trữ bền vững xuống đĩa cứng. Thiết lập chỉ mục khoảng cách Cosine Similarity để so khớp độ tương đồng ngữ nghĩa.
*   **Module Retrieval (`src/retrieval/retriever.py`)**: Thực hiện truy vấn Top-K đoạn văn bản tương đồng nhất, xếp hạng điểm số và gộp thành một chuỗi ngữ cảnh RAG hoàn chỉnh.
*   **Module LLM (`src/llm/llm_client.py`)**: Cấu hình kết nối tới Google Gemini API (tự động tương thích cả SDK cũ và SDK mới). Gửi prompt làm giàu ngữ cảnh kèm chỉ thị hệ thống để đảm bảo câu trả lời chuẩn xác.

---

## 🚀 3. Hướng dẫn cài đặt và Khởi chạy

### Bước 1: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 2: Thiết lập cấu hình biến môi trường
Tạo file `.env` tại thư mục gốc dự án và điền API Key Gemini của bạn:
```text
GEMINI_API_KEY=Khóa_API_Gemini_Của_Bạn
```
*(Nếu muốn chạy offline hoàn toàn, bạn hãy để trống khóa này. Hệ thống sẽ tự động kích hoạt cơ chế băm vector offline dự phòng).*

### Bước 3: Tải bộ dữ liệu y khoa mẫu từ Hugging Face
Hệ thống tích hợp sẵn script tải tự động dữ liệu y học tiếng Việt thực tế từ Hugging Face (edoctor/vinmec). Hãy chạy lệnh sau:
```bash
python src/download_dataset.py
```

### Bước 4: Khởi chạy Trợ lý ảo
```bash
python main.py
```

### Bước 5: Chạy thử nghiệm kiểm thử tự động (Unit Tests)
Để đảm bảo các module hoạt động chính xác trước khi vận hành, hãy khởi chạy bộ kiểm thử tự động:
```bash
python tests/test_app.py
```

---

## 📊 4. Minh họa Hoạt động Lâm sàng (Console Demo Preview)

Để tránh lặp lại toàn bộ hình ảnh đã được trình bày chi tiết trong báo cáo y khoa, phần này chỉ hiển thị **2 hình ảnh minh họa cốt lõi** đại diện cho khả năng vận hành của trợ lý:

### 📸 Hình A: Khởi động hệ thống & Lập chỉ mục Vector tự động (`main.py`)
*Mô tả:* Hệ thống tự phát hiện cơ sở dữ liệu vector rỗng trong lần khởi chạy đầu tiên, tự động gọi loader, phân đoạn văn bản (chunking) và băm vector đẩy dữ liệu vào ChromaDB.
![Khởi động và lập chỉ mục CSDL Vector](reports/images/demo_7.png)

---

### 📸 Hình B: Luồng xử lý RAG trực tiếp khi độ tin cậy cao (`main.py`)
*Mô tả:* Người dùng hỏi thông tin có sẵn trong phác đồ, hệ chuyên gia phát hiện điểm tương đồng cao và định tuyến câu hỏi kết hợp ngữ cảnh RAG để LLM sinh câu trả lời chính xác.
![Luồng RAG trực tiếp độ tin cậy cao](reports/images/demo_9.png)

---

### 📂 Danh mục đầy đủ 10 Hình ảnh chạy thử nghiệm lâm sàng (Xem trong Báo cáo)

Tất cả 10 ảnh bằng chứng chạy thử (screenshots) đã được lưu trữ và mô tả chi tiết tại **Mục VII** của tệp báo cáo **[reports/report_0.md](file:///D:/rag_project/reports/report_0.md)**:

| Mã Hình | Kịch bản thử nghiệm (Console flow) | Script thực thi | Vị trí trong Báo cáo |
| :--- | :--- | :--- | :--- |
| **Hình 1** | Hỏi đáp về mụn ở cuống lưỡi (TF-IDF) | `src/app.py` | Mục VII - Hình 1 |
| **Hình 2** | Bộ lọc an toàn - Phát hiện co giật & khó thở (Cấp cứu 115) | `main.py` | Mục VII - Hình 2 |
| **Hình 3** | Định tuyến tương đồng trung bình (Khớp phác đồ Sốt rét) | `main.py` | Mục VII - Hình 3 |
| **Hình 4** | Hỏi đáp về phòng tránh cận thị học đường | `src/app.py` | Mục VII - Hình 4 |
| **Hình 5** | Hỏi đáp về cách sơ cứu đau răng có mủ | `src/app.py` | Mục VII - Hình 5 |
| **Hình 6** | Can thiệp con người (HITL) - Nạp động bệnh mới (Thủy đậu) | `main.py` | Mục VII - Hình 6 |
| **Hình 7** | Khởi động hệ thống & Indexing lần đầu vào ChromaDB | `main.py` | Mục VII - Hình 7 |
| **Hình 8** | Bộ lọc an toàn - Phát hiện triệu chứng khẩn cấp lần 2 | `main.py` | Mục VII - Hình 8 |
| **Hình 9** | Định tuyến tương đồng cao - Trả lời qua RAG trực tiếp | `main.py` | Mục VII - Hình 9 |
| **Hình 10**| Vòng lặp học tập tự động (Data Flywheel) - Sơ cứu bỏng bô | `src/app.py` | Mục VII - Hình 10 |

---


## 📝 5. Tài liệu Báo cáo chi tiết
Để xem chi tiết hơn về cơ chế toán học TF-IDF, thuật toán Cosine Similarity, mô tả chi tiết vòng đời dữ liệu (CREATE - USED - UPDATE - DELETE) và các định hướng nâng cấp hệ thống, vui lòng truy cập:
👉 **[Xem báo cáo chi tiết dự án (reports/report_0.md)](file:///D:/rag_project/reports/report_0.md)**
