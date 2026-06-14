# Hệ thống Trợ lý Hỏi đáp & Kiểm duyệt Phác đồ Điều trị Y khoa (Medical RAG Assistant)

Dự án này xây dựng một hệ thống chatbot thông minh hỗ trợ các bác sĩ lâm sàng hỏi đáp nhanh về phác đồ điều trị y khoa. 

Hệ thống kết hợp ba trụ cột lý thuyết cốt lõi trong phòng nghiên cứu AI:
1.  **RAG (Retrieval-Augmented Generation)**: Truy xuất các tài liệu phác đồ liên quan từ cơ sở dữ liệu vector để làm ngữ cảnh đầu vào cho mô hình LLM (Gemini), giúp câu trả lời tuyệt đối chính xác và loại bỏ ảo tưởng lâm sàng.
2.  **Hệ chuyên gia (Expert System)**: Đóng vai trò lớp kiểm duyệt an toàn (Safety Guardrail) thông qua các luật logic để định tuyến câu hỏi (nhận diện trường hợp khẩn cấp cấp cứu, phân luồng theo độ tin cậy của kết quả so khớp).
3.  **Vòng lặp tự học & Can thiệp con người (Human-in-the-Loop & Data Flywheel)**: Cho phép Bác sĩ Trưởng khoa trực tiếp duyệt và bổ sung phác đồ điều trị mới cho các câu hỏi lạ, hệ thống sẽ tự động cập nhật cơ sở dữ liệu và nạp trực tiếp vector vào CSDL mà không cần khởi động lại.

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
│   └── expert_rules.json       # Tập luật của hệ chuyên gia (từ khóa cấp cứu, các ngưỡng điểm số)
│
├── data/
│   ├── dataset.json            # Cơ sở dữ liệu phác đồ lâm sàng gốc (JSON)
│   └── chroma_db/              # Thư mục lưu trữ tệp cơ sở dữ liệu vector của ChromaDB
│
├── reports/
│   ├── report_0.md            # Báo cáo học tập dự án chi tiết (vòng đời dữ liệu, thuật toán...)
│   └── images/                # Nơi lưu trữ hình ảnh chạy thử (screenshots) làm bằng chứng demo
│       ├── demo_1.png
│       ├── demo_2.png
│       ├── demo_3.png
│       └── demo_4.png
│
├── src/                       # Thư mục chứa mã nguồn chính của các module
│   ├── __init__.py
│   ├── ingestion/             # Load dữ liệu lâm sàng thô và chuẩn hóa tiếng Việt
│   ├── chunking/              # Chia nhỏ tài liệu thành các khối thông tin kèm metadata
│   ├── embeddings/            # Chuyển đổi văn bản thành vector (Gemini API hoặc Hashing Offline)
│   ├── vectordb/              # Tương tác với CSDL Vector ChromaDB (HNSW Cosine Space)
│   ├── retrieval/             # Tìm kiếm, xếp hạng và gộp ngữ cảnh tương đồng nhất
│   ├── llm/                   # Quản lý kết nối và phiên làm việc với Google Gemini API
│   ├── prompts/               # Tiêu bản câu lệnh chèn ngữ cảnh y khoa
│   └── utils/                 # Các hàm bổ trợ đọc cấu hình, ghi log lỗi
└── tests/                     # Thư mục chứa kịch bản kiểm thử tự động
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

---

## 📊 4. Hình ảnh và Nhật ký Chạy thử Thực tế (Console Demo Screenshots)

Để chứng minh các tính năng của chatbot hoạt động ổn định và chính xác mà không cần bắt buộc phải tải code về cấu hình chạy thử, dưới đây là hình ảnh chụp trực tiếp từ màn hình dòng lệnh CMD khi chạy ứng dụng:

### 📸 Hình 1: Khởi chạy và Thử nghiệm RAG Direct (Khớp trực tiếp)
Hệ thống tự động lập chỉ mục CSDL Vector lần đầu và nạp 100 bản ghi vào ChromaDB. Khi hỏi về triệu chứng nổi mụn cuống lưỡi hoặc đau răng có mủ, điểm tương đồng đạt mức tuyệt đối (1.0000) và mức cao (0.8681), kích hoạt luật `RULE_HIGH_CONFIDENCE` để trả lời trực tiếp phác đồ.

![Khởi chạy và RAG Direct](reports/images/demo_2.png)

---

### 📸 Hình 2: Demo tính năng Tự học (Human-in-the-Loop & Data Flywheel)
Khi người dùng hỏi câu hỏi lạ: *"Bệnh sốt xuất huyết lây qua đường nào hả bác sĩ?"* nhưng cơ sở dữ liệu chưa có, điểm tương đồng thấp (0.4186) $\rightarrow$ Kích hoạt `HUMAN_IN_THE_LOOP`. Hệ thống hỏi ý kiến chuyên gia $\rightarrow$ Bác sĩ Trưởng khoa gõ phác đồ chuẩn $\rightarrow$ Hệ thống tự lưu vào `dataset.json` và cập nhật trực tiếp vào cơ sở dữ liệu vector ChromaDB.

![Luồng Tự học](reports/images/demo_4.png)

---

### 📸 Hình 3: Demo tính năng Xác nhận lại câu hỏi (RAG Confirmation)
Khi người dùng gõ câu hỏi lệch ngữ cảnh khá nhiều: *"Chào bác sĩ, tôi bị mấy con muỗi đốt có thể dẫn đến bệnh gì?"*, điểm số so khớp tương đồng đạt mức trung bình (0.5864) $\rightarrow$ Kích hoạt `RAG_CONFIRMATION`. Chatbot hỏi xác nhận lại ý định hỏi của người dùng trước khi truy xuất câu trả lời cụ thể.

![Hỏi xác nhận](reports/images/demo_1.png)

---

### 📸 Hình 4: Demo giao diện thêm dữ liệu của Quản trị viên (Admin Interface)
Khi gõ từ khóa `admin` hoặc `quản trị`, bộ định tuyến kích hoạt luật cứng `RULE_FORCE_ADMIN` để dẫn người dùng trực tiếp vào giao diện thêm dữ liệu bệnh lý lâm sàng mới bằng tay.

![Giao diện Admin](reports/images/demo_3.png)

---

## 📝 5. Tài liệu Báo cáo chi tiết
Để xem chi tiết hơn về cơ chế toán học TF-IDF, thuật toán Cosine Similarity, mô tả chi tiết vòng đời dữ liệu (CREATE - USED - UPDATE - DELETE) và các định hướng nâng cấp hệ thống, vui lòng truy cập:
👉 **[Xem báo cáo chi tiết dự án (reports/report_0.md)](file:///D:/rag_project/reports/report_0.md)**
