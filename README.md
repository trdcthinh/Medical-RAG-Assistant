# Hệ thống Trợ lý Hỏi đáp & Kiểm duyệt Phác đồ Điều trị Y khoa (Medical RAG Assistant)

Hệ thống chatbot thông minh hỗ trợ hỏi đáp phác đồ điều trị y tế lâm sàng sử dụng mô hình **RAG (Retrieval-Augmented Generation)** kết hợp với **Hệ chuyên gia (Expert System)** làm màng lọc an toàn và cơ chế duyệt dữ liệu **(Human-in-the-Loop)** nhằm khép kín vòng lặp tự học **(Data Flywheel)**.

---

## 🏗️ 1. Cấu trúc Dự án (Project Structure)

```text
rag_project/
├── .env                       # Khóa bảo mật API Gemini (lưu cục bộ)
├── .gitignore                 # Bỏ qua tệp tin rác và CSDL vector khi git push
├── config.yaml                # Cấu hình tham số mô hình và chunking
├── requirements.txt           # Danh sách thư viện cần cài đặt
├── main.py                    # Bộ điều phối trung tâm chạy chatbot (CLI)
│
├── config/
│   └── expert_rules.json       # Tập luật của hệ chuyên gia (cảnh báo khẩn cấp, ngưỡng định tuyến)
│
├── data/
│   ├── dataset.json            # Cơ sở dữ liệu phác đồ y học lâm sàng gốc (JSON)
│   └── chroma_db/              # Thư mục cơ sở dữ liệu vector ChromaDB
│
├── reports/
│   └── report_0.md            # BÁO CÁO DỰ ÁN CHI TIẾT VÀ ĐẦY ĐỦ (Xem ở đây)
│
├── src/                       # Thư mục mã nguồn chính (ingestion, chunking, vectordb, etc.)
└── tests/                     # Kịch bản kiểm thử tự động
```

---

## 🚀 2. Hướng dẫn cài đặt và Chạy nhanh

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Thiết lập API Key Gemini
Tạo file `.env` tại thư mục gốc của dự án và điền khóa API của bạn:
```text
GEMINI_API_KEY=AIzaSyYourGeminiAPIKey
```
*(Nếu chạy ngoại tuyến hoàn toàn, bạn có thể để trống. Hệ thống sẽ tự động kích hoạt cơ chế băm vector offline dự phòng).*

### Bước 3: Tải bộ dữ liệu mẫu về
Để tự động tải 100 câu hỏi y tế tiếng Việt thực tế từ Hugging Face về máy, chạy lệnh:
```bash
python src/download_dataset.py
```

### Bước 4: Khởi chạy Chatbot
```bash
python main.py
```

---

## 📝 3. Xem báo cáo chi tiết dự án
Mọi thông tin chi tiết về sơ đồ kiến trúc, chi tiết thuật toán TF-IDF Cosine, cơ chế hoạt động của các module, vòng đời dữ liệu (CREATE - USED - UPDATE - DELETE) và nhật ký chạy thử (logs) được lưu tại tệp báo cáo chính thức:
👉 **[Xem báo cáo dự án chi tiết tại đây (reports/report_0.md)](file:///D:/rag_project/reports/report_0.md)**
