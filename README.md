# Hệ thống Trợ lý Hỏi đáp & Kiểm duyệt Phác đồ Điều trị Y khoa (Medical RAG Assistant)

Dự án này là hệ thống chatbot RAG (Retrieval-Augmented Generation) kết hợp với Hệ chuyên gia (Expert System) để hỗ trợ hỏi đáp phác đồ điều trị y tế lâm sàng, tích hợp cơ chế duyệt dữ liệu an toàn từ bác sĩ trưởng khoa (Human-in-the-Loop) giúp khép kín vòng lặp học hỏi dữ liệu (Data Flywheel).

---

## 🏗️ 1. Cấu trúc Dự án (RAG Project Structure)
Dự án được xây dựng theo đúng tiêu chuẩn modular do phòng lab yêu cầu:

```text
rag_project/
├── .env                       # Lưu trữ khóa bí mật API (không đẩy lên Github)
├── .gitignore                 # Các tệp Git cần bỏ qua
├── config.yaml                # Các tham số mô hình, chunking, thresholds
├── requirements.txt           # Danh sách thư viện Python cần cài đặt
├── main.py                    # Điểm khởi chạy ứng dụng chính
├── src/
│   ├── ingestion/             # Load dữ liệu phác đồ y khoa (PDF, JSON, TXT)
│   ├── chunking/              # Cắt nhỏ văn bản thành các đoạn văn ngắn
│   ├── embeddings/            # Chuyển đổi văn bản thành Vector nhúng (Gemini/Local)
│   ├── vectordb/              # Lưu trữ và truy vấn Vector (ChromaDB)
│   ├── retrieval/             # Thực hiện tìm kiếm và xếp hạng kết quả tương đồng
│   ├── prompts/               # Định nghĩa các tiêu bản câu lệnh (Prompt Templates)
│   ├── llm/                   # Kết nối và gửi yêu cầu tới Gemini API
│   └── utils/                 # Các hàm bổ trợ ghi log, định dạng
├── tests/
│   └── test_app.py            # Viết unit test tự động hóa cho hệ thống
└── logs/
    └── app.log                # Lưu log chạy hệ thống để giám sát lỗi (DataOps)
```

---

## 🔬 2. Cơ chế Hoạt động Hợp nhất (RAG + Expert System + Flywheel)
1. **RAG**: Tìm kiếm phác đồ điều trị phù hợp nhất từ cơ sở dữ liệu và chuyển cho Gemini LLM sinh câu trả lời chính xác, tránh hiện tượng ảo tưởng lâm sàng.
2. **Hệ chuyên gia (Expert System Router)**: Đóng vai trò lớp kiểm duyệt an toàn (Safety Layer). Các câu hỏi liên quan đến tình huống khẩn cấp (ví dụ: *"đau ngực dữ dội"*) sẽ được nhận diện bằng các luật cứng để đưa ra cảnh báo đi cấp cứu ngay lập tức mà không đi qua LLM. Các câu hỏi có độ tương đồng thấp sẽ kích hoạt luồng **Human-in-the-Loop**.
3. **Data Flywheel (Bánh đà dữ liệu)**: Khi bác sĩ bổ sung tri thức lâm sàng mới thông qua cơ chế duyệt dữ liệu, thông tin này được lưu lại vào cơ sở tri thức để chatbot tự động trả lời cho các lần hỏi tiếp theo.
