# -*- coding: utf-8 -*-
import os

class MedicalEmbedder:
    """
    Module Embeddings: Chuyển đổi văn bản y tế thành các vector số học (Vector Embeddings).
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self.is_gemini_active = False
        self.initialize_gemini()

    def initialize_gemini(self):
        """Khởi tạo kết nối Gemini để sinh vector nhúng"""
        if self.api_key:
            try:
                # Hỗ trợ SDK thế hệ cũ
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.is_gemini_active = True
                self.sdk_type = "legacy"
            except ImportError:
                try:
                    # Hỗ trợ SDK thế hệ mới (google-genai)
                    from google import genai
                    self.client = genai.Client(api_key=self.api_key)
                    self.is_gemini_active = True
                    self.sdk_type = "modern"
                except ImportError:
                    self.is_gemini_active = False

    def embed_text(self, text):
        """Sinh vector 768 chiều cho đoạn văn bản"""
        if self.is_gemini_active:
            try:
                if self.sdk_type == "legacy":
                    import google.generativeai as genai
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document"
                    )
                    return result['embedding']
                else:
                    result = self.client.models.embed_content(
                        model="text-embedding-004",
                        contents=text
                    )
                    return result.embeddings[0].values
            except Exception as e:
                # Ghi nhận lỗi và tự động chuyển sang chế độ fallback ngoại tuyến dưới đây
                pass
        
        # ⚠️ CHẾ ĐỘ FALLBACK OFFLINE (Chạy khi không có internet/API Key)
        # Sử dụng thuật toán băm từ vựng (Word-Level Hashing với djb2) kết hợp loại bỏ từ dừng (Stopwords)
        # để tránh việc các từ chung chung (bệnh, điều trị, bác sĩ...) làm nhiễu điểm số tương đồng.
        import re
        words = re.findall(r'\w+', text.lower())
        vector = [0.0] * 768
        
        # Danh sách từ dừng/từ cấu trúc cần loại bỏ
        STOPWORDS = {
            "bệnh", "lý", "nội", "dung", "là", "và", "của", "để", "trong", "có", 
            "thế", "nào", "chữa", "triệu", "chứng", "phác", "đồ", "điều", "trị", 
            "cảnh", "báo", "dấu", "hiệu", "lâm", "sàng", "nguy", "hiểm", "bác", 
            "sĩ", "cho", "hỏi", "tôi", "em", "cháu", "bị", "những", "các", "như"
        }
        
        def djb2_hash(s):
            h = 5381
            for c in s:
                h = ((h << 5) + h) + ord(c)
            return h & 0xFFFFFFFF

        for word in words:
            if word in STOPWORDS:
                continue
            idx = djb2_hash(word) % 768
            vector[idx] += 1.0

        # Chuẩn hóa vector về độ dài đơn vị (L2 Normalization)
        magnitude = sum(x**2 for x in vector)**0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        return vector