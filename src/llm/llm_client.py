# -*- coding: utf-8 -*-
import os

class GeminiLLMClient:
    """
    Module LLM: Giao tiếp với Google Gemini API để thực hiện sinh câu trả lời (Generation).
    """
    def __init__(self, model_name="gemini-2.5-flash", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self.model = None
        self.is_active = False
        self.sdk_type = None
        self.initialize_client()

    def initialize_client(self):
        """Khởi tạo Client kết nối Google Gemini API"""
        if not self.api_key:
            self.is_active = False
            return
        
        try:
            # Kiểm tra và nạp SDK thế hệ cũ (google-generativeai)
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.sdk_type = "legacy"
            self.is_active = True
        except ImportError:
            try:
                # Kiểm tra và nạp SDK thế hệ mới (google-genai)
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.sdk_type = "modern"
                self.is_active = True
            except ImportError:
                self.is_active = False

    def generate_answer(self, prompt, system_instruction=None):
        """Gửi prompt đã được làm giàu ngữ cảnh sang Gemini để sinh văn bản"""
        if not self.is_active:
            return "Hệ thống đang chạy Offline (Không có API Key). Vui lòng nạp GEMINI_API_KEY để kích hoạt sinh câu trả lời."
            
        try:
            if self.sdk_type == "legacy":
                # Ở SDK cũ, gộp system instruction trực tiếp vào trước prompt
                full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = self.model.generate_content(full_prompt)
                return response.text.strip()
            else:
                # Ở SDK mới, truyền system instruction qua cấu hình config độc lập
                config = {}
                if system_instruction:
                    config["system_instruction"] = system_instruction
                    
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text.strip()
        except Exception as e:
            return f"Lỗi gọi Gemini API sinh câu trả lời: {e}"