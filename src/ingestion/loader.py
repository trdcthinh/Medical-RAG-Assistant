# -*- coding: utf-8 -*-
import json
import os
import re

class MedicalProtocolLoader:
    """
    Module Ingestion: Tải dữ liệu phác đồ y tế từ file JSON và làm sạch dữ liệu.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.protocols = []

    def load_protocols(self):
        """Đọc danh sách phác đồ y khoa từ tệp tin"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu y tế tại: {self.filepath}")
            
        with open(self.filepath, "r", encoding="utf-8") as f:
            self.protocols = json.load(f)
        return self.protocols

    def clean_text(self, text):
        """
        Tiền xử lý văn bản lâm sàng:
        1. Chuyển thành chữ thường.
        2. Loại bỏ ký tự đặc biệt, giữ lại chữ cái, số tiếng Việt và khoảng trắng.
        """
        if not text:
            return ""
        text = text.lower()
        # Loại bỏ các ký tự đặc biệt như ?, !, ., ,, -, dấu ngoặc...
        text = re.sub(r'[^\w\s\s]', ' ', text)
        # Loại bỏ các khoảng trắng thừa
        text = " ".join(text.split())
        return text