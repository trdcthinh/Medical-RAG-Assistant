# -*- coding: utf-8 -*-
import unittest
import os
import sys

# Đảm bảo đường dẫn gốc dự án có trong PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.loader import MedicalProtocolLoader
from src.chunking.chunker import MedicalProtocolChunker
from src.embeddings.embedder import MedicalEmbedder

class TestMedicalRAGComponents(unittest.TestCase):
    
    def setUp(self):
        # Thiết lập dữ liệu giả lập cho các bài test
        self.mock_dataset_path = "tests/mock_dataset.json"
        self.loader = MedicalProtocolLoader(self.mock_dataset_path)
        self.chunker = MedicalProtocolChunker(chunk_size=500)
        self.embedder = MedicalEmbedder(api_key="") # Chạy chế độ fallback offline

    def test_clean_text(self):
        """Kiểm thử hàm làm sạch văn bản y tế"""
        raw_text = "Bệnh lý: Sốt xuất huyết!!! Triệu chứng lâm sàng, bao gồm: Sốt cao, nổi mẩn..."
        cleaned = self.loader.clean_text(raw_text)
        # Kỳ vọng: chuyển chữ thường, loại bỏ ký tự đặc biệt, không thừa khoảng trắng
        self.assertNotIn("!!!", cleaned)
        self.assertNotIn("...", cleaned)
        self.assertEqual(cleaned, "bệnh lý sốt xuất huyết triệu chứng lâm sàng bao gồm sốt cao nổi mẩn")

    def test_create_chunks(self):
        """Kiểm thử cơ chế phân đoạn phác đồ y khoa (chunking)"""
        mock_protocols = [
            {
                "id": 1,
                "disease": "Sốt xuất huyết Dengue",
                "symptoms": "Sốt cao đột ngột",
                "treatment": "Uống nhiều nước, bù điện giải",
                "warning_signs": "Chảy máu chân răng, đau bụng cấp"
            }
        ]
        chunks = self.chunker.create_chunks(mock_protocols)
        
        # Có 3 trường thông tin lập bản đồ -> Kỳ vọng có đúng 3 chunks được tạo ra
        self.assertEqual(len(chunks), 3)
        
        # Kiểm tra metadata và nội dung của chunk đầu tiên
        first_chunk = chunks[0]
        self.assertEqual(first_chunk["disease_id"], 1)
        self.assertEqual(first_chunk["disease_name"], "Sốt xuất huyết Dengue")
        self.assertIn("symptoms", first_chunk["chunk_id"])
        self.assertTrue(first_chunk["text"].startswith("Bệnh lý: Sốt xuất huyết Dengue."))

    def test_offline_embeddings(self):
        """Kiểm thử thuật toán băm offline và chuẩn hóa L2 của embedder"""
        text = "Sốt xuất huyết Dengue triệu chứng sốt cao"
        vector = self.embedder.embed_text(text)
        
        # Độ dài vector phải đúng 768 chiều
        self.assertEqual(len(vector), 768)
        
        # Độ dài L2 Norm của vector sau chuẩn hóa phải xấp xỉ 1.0 (hoặc 0 nếu rỗng)
        magnitude = sum(x**2 for x in vector)**0.5
        self.assertAlmostEqual(magnitude, 1.0, places=5)

if __name__ == "__main__":
    unittest.main()
