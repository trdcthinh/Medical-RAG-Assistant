# -*- coding: utf-8 -*-

class MedicalProtocolChunker:
    """
    Module Chunking: Chuyển đổi các bản ghi thô thành các phân đoạn văn bản nhỏ (Chunks)
    kèm theo Metadata chi tiết để tối ưu hóa khả năng truy xuất RAG.
    """
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size

    def create_chunks(self, raw_protocols):
        """
        Nhận danh sách phác đồ từ Module Ingestion và chuyển hóa thành các 
        chunk ngữ cảnh chuyên sâu theo từng chủ đề lâm sàng.
        """
        chunks = []
        
        for protocol in raw_protocols:
            disease_name = protocol.get("disease", "Bệnh chưa rõ")
            disease_id = protocol.get("id", 0)
            
            # Khởi tạo các đoạn ngữ cảnh theo từng trường chức năng
            field_mappings = [
                ("symptoms", "Triệu chứng lâm sàng"),
                ("treatment", "Phác đồ điều trị lâm sàng"),
                ("warning_signs", "Dấu hiệu cảnh báo lâm sàng nguy hiểm")
            ]
            
            for field_key, field_title in field_mappings:
                content = protocol.get(field_key, "")
                if content:
                    # Tạo cấu trúc văn bản RAG tự chứa (Self-contained context string)
                    formatted_text = (
                        f"Bệnh lý: {disease_name}.\n"
                        f"Nội dung: {field_title} là: {content}"
                    )
                    
                    # Lưu kèm theo siêu dữ liệu (Metadata)
                    chunk_metadata = {
                        "chunk_id": f"{disease_id}_{field_key}",
                        "disease_id": disease_id,
                        "disease_name": disease_name,
                        "info_type": field_key,
                        "text": formatted_text
                    }
                    chunks.append(chunk_metadata)
                    
        return chunks