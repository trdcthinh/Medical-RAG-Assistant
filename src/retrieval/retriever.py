# -*- coding: utf-8 -*-

class MedicalRetriever:
    """
    Module Retrieval: Thực hiện truy xuất ngữ cảnh và xếp hạng kết quả tương đồng.
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve_context(self, query, top_k=2):
        """
        Truy xuất các chunk phác đồ liên quan.
        Trả về: Ngữ cảnh gộp (context_str), điểm tin cậy cao nhất (max_similarity), và bản ghi khớp tốt nhất.
        """
        matched_chunks = self.vector_store.search_similar(query, top_k=top_k)
        
        if not matched_chunks:
            return "", 0.0, None
            
        # Bản ghi khớp tốt nhất nằm ở vị trí đầu tiên
        best_match = matched_chunks[0]
        max_similarity = best_match["similarity"]
        
        # Gộp nội dung các chunk làm ngữ cảnh RAG
        context_texts = []
        for chunk in matched_chunks:
            context_texts.append(chunk["text"])
            
        context_str = "\n---\n".join(context_texts)
        return context_str, max_similarity, best_match