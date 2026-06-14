# -*- coding: utf-8 -*-
import chromadb
import os

class MedicalVectorStore:
    """
    Module VectorDB: Lưu trữ và quản lý cơ sở dữ liệu vector sử dụng ChromaDB.
    """
    def __init__(self, db_dir, embedder):
        self.db_dir = db_dir
        self.embedder = embedder
        os.makedirs(db_dir, exist_ok=True)
        
        # Khởi tạo client lưu trữ dữ liệu bền vững xuống đĩa cứng (Persistent Client)
        self.client = chromadb.PersistentClient(path=db_dir)
        # Lấy hoặc tạo Collection mới
        self.collection = self.client.get_or_create_collection(
            name="medical_protocols_collection",
            metadata={"hnsw:space": "cosine"} # Sử dụng độ đo khoảng cách Cosine
        )

    def add_chunks(self, chunks):
        """Nạp các chunk và vector tương ứng vào ChromaDB"""
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            # 1. Gọi module Embedder để biến đổi văn bản thành Vector số học
            vector = self.embedder.embed_text(chunk["text"])
            
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            embeddings.append(vector)
            metadatas.append({
                "disease_id": chunk["disease_id"],
                "disease_name": chunk["disease_name"],
                "info_type": chunk["info_type"]
            })

        # 2. Lưu vào CSDL ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search_similar(self, query_text, top_k=2):
        """Tìm kiếm các đoạn phác đồ tương đồng nhất với câu hỏi"""
        # Sinh vector cho câu hỏi truy vấn của người dùng
        query_vector = self.embedder.embed_text(query_text)
        
        # Truy vấn CSDL
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        # Định dạng lại kết quả trả về
        matched_chunks = []
        if results and results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                # ChromaDB trả về khoảng cách Cosine (distance). Độ tương đồng = 1 - khoảng cách
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                
                matched_chunks.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "similarity": similarity
                })
        return matched_chunks