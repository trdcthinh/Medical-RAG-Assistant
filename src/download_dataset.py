# -*- coding: utf-8 -*-
"""
Script tải tự động bộ dữ liệu câu hỏi - câu trả lời y tế tiếng Việt (Medical QA)
từ Hugging Face về làm Cơ sở dữ liệu cho Chatbot RAG.
Được cấu hình để lưu dữ liệu vào D:\rag_project\data\dataset.json
"""

import os
import json
import urllib.request
import urllib.parse
import re
import sys

# Cấu hình UTF-8 cho console để hiển thị tiếng Việt trên Windows tránh lỗi UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def clean_vietnamese_text(text):
    """Làm sạch và chuẩn hóa văn bản tiếng Việt cơ bản"""
    if not text:
        return ""
    text = text.replace("\r", "")
    # Thay thế nhiều dấu xuống dòng liên tiếp bằng 1 dấu xuống dòng
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def tokenize(text):
    """Tiền xử lý và tách từ tiếng Việt cơ bản để tạo keywords"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [word for word in text.split() if len(word) > 1]

def download_vietnamese_medical_qa(limit=100, output_file="dataset.json"):
    print(f"[*] Bắt đầu tải {limit} câu hỏi y tế tiếng Việt từ Hugging Face...")
    
    # URL API Hugging Face Dataset Viewer
    base_url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "hungnm/vietnamese-medical-qa",
        "config": "default",
        "split": "train",
        "offset": 0,
        "limit": limit
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        rows = data.get("rows", [])
        if not rows:
            print("[!] Không tìm thấy dữ liệu nào từ API.")
            return False
            
        formatted_docs = []
        for idx, row_data in enumerate(rows):
            row = row_data.get("row", {})
            raw_q = row.get("question", "")
            raw_a = row.get("answer", "")
            
            clean_q = clean_vietnamese_text(raw_q)
            clean_a = clean_vietnamese_text(raw_a)
            
            if not clean_q or not clean_a:
                continue
                
            doc = {
                "id": idx + 1,
                "question": clean_q,
                "answer": clean_a,
                "category": "Y học thường thức",
                "keywords": tokenize(clean_q)[:15] # Lấy tối đa 15 từ khóa đại diện
            }
            formatted_docs.append(doc)
            
        # Đảm bảo thư mục cha tồn tại
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Lưu file dataset.json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_docs, f, ensure_ascii=False, indent=2)
            
        print(f"[+] Đã tải và lưu thành công {len(formatted_docs)} câu hỏi y tế vào file '{output_file}'!")
        return True
        
    except Exception as e:
        print(f"[!] Lỗi trong quá trình tải dữ liệu: {e}")
        print("[*] Gợi ý: Kiểm tra lại kết nối mạng của bạn.")
        return False

if __name__ == "__main__":
    # Xác định đường dẫn tương đối từ vị trí file chạy
    # file chạy nằm ở D:\rag_project\src\download_dataset.py
    # Thư mục gốc dự án là D:\rag_project
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    output_path = os.path.join(project_root, "data", "dataset.json")
    
    # Mặc định tải 150 câu hỏi
    download_vietnamese_medical_qa(limit=150, output_file=output_path)
