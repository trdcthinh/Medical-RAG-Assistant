# -*- coding: utf-8 -*-
import json
import urllib.request
import os
import shutil

def download_and_enrich(num_records=100):
    """
    Tải tập dữ liệu y tế công cộng (WebMD QA) từ Github và đồng bộ vào dự án RAG.
    """
    url = "https://raw.githubusercontent.com/LasseRegin/medical-question-answer-data/master/webmdQAs.json"
    dataset_path = "data/medical_protocols.json"
    db_path = "data/chroma_db"
    
    print("1. Đang kết nối và tải dataset WebMD Q&A y khoa công cộng...")
    try:
        # Tải tệp JSON từ Github
        with urllib.request.urlopen(url) as response:
            raw_data = json.loads(response.read().decode('utf-8'))
        
        print(f"-> Tải thành công! Tìm thấy {len(raw_data)} bản ghi hỏi đáp y khoa.")
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return

    # 2. Đọc tệp CSDL hiện tại
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            protocols = json.load(f)
    else:
        protocols = []
        
    start_id = max([p["id"] for p in protocols]) + 1 if protocols else 1
    
    print(f"2. Đang tiền xử lý và chuyển đổi dữ liệu ({num_records} bản ghi đầu)...")
    added_count = 0
    
    # Duyệt qua các bản ghi từ WebMD và chuyển đổi định dạng tương thích với phác đồ
    for entry in raw_data[:num_records]:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        
        # Bỏ qua nếu dữ liệu trống
        if not question or not answer:
            continue
            
        # Trích xuất "tên bệnh" sơ bộ từ câu hỏi hoặc gán nhãn chủ đề
        disease_label = "Y học Tổng quát (WebMD QA)"
        # Thử đoán tên bệnh từ các từ khóa trong câu hỏi
        words = question.split()
        if len(words) > 3:
            disease_label = f"Chẩn đoán: {' '.join(words[:4])}..."
            
        new_protocol = {
            "id": start_id,
            "disease": disease_label,
            "symptoms": question,
            "treatment": answer,
            "warning_signs": "Liên hệ ngay với bác sĩ nếu các triệu chứng trở nên nghiêm trọng hoặc không thuyên giảm."
        }
        
        protocols.append(new_protocol)
        start_id += 1
        added_count += 1
        
    # 3. Ghi dữ liệu phong phú mới vào dataset.json
    print(f"3. Đang lưu {added_count} bản ghi mới vào {dataset_path}...")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(protocols, f, ensure_ascii=False, indent=2)
        
    # 4. Xóa ChromaDB cũ để ép hệ thống re-index lại toàn bộ từ đầu
    if os.path.exists(db_path):
        print("4. Đang dọn dẹp ChromaDB cũ để thiết lập cơ sở dữ liệu vector mới...")
        shutil.rmtree(db_path)
        
    print("\n" + "="*50)
    print(f"HOÀN THÀNH LÀM GIÀU DATA!")
    print(f"Tổng số phác đồ và câu hỏi y tế hiện tại trong hệ thống: {len(protocols)} bản ghi.")
    print("Vui lòng khởi chạy lại lệnh: python main.py để ChromaDB tự động lập chỉ mục.")
    print("="*50)

if __name__ == "__main__":
    download_and_enrich(100) # Mặc định lấy thêm 100 bản ghi
