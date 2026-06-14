# -*- coding: utf-8 -*-
import os
import json
import colorama
from colorama import Fore, Style

# Import các module tự xây dựng từ thư mục src/
from src.ingestion.loader import MedicalProtocolLoader
from src.chunking.chunker import MedicalProtocolChunker
from src.embeddings.embedder import MedicalEmbedder
from src.vectordb.vector_store import MedicalVectorStore
from src.retrieval.retriever import MedicalRetriever
from src.prompts.prompt_templates import SYSTEM_INSTRUCTION, RAG_PROMPT_TEMPLATE
from src.llm.llm_client import GeminiLLMClient
from src.utils.helpers import load_config, load_env, setup_logging

# Khởi tạo thư viện màu sắc terminal
colorama.init()

class MedicalRAGApp:
    """
    Orchestrator: Bộ điều phối trung tâm của toàn bộ hệ thống RAG + Expert System + Flywheel.
    """
    def __init__(self):
        # 1. Nạp cấu hình và biến môi trường
        load_env()
        self.config = load_config()
        self.logger = setup_logging(self.config["paths"]["logs"])
        self.logger.info("Khởi động hệ thống Medical RAG Assistant.")
        
        # 2. Khởi tạo các module thành phần
        self.loader = MedicalProtocolLoader(self.config["paths"]["dataset"])
        self.chunker = MedicalProtocolChunker(self.config["chunking"]["size"])
        self.embedder = MedicalEmbedder()
        self.vector_store = MedicalVectorStore(self.config["paths"]["db_dir"], self.embedder)
        self.retriever = MedicalRetriever(self.vector_store)
        self.llm = GeminiLLMClient(model_name=self.config["models"]["llm"])
        
        # 3. Đọc luật của Hệ chuyên gia
        with open(self.config["paths"]["rules"], "r", encoding="utf-8") as f:
            self.rules_config = json.load(f)
            
        self.chat_history = []
        
        # 4. Ingestion & Indexing ban đầu nếu CSDL Vector trống
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        """Kiểm tra và tự động nạp dữ liệu phác đồ vào ChromaDB khi khởi chạy lần đầu"""
        try:
            # Kiểm tra xem collection đã có dữ liệu chưa
            count = self.vector_store.collection.count()
            if count == 0:
                print(f"{Fore.YELLOW}Đang lập chỉ mục CSDL Vector lần đầu...{Style.RESET_ALL}")
                raw_data = self.loader.load_protocols()
                chunks = self.chunker.create_chunks(raw_data)
                self.vector_store.add_chunks(chunks)
                self.logger.info(f"Đã nạp thành công {len(chunks)} chunks vào ChromaDB.")
                print(f"{Fore.GREEN}Lập chỉ mục thành công! Tổng số chunks: {len(chunks)}{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"Lỗi khởi tạo CSDL: {e}")

    def add_new_protocol_dynamically(self, disease, symptoms, treatment, warning_signs):
        """Khép kín Data Flywheel: Lưu phác đồ mới vào file JSON và nạp trực tiếp vào ChromaDB"""
        # Đọc dữ liệu JSON hiện tại
        raw_protocols = self.loader.load_protocols()
        new_id = max([p["id"] for p in raw_protocols]) + 1 if raw_protocols else 1
        
        new_protocol = {
            "id": new_id,
            "disease": disease,
            "symptoms": symptoms,
            "treatment": treatment,
            "warning_signs": warning_signs
        }
        
        # 1. Ghi đè file JSON vật lý
        raw_protocols.append(new_protocol)
        with open(self.config["paths"]["dataset"], "w", encoding="utf-8") as f:
            json.dump(raw_protocols, f, ensure_ascii=False, indent=2)
            
        # 2. Cắt chunk và nạp động vào ChromaDB
        new_chunks = self.chunker.create_chunks([new_protocol])
        self.vector_store.add_chunks(new_chunks)
        self.logger.info(f"Flywheel cập nhật: Thêm bệnh lý '{disease}', ID: {new_id}")
        return new_id

    def process_chat(self, user_query):
        """Động cơ suy diễn định tuyến câu hỏi và xử lý phản hồi"""
        self.logger.info(f"Người dùng hỏi: {user_query}")
        query_lower = user_query.lower()
        
        # --- BƯỚC 1: LUẬT KIỂM DUYỆT KHẨN CẤP (Expert System Rule 1) ---
        emergency_kws = self.rules_config["emergency_keywords"]
        if any(kw in query_lower for kw in emergency_kws):
            explanation = "Kích hoạt LUẬT KHẨN CẤP: Dấu hiệu đe dọa tính mạng lâm sàng. Chuyển thẳng cảnh báo khẩn cấp."
            self.logger.warning("Cảnh báo khẩn cấp được kích hoạt.")
            answer = (
                f"{Fore.RED}{Style.BRIGHT}[CẢNH BÁO Y TẾ KHẨN CẤP]{Style.RESET_ALL}\n"
                f"Phát hiện triệu chứng nguy kịch đe dọa tính mạng. Vui lòng lập tức đưa bệnh nhân "
                f"đến cơ sở y tế gần nhất hoặc gọi số điện thoại cấp cứu 115 ngay lập tức!"
            )
            return answer, "RULE_EMERGENCY_ALARM", explanation

        # --- BƯỚC 2: TRUY XUẤT CSDL VECTOR (Retrieval) ---
        context_str, similarity, best_match = self.retriever.retrieve_context(user_query)
        self.logger.info(f"Độ tương đồng Cosine: {similarity:.4f}")

        # --- BƯỚC 3: ĐỊNH TUYẾN DỰA TRÊN ĐỘ TIN CẬY (Inference Rules) ---
        thresholds = self.rules_config["confidence_thresholds"]
        
        # Phác đồ khớp cao -> RAG trực tiếp kết hợp Gemini sinh câu trả lời
        if similarity >= thresholds["high"]:
            explanation = f"LUẬT ĐỘ TIN CẬY CAO: Điểm số ({similarity:.2f}) >= Ngưỡng cao ({thresholds['high']:.2f}). Đủ an toàn để trả lời bằng RAG."
            # Tạo prompt làm giàu ngữ cảnh
            history_str = ""
            for turn in self.chat_history[-3:]:
                history_str += f"User: {turn['user']}\nBot: {turn['bot']}\n"
                
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context_str,
                history=history_str,
                query=user_query
            )
            answer = self.llm.generate_answer(prompt, system_instruction=SYSTEM_INSTRUCTION)
            route = "RAG_DIRECT"
            
        # Phác đồ khớp vừa -> Hỏi xác nhận thông tin bệnh
        elif similarity >= thresholds["medium"]:
            disease_name = best_match["metadata"]["disease_name"]
            explanation = f"LUẬT ĐỘ TIN CẬY VỪA PHẢI: Điểm số ({similarity:.2f}) nằm trong khoảng nghi vấn. Cần xác nhận thông tin bệnh lý."
            answer = f"Hệ thống: Có phải bạn đang muốn tìm hiểu về phác đồ điều trị của bệnh lý: \033[1m{disease_name}\033[0m không?"
            route = "RAG_CONFIRMATION"
            
        # Không tìm thấy phác đồ -> Kích hoạt Human-in-the-Loop
        else:
            explanation = f"LUẬT ĐỘ TIN CẬY THẤP: Điểm số ({similarity:.2f}) < Ngưỡng an toàn ({thresholds['medium']:.2f}). Kích hoạt Human-in-the-loop."
            answer = self.handle_human_in_the_loop(user_query)
            route = "HUMAN_IN_THE_LOOP"
            
        self.chat_history.append({"user": user_query, "bot": answer})
        return answer, route, explanation

    def handle_human_in_the_loop(self, query):
        """Giao diện con người can thiệp (Bác sĩ trưởng khoa phê duyệt tri thức mới)"""
        print(f"\n{Fore.RED}[HUMAN-IN-THE-LOOP] Hệ thống không tìm thấy phác đồ chính thức cho câu hỏi này.{Style.RESET_ALL}")
        choice = input("Bạn có muốn đóng vai Bác sĩ Trưởng khoa cập nhật phác đồ điều trị này không? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            disease = input("Nhập tên bệnh lý chính thức: ").strip()
            symptoms = input("Mô tả các triệu chứng chính: ").strip()
            treatment = input("Mô tả phác đồ điều trị chuẩn: ").strip()
            warning_signs = input("Mô tả các dấu hiệu nguy hiểm cần cảnh báo: ").strip()
            
            if disease and treatment:
                # Gọi hàm khép bánh đà dữ liệu
                new_id = self.add_new_protocol_dynamically(disease, symptoms, treatment, warning_signs)
                return f"Hệ thống đã lưu nhận phác đồ bệnh '{disease}' (ID: {new_id}). Bánh đà dữ liệu đã tự cập nhật!"
            else:
                return "Cập nhật thất bại do điền thiếu thông tin bệnh hoặc phác đồ."
        return "Xin lỗi, câu hỏi này nằm ngoài phạm vi CSDL phác đồ hiện tại của chúng tôi."

def main():
    app = MedicalRAGApp()
    
    print("=" * 80)
    print(f" {Fore.GREEN}{Style.BRIGHT}TRỢ LÝ Y KHOA LÂM SÀNG LÂM SÀNG - KẾT HỢP RAG & HỆ CHUYÊN GIA{Style.RESET_ALL}")
    print("=" * 80)
    print(f"Trạng thái Gemini API: {Fore.GREEN + 'ONLINE' if app.llm.is_active else Fore.RED + 'OFFLINE (Chạy dữ liệu tĩnh)'}{Style.RESET_ALL}")
    print("Mẹo: Nhập 'exit' để thoát hội thoại.")
    print("=" * 80)

    while True:
        try:
            query = input(f"\n{Fore.CYAN}{Style.BRIGHT}Bác sĩ: {Style.RESET_ALL}").strip()
            if not query:
                continue
            if query.lower() == 'exit':
                print(f"\n{Fore.GREEN}Chào tạm biệt! Chúc bạn bảo cáo lab thành công!{Style.RESET_ALL}")
                break
                
            answer, route, explanation = app.process_chat(query)
            
            # Hiển thị kết xuất của Hệ chuyên gia (Explanation Facility)
            print(f"\n{Fore.YELLOW}[EXPLANATION FACILITY - ĐƯỜNG SUY DIỄN CỦA HỆ CHUYÊN GIA]{Style.RESET_ALL}")
            print(f"  - Cơ chế kích hoạt: {route}")
            print(f"  - Lý do: {explanation}")
            print("-" * 50)
            
            print(f"\n{Fore.GREEN}{Style.BRIGHT}Trợ lý ảo:{Style.RESET_ALL} {answer}")
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break

if __name__ == "__main__":
    main()