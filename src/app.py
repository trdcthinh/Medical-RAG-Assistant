# -*- coding: utf-8 -*-
"""
HỆ THỐNG CHATBOT THỰC HÀNH CÓ LỊCH SỬ CHAT (HISTORY)
Tích hợp: RAG (Retrieval-Augmented Generation) + Expert System (Hệ chuyên gia) + Human-in-the-Loop & Data Flywheel
Địa chỉ lưu trữ trên ổ D: D:\rag_project
"""

import os
import json
import math
import re
import sys
from datetime import datetime

# Import Gemini SDK (hỗ trợ cả thư viện cũ và mới)
HAS_GEMINI_SDK = False
try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
    SDK_VERSION = "legacy"
except ImportError:
    try:
        from google import genai
        HAS_GEMINI_SDK = True
        SDK_VERSION = "modern"
    except ImportError:
        HAS_GEMINI_SDK = False
        SDK_VERSION = None

# ANSI colors for premium terminal UI
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

class TextRetriever:
    """
    Tầng Retrieval: Tìm kiếm văn bản liên quan sử dụng thuật toán TF-IDF Cosine Similarity từ đầu (from scratch)
    để đảm bảo chạy offline và tin cậy cao, đồng thời hỗ trợ gọi API nhúng của Gemini nếu trực tuyến.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.documents = []
        self.load_dataset()
        
    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        else:
            self.documents = []

    def save_dataset(self):
        # Đảm bảo thư mục cha tồn tại trước khi lưu
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def add_new_document(self, question, answer, category="Expert Input"):
        # Lấy ID mới tiếp theo
        new_id = max([doc["id"] for doc in self.documents]) + 1 if self.documents else 1
        new_doc = {
            "id": new_id,
            "question": question,
            "answer": answer,
            "category": category,
            "keywords": self._tokenize(question)
        }
        self.documents.append(new_doc)
        self.save_dataset()
        return new_doc

    def _tokenize(self, text):
        """Tiền xử lý và tách từ tiếng Việt cơ bản"""
        text = text.lower()
        # Loại bỏ các ký tự đặc biệt
        text = re.sub(r'[^\w\s\s]', ' ', text)
        return [word for word in text.split() if word]

    def _calculate_tf(self, tokens):
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        length = len(tokens)
        if length == 0:
            return {}
        return {k: v / length for k, v in tf.items()}

    def _calculate_idf(self):
        idf = {}
        total_docs = len(self.documents)
        if total_docs == 0:
            return idf
        
        # Đếm số tài liệu chứa mỗi từ
        for doc in self.documents:
            tokens = set(self._tokenize(doc["question"] + " " + doc["answer"]))
            for token in tokens:
                idf[token] = idf.get(token, 0) + 1
                
        # Tính công thức IDF
        for token, count in idf.items():
            idf[token] = math.log(1 + (total_docs / count))
        return idf

    def get_relevant_chunk(self, query):
        """
        Thực hiện tìm kiếm ngữ nghĩa bằng cách so khớp độ tương đồng Cosine
        trên không gian vector TF-IDF của câu hỏi trong CSDL.
        """
        if not self.documents:
            return None, 0.0

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return None, 0.0

        # Tính toán IDF toàn cục
        idf = self._calculate_idf()
        
        # Vector hóa câu hỏi truy vấn
        query_tf = self._calculate_tf(query_tokens)
        query_vector = {}
        for token, tf_val in query_tf.items():
            query_vector[token] = tf_val * idf.get(token, 0.0)

        best_doc = None
        max_similarity = 0.0

        # Duyệt qua từng tài liệu trong CSDL để tính Cosine Similarity
        for doc in self.documents:
            doc_tokens = self._tokenize(doc["question"])
            doc_tf = self._calculate_tf(doc_tokens)
            
            doc_vector = {}
            for token, tf_val in doc_tf.items():
                doc_vector[token] = tf_val * idf.get(token, 0.0)

            # Tính tích vô hướng (Dot Product) và độ dài vector (Magnitude)
            dot_product = 0.0
            query_magnitude = 0.0
            doc_magnitude = 0.0

            # Tập từ vựng chung
            all_tokens = set(list(query_vector.keys()) + list(doc_vector.keys()))
            for token in all_tokens:
                q_val = query_vector.get(token, 0.0)
                d_val = doc_vector.get(token, 0.0)
                
                dot_product += q_val * d_val
                query_magnitude += q_val ** 2
                doc_magnitude += d_val ** 2

            query_magnitude = math.sqrt(query_magnitude)
            doc_magnitude = math.sqrt(doc_magnitude)

            similarity = 0.0
            if query_magnitude > 0 and doc_magnitude > 0:
                similarity = dot_product / (query_magnitude * doc_magnitude)

            # Tăng trọng số nếu có nhiều từ khóa quan trọng khớp trực tiếp
            exact_matches = set(query_tokens).intersection(set(doc_tokens))
            if exact_matches:
                similarity += len(exact_matches) * 0.05
                similarity = min(similarity, 1.0) # Giới hạn max là 1.0

            if similarity > max_similarity:
                max_similarity = similarity
                best_doc = doc

        return best_doc, max_similarity


class ExpertSystemRouter:
    """
    Tầng Expert System: Sử dụng các luật logic để đánh giá độ tin cậy và
    định tuyến (routing) truy vấn sang luồng xử lý phù hợp.
    """
    def __init__(self, rules_path):
        self.rules_path = rules_path
        self.rules_config = {}
        self.load_rules()

    def load_rules(self):
        if os.path.exists(self.rules_path):
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules_config = json.load(f)
        else:
            self.rules_config = {
                "confidence_thresholds": {"high": 0.80, "medium": 0.50},
                "rules": []
            }

    def evaluate_query(self, query, max_similarity):
        """
        Suy diễn luật của Hệ chuyên gia dựa trên dữ liệu đầu vào.
        Trả về: Quyết định luồng đi (route), cờ cần can thiệp con người (require_human), và chuỗi suy diễn (explanation).
        """
        query_lower = query.lower()
        thresholds = self.rules_config.get("confidence_thresholds", {"high": 0.80, "medium": 0.50})
        high_t = thresholds["high"]
        medium_t = thresholds["medium"]

        # Kiểm tra Luật 4: Định tuyến quản trị (Force Admin)
        admin_keywords = ["quản trị", "cập nhật", "update", "thêm câu hỏi"]
        if any(kw in query_lower for kw in admin_keywords):
            return {
                "route": "ADMIN_INTERFACE",
                "require_human": True,
                "rule_id": "RULE_FORCE_ADMIN",
                "explanation": "Phát hiện từ khóa quản trị hệ thống. Định tuyến người dùng trực tiếp vào Giao diện Cập nhật Tri thức."
            }

        # Suy diễn dựa trên độ tin cậy kết quả so khớp RAG
        if max_similarity >= high_t:
            return {
                "route": "RAG_DIRECT",
                "require_human": False,
                "rule_id": "RULE_HIGH_CONFIDENCE",
                "explanation": f"Độ tương đồng ({max_similarity:.2f}) >= Ngưỡng cao ({high_t:.2f}). Đủ tin cậy để trả lời trực tiếp."
            }
        elif max_similarity >= medium_t:
            return {
                "route": "RAG_CONFIRMATION",
                "require_human": False,
                "rule_id": "RULE_MEDIUM_CONFIDENCE",
                "explanation": f"Độ tương đồng ({max_similarity:.2f}) nằm trong khoảng [{medium_t:.2f}, {high_t:.2f}). Cần người dùng xác nhận thông tin gợi ý."
            }
        else:
            return {
                "route": "HUMAN_IN_THE_LOOP",
                "require_human": True,
                "rule_id": "RULE_LOW_CONFIDENCE",
                "explanation": f"Độ tương đồng ({max_similarity:.2f}) < Ngưỡng thấp ({medium_t:.2f}). Cần kích hoạt Human-in-the-loop để ghi nhận tri thức mới."
            }


class GeminiGenerator:
    """
    Tầng Generation: Kết nối với Gemini API để thực hiện sinh câu trả lời
    dựa trên Prompt được làm giàu ngữ cảnh (Augmented) và quản lý Lịch sử chat (History).
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self.model = None
        self.is_active = False
        
        self.initialize_client()

    def initialize_client(self):
        if not self.api_key:
            return
        
        try:
            if HAS_GEMINI_SDK:
                if SDK_VERSION == "legacy":
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                else:  # modern SDK
                    self.client = genai.Client(api_key=self.api_key)
                self.is_active = True
        except Exception as e:
            print(f"{C_RED}Lỗi khởi tạo Gemini API: {e}{C_RESET}")
            self.is_active = False

    def generate(self, query, context, history):
        """Sinh câu trả lời sử dụng Gemini LLM"""
        if not self.is_active:
            # Fallback nếu không có API Key hoạt động
            return "Hệ thống đang chạy offline (Không có API Key Gemini hợp lệ). Vui lòng cấu hình biến môi trường GEMINI_API_KEY để kích hoạt Generative AI."

        # Xây dựng Prompt tăng cường (RAG Prompt)
        augmented_prompt = f"""Bạn là một trợ lý ảo thông minh của phòng nghiên cứu Lab. 
Dựa vào các TÀI LIỆU THAM KHẢO dưới đây và LỊCH SỬ CHAT, hãy trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và tự nhiên. 
Nếu thông tin không có trong tài liệu tham khảo, hãy trả lời dựa trên những gì bạn biết nhưng hãy nói rõ đây là thông tin bổ sung ngoài CSDL.

TÀI LIỆU THAM KHẢO CỦA PHÒNG LAB:
{context}

LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:
{self._format_history(history)}

CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG:
{query}

CÂU TRẢ LỜI:"""

        try:
            if SDK_VERSION == "legacy":
                response = self.model.generate_content(augmented_prompt)
                return response.text.strip()
            else:  # modern SDK
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=augmented_prompt
                )
                return response.text.strip()
        except Exception as e:
            return f"Lỗi gọi Gemini API sinh câu trả lời: {e}"

    def _format_history(self, history):
        formatted = ""
        for turn in history[-5:]: # Chỉ gửi tối đa 5 lượt chat gần nhất để tiết kiệm token
            formatted += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        return formatted


class ChatbotSystem:
    """
    Lớp điều phối chính (Orchestrator): Kết nối tất cả các tầng,
    quản lý lịch sử hội thoại và tương tác với người dùng.
    """
    def __init__(self):
        # Xác định vị trí thư mục của dự án
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)

        dataset_path = os.path.join(project_root, "data", "dataset.json")
        rules_path = os.path.join(project_root, "config", "expert_rules.json")

        self.retriever = TextRetriever(dataset_path)
        self.router = ExpertSystemRouter(rules_path)
        self.generator = GeminiGenerator()
        self.chat_history = [] # Lưu history hội thoại

    def handle_user_input(self, user_query):
        print(f"\n{C_CYAN}--- QUY TRÌNH XỬ LÝ HỆ THỐNG ---{C_RESET}")
        
        # BƯỚC 1: Retrieval (Truy xuất)
        best_doc, score = self.retriever.get_relevant_chunk(user_query)
        print(f"{C_BLUE}[1. RETRIEVAL]{C_RESET} Thực hiện Vector similarity...")
        if best_doc:
            # Rút ngắn câu hỏi hiển thị nếu quá dài
            display_q = best_doc['question'][:60] + "..." if len(best_doc['question']) > 60 else best_doc['question']
            print(f"  - Câu hỏi khớp nhất trong CSDL: '{display_q}'")
            print(f"  - Độ tương đồng tính toán: {C_BOLD}{score:.4f}{C_RESET}")
        else:
            print("  - CSDL đang trống rỗng!")

        # BƯỚC 2: Expert System Routing (Hệ chuyên gia định tuyến bằng luật logic)
        routing_decision = self.router.evaluate_query(user_query, score)
        print(f"{C_GREEN}[2. EXPERT SYSTEM ROUTING]{C_RESET}")
        print(f"  - Luật kích hoạt: {C_BOLD}{routing_decision['rule_id']}{C_RESET}")
        print(f"  - Luồng quyết định: {C_BOLD}{routing_decision['route']}{C_RESET}")
        print(f"  - Trình giải thích (Explanation): {routing_decision['explanation']}")

        # BƯỚC 3 & 4: Thực thi hành động dựa trên định tuyến
        route = routing_decision["route"]
        final_answer = ""

        if route == "RAG_DIRECT":
            # Chèn context lấy được từ CSDL làm RAG và gửi sang Gemini
            context = f"Câu hỏi: {best_doc['question']}\nTrả lời: {best_doc['answer']}"
            print(f"{C_MAGENTA}[3. RAG - AUGMENTED & GENERATION]{C_RESET} Chạy RAG trực tiếp...")
            
            if self.generator.is_active:
                final_answer = self.generator.generate(user_query, context, self.chat_history)
            else:
                print("  - Chế độ offline: Sử dụng câu trả lời mẫu trực tiếp từ Cơ sở dữ liệu.")
                final_answer = best_doc['answer']

        elif route == "RAG_CONFIRMATION":
            # Gợi ý cho người dùng xem có phải ý họ muốn hỏi câu trong DB không
            print(f"{C_YELLOW}[3. CONFIRMATION FLOW]{C_RESET} Phân tích độ mờ...")
            display_q = best_doc['question'][:80] + "..." if len(best_doc['question']) > 80 else best_doc['question']
            print(f"{C_YELLOW}Hệ thống: Có phải bạn muốn hỏi về: '{display_q}'?{C_RESET}")
            confirm = input(f"{C_BOLD}Xác nhận (y/n): {C_RESET}").strip().lower()
            
            if confirm in ['y', 'yes', '']:
                context = f"Câu hỏi: {best_doc['question']}\nTrả lời: {best_doc['answer']}"
                if self.generator.is_active:
                    final_answer = self.generator.generate(user_query, context, self.chat_history)
                else:
                    final_answer = best_doc['answer']
            else:
                # Nếu người dùng phủ nhận câu hỏi gợi ý, tự động chuyển sang Human-in-the-Loop
                print(f"{C_RED}Người dùng phủ nhận gợi ý. Chuyển tiếp sang luồng Human-in-the-Loop...{C_RESET}")
                final_answer = self.trigger_human_in_the_loop(user_query)

        elif route == "HUMAN_IN_THE_LOOP":
            final_answer = self.trigger_human_in_the_loop(user_query)

        elif route == "ADMIN_INTERFACE":
            print(f"{C_GREEN}[ADMIN INTERFACE] Chế độ cập nhật tri thức thủ công.{C_RESET}")
            q = input(f"{C_BOLD}Nhập câu hỏi cần định nghĩa mới: {C_RESET}").strip()
            a = input(f"{C_BOLD}Nhập câu trả lời tương ứng: {C_RESET}").strip()
            if q and a:
                new_doc = self.retriever.add_new_document(q, a)
                final_answer = f"Đã cập nhật thành công câu hỏi mới vào Cơ sở tri thức phòng lab! ID: {new_doc['id']}"
            else:
                final_answer = "Đã hủy cập nhật tri thức."

        # Cập nhật lịch sử chat (History)
        self.chat_history.append({"user": user_query, "bot": final_answer})
        return final_answer

    def trigger_human_in_the_loop(self, query):
        """Cơ chế Human-in-the-loop: Con người bổ sung tri thức, khép kín Bánh đà dữ liệu (Data Flywheel)"""
        print(f"\n{C_RED}[HUMAN-IN-THE-LOOP ACTIVATED]{C_RESET}")
        print("Chatbot không tìm thấy câu trả lời có độ tin cậy tối thiểu trong CSDL.")
        print(f"Câu hỏi của bạn: {C_BOLD}'{query}'{C_RESET}")
        
        expert_choice = input("Bạn có muốn đóng vai chuyên gia cập nhật tri thức mới này không? (y/n): ").strip().lower()
        if expert_choice in ['y', 'yes']:
            expert_answer = input(f"{C_GREEN}Nhập câu trả lời chuẩn (chuyên gia): {C_RESET}").strip()
            if expert_answer:
                # Ghi dữ liệu mới vào dataset (Flywheel quay)
                new_doc = self.retriever.add_new_document(query, expert_answer)
                print(f"{C_GREEN}[DATA FLYWHEEL] Đã tự động cập nhật dataset.json! Tri thức mới đã được ghi nhận.{C_RESET}")
                return f"Cảm ơn chuyên gia! Dữ liệu đã được nạp vào CSDL với ID {new_doc['id']}. Từ nay tôi đã có thể tự động trả lời câu hỏi này."
            else:
                return "Câu trả lời bị bỏ trống. Bỏ qua cập nhật."
        else:
            return "Xin lỗi, hiện tại tôi chưa được cập nhật dữ liệu để trả lời câu hỏi này của bạn."


def main():
    # Kích hoạt ANSI escape codes trên Windows CMD/PowerShell nếu cần thiết
    os.system("")
    
    # Cấu hình UTF-8 cho console để hiển thị tiếng Việt trên Windows tránh lỗi UnicodeEncodeError
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    chatbot = ChatbotSystem()
    
    # In tiêu đề thiết kế UI Terminal
    print("=" * 80)
    print(f" {C_BOLD}{C_GREEN}CHATBOT HỖ TRỢ PHÒNG LAB - THỰC HÀNH DATA FLYWHEEL & EXPERT SYSTEM & RAG{C_RESET}")
    print("=" * 80)
    print(f"Đường dẫn dự án: {C_CYAN}D:\\rag_project{C_RESET}")
    print(f"Trạng thái Gemini API SDK: {C_GREEN if HAS_GEMINI_SDK else C_RED}{'ĐÃ CÀI ĐẶT' if HAS_GEMINI_SDK else 'CHƯA CÀI ĐẶT'}{C_RESET}")
    print(f"API Key: {C_GREEN + 'Đã nạp qua biến môi trường' if chatbot.generator.is_active else C_YELLOW + 'Chạy Offline (Chưa nạp GEMINI_API_KEY)'}{C_RESET}")
    print(f"Tổng số bản ghi tri thức hiện tại: {C_CYAN}{len(chatbot.retriever.documents)}{C_RESET}")
    print("Mẹo: Nhập 'exit' để thoát, nhập 'admin' để truy cập giao diện cấu hình dữ liệu.")
    print("=" * 80)

    while True:
        try:
            user_input = input(f"\n{C_BOLD}User: {C_RESET}").strip()
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                print(f"\n{C_GREEN}Chào tạm biệt! Chúc bạn có buổi báo cáo lab thành công tốt đẹp!{C_RESET}")
                break
                
            bot_response = chatbot.handle_user_input(user_input)
            
            print(f"\n{C_BOLD}{C_GREEN}Bot:{C_RESET} {bot_response}")
            print("-" * 80)
            
        except KeyboardInterrupt:
            print(f"\n{C_GREEN}Chào tạm biệt!{C_RESET}")
            break

if __name__ == "__main__":
    main()
